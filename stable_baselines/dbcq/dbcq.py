from functools import partial

import tensorflow as tf
import numpy as np
import gym

from stable_baselines import logger
from stable_baselines.common import tf_util, OffPolicyRLModel, SetVerbosity, TensorboardWriter
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common.schedules import LinearSchedule
from stable_baselines.dbcq.build_graph import build_train
from stable_baselines.deepq.policies import DQNPolicy
from stable_baselines.gail import ExpertDataset
from stable_baselines.dbcq.replay_buffer import ExperienceDataset
from stable_baselines.common.evaluation import evaluate_policy as online_policy_eval


class DBCQ(OffPolicyRLModel):
    """
    The DBCQ model class.
    DQN paper: https://arxiv.org/abs/1312.5602
    Dueling DQN: https://arxiv.org/abs/1511.06581
    Double-Q Learning: https://arxiv.org/abs/1509.06461
    Prioritized Experience Replay: https://arxiv.org/abs/1511.05952
    Batch Constrained Q learning: https://arxiv.org/abs/1812.02900
    Discrete Batch Constrained Q learning: https://arxiv.org/abs/1910.01708


    :param policy: (DQNPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, LnMlpPolicy, ...)
    :param env: (Gym environment or str) The environment to evaluate on (if registered in Gym, can be str)
                Should be the same environment with which we created the buffer we learn from.
                if env=None and val_freq>0 we need to do OPE. currently its not supported.
    :param replay_buffer: (ReplayBuffer) - the buffer from which we'll learn
    :param gen_act_model: (DQNPolicy or str) the generative model that we'll learn. can also be 'knn'
            for k nearest neighbor
    :param gamma: (float) discount factor
    :param learning_rate: (float) learning rate for adam optimizer
    :param buffer_size: (int) size of the replay buffer
    :param val_freq: (int) perform validation every `val_freq` epochs. set to 0 to avoid validation.
    :param batch_size: (int) size of a batched sampled from replay buffer for training
    :param target_network_update_freq: (int) update the target network every `target_network_update_freq` steps.
    :param param_noise: (bool) Whether or not to apply noise to the parameters of the policy.
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        WARNING: this logging can take a lot of space quickly
    :param seed: (int) Seed for the pseudo-random generators (python, numpy, tensorflow).
        If None (default), use random seed. Note that if you want completely deterministic
        results, you must set `n_cpu_tf_sess` to 1.
    :param n_cpu_tf_sess: (int) The number of threads for TensorFlow operations
        If None, the number of cpu of the current machine will be used.
    """
    def __init__(self, policy, env, replay_buffer, gen_act_model=None ,gamma=0.99, learning_rate=5e-4,
                 val_freq=0, batch_size=32, target_network_update_freq=500,
                 buffer_train_fraction=0.8, gen_act_params = None,param_noise=False, act_distance_thresh=0.3,
                 n_cpu_tf_sess=None, verbose=0, tensorboard_log=None,
                 _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False, seed=None):

        super(DBCQ, self).__init__(policy=policy, env=env, replay_buffer=replay_buffer, verbose=verbose,
                                   policy_base=DQNPolicy, requires_vec_env=False, policy_kwargs=policy_kwargs,
                                   seed=seed, n_cpu_tf_sess=n_cpu_tf_sess)

        self.param_noise = param_noise
        self.val_freq = val_freq
        self.batch_size = batch_size
        self.target_network_update_freq = target_network_update_freq
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.buffer_train_fraction = buffer_train_fraction
        self.gen_act_params = gen_act_params
        self.act_distance_th = act_distance_thresh
        self.gen_act_model = gen_act_model

        if self.gen_act_model == 'NN':      # if using neural net for the generative model, use the same policy func
                                            # as the main policy
            self.gen_act_model = self.policy

        self.tensorboard_log = tensorboard_log
        self.full_tensorboard_log = full_tensorboard_log
        self.graph = None
        self.sess = None
        self._train_step = None
        self._gen_train_step = None
        self.step_model = None
        self.update_target = None
        self.act = None
        self.proba_step = None
        self.beta_schedule = None
        self.exploration = None
        self.params = None
        self.summary = None

        if _init_setup_model:
            self.setup_model()

        if self.env is None and self.val_freq>0:
            print('env is not provided. validation is skipped.')
            self.val_freq = 0       # currently we dont have Off Policy Evaluation so without env we skip validation

    def _get_pretrain_placeholders(self):
        policy = self.step_model
        return policy.obs_ph, tf.placeholder(tf.int32, [None]), policy.q_values

    def _get_gen_act_placeholders(self):
        policy = self.gen_act_model
        return policy.obs_ph, tf.placeholder(tf.int32, [None]), policy.q_values


    def setup_model(self):

        with SetVerbosity(self.verbose):
            assert not isinstance(self.action_space, gym.spaces.Box), \
                "Error: DBCQ cannot output a gym.spaces.Box action space."

            # If the policy is wrap in functool.partial (e.g. to disable dueling)
            # unwrap it to check the class type
            if isinstance(self.policy, partial):
                test_policy = self.policy.func
            else:
                test_policy = self.policy
            assert issubclass(test_policy, DQNPolicy), "Error: the input policy for the DBCQ model must be " \
                                                       "an instance of DQNPolicy."

            self.graph = tf.Graph()
            with self.graph.as_default():
                self.set_random_seed(self.seed)
                self.sess = tf_util.make_session(num_cpu=self.n_cpu_tf_sess, graph=self.graph)

                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                # build the training graph operations
                self.act, self._train_step, self.update_target, self.step_model, \
                self.gen_act_model,self._gen_train_step = build_train(
                    q_func=partial(self.policy, **self.policy_kwargs),
                    gen_act_policy=self.gen_act_model,
                    ob_space=self.observation_space,
                    ac_space=self.action_space,
                    optimizer=optimizer,
                    gamma=self.gamma,
                    grad_norm_clipping=10,
                    param_noise=self.param_noise,
                    sess=self.sess,
                    full_tensorboard_log=self.full_tensorboard_log,
                )
                self.proba_step = self.step_model.proba_step
                self.params = tf_util.get_trainable_vars("dbcq")
                self.gen_act_trainables = tf_util.get_globals_vars("dbcq/gen_act_model")

                # Initialize the parameters and copy them to the target network.
                tf_util.initialize(self.sess)
                self.update_target(sess=self.sess)

                self.summary = tf.summary.merge_all()

    def set_replay_buffer(self,replay_buffer):
        # should I deep copy it ?
        self.replay_buffer=replay_buffer

    def _setup_learn(self):
        # call base class to check we have an environment (currently needed for evaluation. in the future we'll do OPE
        # so we wont need an active env as long as we derive the action and observation space from the buffer)
        # make sure we have a replay buffer
        if self.replay_buffer is None:
            raise ValueError("Error: cannot train the BCQ model without a valid replay buffer"
                             "please set a buffer with set_replay_buffer(self,replay_buffer) method.")
        # wrap the replay buffer as ExperienceDataset
        self.dataset = ExperienceDataset(traj_data=self.replay_buffer,train_fraction=self.buffer_train_fraction,
                                         batch_size=self.batch_size,sequential_preprocessing=True)

    def train_gen_act_model(self,val_interval=None):
        # look at how the base model performs the pretraining. you should do the same here.
        # todo: according to the type of the generative model, set the parameters accordingly
        # currently we assume the generative model is neural network. need to add support in knn
        self.verbose = 1        # for debug
        if self.gen_act_params is None:
            n_epochs = 50
            lr=1e-3     # learning rate
            batch_size=64
            train_frac=0.7
        else:
            n_epochs = self.gen_act_params.get('n_epochs',50)
            lr = self.gen_act_params.get('lr',1e-3)
            batch_size = self.gen_act_params.get('batch_size',64)
            train_frac = self.gen_act_params.get('train_frac',0.7)

        if val_interval is None:
            # Prevent modulo by zero
            if n_epochs < 10:
                val_interval = 1
            else:
                val_interval = int(n_epochs / 10)
        dataset = ExpertDataset(traj_data=self.replay_buffer,train_fraction=train_frac,batch_size=batch_size,
                                sequential_preprocessing=True)
        with self.graph.as_default():
            with tf.variable_scope('gen_act_train'):
                obs_ph, actions_ph, actions_logits_ph = self._get_gen_act_placeholders()
                one_hot_actions = tf.one_hot(actions_ph, self.action_space.n)
                loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=actions_logits_ph,
                    labels=tf.stop_gradient(one_hot_actions)
                )
                loss = tf.reduce_mean(loss)
                optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-8)
                optim_op = optimizer.minimize(loss, var_list=self.gen_act_trainables)

            self.sess.run(tf.global_variables_initializer())

        if self.verbose > 0:
            print("Training generative model with Behavior Cloning...")

        for epoch_idx in range(int(n_epochs)):
            train_loss = 0.0
            # Full pass on the training set
            for _ in range(len(dataset.train_loader)):
                expert_obs, expert_actions = dataset.get_next_batch('train')
                feed_dict = {
                    obs_ph: expert_obs,
                    actions_ph: expert_actions,
                }
                train_loss_, _ = self.sess.run([loss, optim_op], feed_dict)
                train_loss += train_loss_

            train_loss /= len(dataset.train_loader)

            if self.verbose > 0 and (epoch_idx + 1) % val_interval == 0:
                val_loss = 0.0
                # Full pass on the validation set
                for _ in range(len(dataset.val_loader)):
                    expert_obs, expert_actions = dataset.get_next_batch('val')
                    val_loss_, = self.sess.run([loss], {obs_ph: expert_obs,
                                                        actions_ph: expert_actions})
                    val_loss += val_loss_

                val_loss /= len(dataset.val_loader)
                if self.verbose > 0:
                    print("==== Training progress {:.2f}% ====".format(100 * (epoch_idx + 1) / n_epochs))
                    print('Epoch {}'.format(epoch_idx + 1))
                    print("Training loss: {:.6f}, Validation loss: {:.6f}".format(train_loss, val_loss))
                    print()
            # Free memory
            del expert_obs, expert_actions
        return

    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="DBCQ",
              reset_num_timesteps=True, replay_wrapper=None):
        new_tb_log = self._init_num_timesteps(reset_num_timesteps)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
            self._setup_learn()
            print('training the generative model')
            self.train_gen_act_model()
            print('finished training the generative model')
            iter_cnt=0        # iterations counter
            ts=0
            epoch = 0   # epochs counter
            n_minibatches = len(self.dataset.train_loader)
            while ts < total_timesteps:
                if callback is not None:
                    # Only stop training if return value is False, not when it is None. This is for backwards
                    # compatibility with callbacks that have no return statement.
                    if callback(locals(), globals()) is False:
                        break
                # Full pass on the training set
                for _ in range(n_minibatches):
                    obses_t, actions, rewards, obses_tp1, dones = self.dataset.get_next_batch('train')
                    weights, batch_idxes = np.ones_like(rewards), None
                    if writer is not None:
                        # run loss backprop with summary, but once every 10 epochs save the metadata
                        # (memory, compute time, ...)
                        if (1 + epoch) % 10 == 0:
                            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                            run_metadata = tf.RunMetadata()
                            summary, loss = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1, obses_tp1,
                                                             dones, weights, self.act_distance_th , sess=self.sess,
                                                             options=run_options,run_metadata=run_metadata)
                            writer.add_run_metadata(run_metadata, 'step%d' % self.num_timesteps)
                        else:
                            summary, loss = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1, obses_tp1,
                                                             dones, weights, self.act_distance_th, sess=self.sess)
                        writer.add_summary(summary, self.num_timesteps)
                    else:
                        _, loss = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1, obses_tp1,
                                                   dones, weights, self.act_distance_th, sess=self.sess)
                    # update counters
                    self.num_timesteps += len(obses_t)
                    iter_cnt+=1
                    ts += len(obses_t)
                epoch += 1     # inc
                # finished going through the data. summarize the epoch:
                avg_epoch_loss = loss/n_minibatches
                if epoch % self.target_network_update_freq == 0:
                    # Update target network periodically.
                    self.update_target(sess=self.sess)

                if self.val_freq>0 and ((epoch+1) % self.val_freq) == 0:
                    if self.env is not None:
                        mean_reward,_ = online_policy_eval(self,self.env)
                        print("Evaluating on env: mean reward={0}".format(mean_reward))
                    else:
                        raise RuntimeError("Off Policy Evaluation is not supported yet")

                if self.verbose >= 1 and log_interval is not None:
                    logger.record_tabular("epoch", epoch)
                    logger.record_tabular("epoch_loss", avg_epoch_loss)
                    logger.record_tabular("mean 100 episode reward", mean_reward)
                    logger.dump_tabular()

        return self

    def predict(self, observation, state=None, mask=None, deterministic=True):
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        with self.sess.as_default():
            actions, _, _ = self.step_model.step(observation, deterministic=deterministic)

        if not vectorized_env:
            actions = actions[0]

        return actions, None

    def action_probability(self, observation, state=None, mask=None, actions=None, logp=False):
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        actions_proba = self.proba_step(observation, state, mask)

        if actions is not None:  # comparing the action distribution, to given actions
            actions = np.array([actions])
            assert isinstance(self.action_space, gym.spaces.Discrete)
            actions = actions.reshape((-1,))
            assert observation.shape[0] == actions.shape[0], "Error: batch sizes differ for actions and observations."
            actions_proba = actions_proba[np.arange(actions.shape[0]), actions]
            # normalize action proba shape
            actions_proba = actions_proba.reshape((-1, 1))
            if logp:
                actions_proba = np.log(actions_proba)

        if not vectorized_env:
            if state is not None:
                raise ValueError("Error: The environment must be vectorized when using recurrent policies.")
            actions_proba = actions_proba[0]

        return actions_proba

    def get_parameter_list(self):
        return self.params

    def save(self, save_path, cloudpickle=False):
        # params
        data = {
            "param_noise": self.param_noise,
            "val_freq": self.val_freq,
            "batch_size": self.batch_size,
            "target_network_update_freq": self.target_network_update_freq,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            # "gen_act_model": self.gen_act_model,      # todo: uncommenting this line cause error. check it.
            "gen_act_params": self.gen_act_params,
            "buffer_train_fraction": self.buffer_train_fraction,
            # variables saved by parent class
            "verbose": self.verbose,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "policy": self.policy,
            "n_envs": self.n_envs,
            "n_cpu_tf_sess": self.n_cpu_tf_sess,
            "seed": self.seed,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs
        }

        params_to_save = self.get_parameters()
        # params_to_save = None
        # data = None

        self._save_to_file(save_path, data=data, params=params_to_save, cloudpickle=cloudpickle)
