from functools import partial

import tensorflow as tf
import numpy as np
from scipy.special import softmax
import gym

from stable_baselines import logger
from stable_baselines.common import tf_util, OffPolicyRLModel, SetVerbosity, TensorboardWriter
from stable_baselines.dbcq.build_graph import build_train
from stable_baselines.deepq.policies import DQNPolicy
from stable_baselines.common.dataset import ExperienceDataset
from stable_baselines.common.schedules import get_schedule_fn


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
                Note: the algorithm builds its model based on env.observation_space and env.action_space so the env
                must be provided even if not used.
    :param replay_buffer: (ReplayBuffer) - the buffer from which we'll learn
    :param gen_act_model: (str) the generative model that we'll learn.
    :param gamma: (float) discount factor
    :param learning_rate: (float) learning rate for adam optimizer
    :param buffer_size: (int) size of the replay buffer
    :param val_freq: (int) perform validation every `val_freq` epochs. set to 0 to avoid validation.
    :param batch_size: (int) size of a batched sampled from replay buffer for training
    :param target_network_update_freq: (int) update the target network every `target_network_update_freq` steps.
    :param buffer_train_fraction: (float) how to split the experience buffer (relevant when using OPE)
    :param gen_act_params: (dict) dictionary that defines how to build the generative model
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
    def __init__(self, policy, env, replay_buffer=None, gen_act_policy=None ,gamma=0.99, learning_rate=5e-4,
                 batch_size=32, target_network_update_freq=500,buffer_train_fraction=1.0,
                 gen_act_params = None,gen_train_with_main=False,param_noise=False, act_distance_thresh=0.3,
                 n_cpu_tf_sess=None, verbose=0, tensorboard_log=None,
                 _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False, seed=None):

        super(DBCQ, self).__init__(policy=policy, env=env, replay_buffer=replay_buffer, verbose=verbose,
                                   policy_base=DQNPolicy, requires_vec_env=False, policy_kwargs=policy_kwargs,
                                   seed=seed, n_cpu_tf_sess=n_cpu_tf_sess)

        self.param_noise = param_noise
        self.batch_size = batch_size
        self.target_network_update_freq = target_network_update_freq
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.buffer_train_fraction = buffer_train_fraction      # ope - how much of the buffer will be used to train
                                                                # the reward model vs. to evaluate the policy
                                                                # this is obsolete
        self.gen_act_params = gen_act_params
        self.act_distance_th = act_distance_thresh
        self.gen_act_policy = gen_act_policy
        self.gen_train_with_main = gen_train_with_main

        self.tensorboard_log = tensorboard_log
        self.full_tensorboard_log = full_tensorboard_log
        self.graph = None
        self.sess = None
        self._train_step = None
        self.step_model = None
        self.gen_act_model = None
        self.ope_reward_model = None
        self.update_target = None
        self.act = None
        self.proba_step = None
        self.beta_schedule = None
        self.exploration = None
        self.params = None
        self.gen_act_trainables = None
        self.summary = None

        if _init_setup_model:
            self.setup_model()

    def _get_pretrain_placeholders(self):
        policy = self.step_model
        return policy.obs_ph, tf.placeholder(tf.int32, [None]), policy.q_values

    def _get_gen_act_placeholders(self):
        policy = self.gen_act_model
        return policy.obs_ph, tf.placeholder(tf.int32, [None]), policy.q_values

    def _get_ope_reward_placeholders(self):
        # obs, action,reward,q_values
        policy = self.ope_reward_model
        return policy.obs_ph, tf.placeholder(tf.int32, [None]), tf.placeholder(tf.float32, [None]), policy.q_values

    def setup_model(self):

        with SetVerbosity(self.verbose):
            assert not isinstance(self.action_space, gym.spaces.Box), \
                "Error: DBCQ cannot output a gym.spaces.Box action space."
            if self.gen_act_policy is None:
                if self.gen_act_params['type'] == 'NN':
                    # if using neural net for the generative model, use the same policy func
                    # as the main policy
                    self.gen_act_policy = partial(self.policy, **self.policy_kwargs)
                else:
                    raise TypeError('K nearest neighbor is not yet supported in DBCQ')

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
                # if lr_scheduling: comment out the following. it will be defined inside the build_graph using
                # the placeholder
                # optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                # build the training graph operations
                self.act, self._train_step, self.update_target, self.step_model,self.gen_act_model, \
                self.ope_reward_model = build_train(
                    q_func=partial(self.policy, **self.policy_kwargs),
                    gen_act_policy=self.gen_act_policy,
                    ob_space=self.observation_space,
                    ac_space=self.action_space,
                    reward_model= partial(self.policy, **self.policy_kwargs),
                    gamma=self.gamma,
                    grad_norm_clipping=10,
                    gen_train_with_main=self.gen_train_with_main,
                    param_noise=self.param_noise,
                    sess=self.sess,
                    full_tensorboard_log=self.full_tensorboard_log,
                )
                self.proba_step = self.step_model.proba_step
                self.params = tf_util.get_trainable_vars("dbcq")
                self.gen_act_trainables = tf_util.get_trainable_vars("dbcq/gen_act_model")
                self.ope_reward_trainables = tf_util.get_trainable_vars("dbcq/ope_reward_model")
                # Initialize the parameters and copy them to the target network.
                tf_util.initialize(self.sess)
                self.update_target(sess=self.sess)

                self.summary = tf.summary.merge_all()

    def set_replay_buffer(self,replay_buffer):
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

    def _train_gen_act_model(self,val_interval=None):
        # look at how the base model performs the pretraining. you should do the same here.
        # currently we assume the generative model is neural network.
        # todo: add support in knn

        # Note : here we train all models involved in batch rl
        #   - generative model for DBCQ
        #   - reward model for OPE

        self.verbose = 1        # for debug
        if self.gen_act_params is None:
            gen_n_epochs = 50
            lr=1e-3     # learning rate
            batch_size=64
            train_frac=0.7
        else:
            gen_n_epochs = self.gen_act_params.get('n_epochs',50)
            lr = self.gen_act_params.get('lr',1e-3)
            batch_size = self.gen_act_params.get('batch_size',64)
            train_frac = self.gen_act_params.get('train_frac',0.7)
        rew_n_epochs=50
        n_epochs = max(rew_n_epochs,gen_n_epochs)
        if val_interval is None:
            # Prevent modulo by zero
            if n_epochs < 10:
                val_interval = 1
            else:
                val_interval = int(n_epochs / 10)
        dataset = ExperienceDataset(traj_data=self.replay_buffer,train_fraction=train_frac,batch_size=batch_size,
                                sequential_preprocessing=True)
        with self.graph.as_default():
            # build the graph for generative model
            with tf.variable_scope('gen_act_pretrain'):
                gen_obs_ph, gen_actions_ph, gen_actions_logits_ph = self._get_gen_act_placeholders()
                gen_one_hot_actions = tf.one_hot(gen_actions_ph, self.action_space.n)
                gen_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=gen_actions_logits_ph,
                    labels=tf.stop_gradient(gen_one_hot_actions)
                )
                gen_loss = tf.reduce_mean(gen_loss)
                optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-8)
                gen_optim_op = optimizer.minimize(gen_loss, var_list=self.gen_act_trainables)
                tf.summary.scalar('gen_pretrain_loss',gen_loss)
            # build the graph for reward model (for off policy evaluation)
            with tf.variable_scope('ope_reward_pretrain'):
                rew_obs_ph, rew_actions_ph, rewards_ph, model_rewards_ph = self._get_ope_reward_placeholders()
                one_hot_actions = tf.one_hot(rew_actions_ph, self.action_space.n)
                target_rewards = tf.where(one_hot_actions>0,tf.tile(rewards_ph[:,tf.newaxis],[1,self.action_space.n]),
                                          model_rewards_ph)
                rew_loss = tf_util.mse(model_rewards_ph,tf.stop_gradient(target_rewards))
                rew_optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-8)
                rew_optim_op = rew_optimizer.minimize(rew_loss, var_list=self.ope_reward_trainables)
                tf.summary.scalar('rew_pretrain_loss', rew_loss)
            # pretrain_summary=tf.summary.merge_all()
            self.sess.run(tf.global_variables_initializer())

        if self.verbose > 0:
            logger.info("Training reward model with regression loss and generative model with Behavior Cloning ...")

        for epoch_idx in range(int(n_epochs)):
            gen_epoch_loss = 0.0
            rew_epoch_loss = 0.0
            # Full pass on the training set
            for _ in range(len(dataset.train_loader)):
                expert_obs, expert_actions,expert_rewards,_,_,_ = dataset.get_next_batch('train')
                feed_dict = {gen_obs_ph: expert_obs,
                             gen_actions_ph: expert_actions,
                             rew_obs_ph: expert_obs,
                             rew_actions_ph: expert_actions,
                             rewards_ph: expert_rewards}

                # gen_batch_loss, _, rew_batch_loss,_ = self.sess.run([gen_loss, gen_optim_op,rew_loss,rew_optim_op],
                #                                                     feed_dict)
                if epoch_idx<gen_n_epochs:
                    gen_batch_loss, _ = self.sess.run([gen_loss, gen_optim_op],feed_dict)
                    gen_epoch_loss += gen_batch_loss

                if epoch_idx<rew_n_epochs:
                    rew_batch_loss,_ = self.sess.run([rew_loss,rew_optim_op],feed_dict)
                    rew_epoch_loss += rew_batch_loss

            gen_epoch_loss /= len(dataset.train_loader)
            rew_epoch_loss /= len(dataset.train_loader)

            if self.verbose > 0 and (epoch_idx + 1) % val_interval == 0:
                gen_val_loss = 0.0
                rew_val_loss = 0.0
                # Full pass on the validation set
                for _ in range(len(dataset.val_loader)):
                    expert_obs, expert_actions,expert_rewards,_,_,_ = dataset.get_next_batch('val')
                    feed_dict = {gen_obs_ph: expert_obs,
                                 gen_actions_ph: expert_actions,
                                 rew_obs_ph: expert_obs,
                                 rew_actions_ph: expert_actions,
                                 rewards_ph: expert_rewards}

                    # gen_batch_loss, rew_batch_loss= self.sess.run([gen_loss,rew_loss], feed_dict)
                    if epoch_idx<gen_n_epochs:
                        gen_batch_loss = self.sess.run(gen_loss, feed_dict)
                        gen_val_loss += gen_batch_loss

                    if epoch_idx<rew_n_epochs:
                        rew_batch_loss = self.sess.run(rew_loss, feed_dict)
                        rew_val_loss += rew_batch_loss

                gen_val_loss /= len(dataset.val_loader)
                rew_val_loss /= len(dataset.val_loader)
                if self.verbose > 0:
                    logger.info("================ Pre-Training epoch {0}/{1}: =============\n".format(epoch_idx,n_epochs))
                    if epoch_idx<gen_n_epochs:
                        logger.info("Gen Model: train loss {0:.6f}, val loss {1:.6f} ===\n".format(gen_epoch_loss,gen_val_loss))
                    if epoch_idx<rew_n_epochs:
                        logger.info(
                            "Reward Model: train loss {0:.6f}, val loss {1:.6f} ===\n".format(rew_epoch_loss, rew_val_loss))
            # Free memory
            del expert_obs, expert_actions, expert_rewards
        return

    def learn(self, total_timesteps, callback=None, log_interval=10, tb_log_name="DBCQ",
              reset_num_timesteps=True, replay_wrapper=None):

        # Transform to callable if needed
        self.learning_rate = get_schedule_fn(self.learning_rate)

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        callback = self._init_callback(callback)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
            self._setup_learn()
            logger.info('training the reward model and generative model')
            self._train_gen_act_model(val_interval=1)
            logger.info('finished training the reward model and generative model')
            iter_cnt=0        # iterations counter
            ts=0
            epoch = 0   # epochs counter
            mean_reward = None
            last_updadte_target_ts = 0
            n_minibatches = len(self.dataset.train_loader)
            callback.on_training_start(locals(),globals())
            while ts < total_timesteps:
                # Full pass on the training set
                frac = 1.0 - ts/total_timesteps
                lr_now = self.learning_rate(frac)         # get the learning rate for the current epoch
                tot_epoch_loss={'main':0,'gen':0}
                for _ in range(n_minibatches):
                    obses_t, actions, rewards, obses_tp1, dones,_ = self.dataset.get_next_batch('train')
                    weights, batch_idxes = np.ones_like(rewards), None
                    # if lr_scheduling set the learning rate here and send it in the _train_step
                    if writer is not None:
                        # run loss backprop with summary, but once every 10 epochs save the metadata
                        # (memory, compute time, ...)
                        if (1 + epoch) % 10 == 0:
                            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                            run_metadata = tf.RunMetadata()
                            summary, losses = self._train_step(obses_t, obses_t, actions, rewards, obses_tp1, obses_tp1,
                                                             obses_tp1,dones, weights, self.act_distance_th ,lr_now,
                                                             sess=self.sess,options=run_options,run_metadata=run_metadata)
                            writer.add_run_metadata(run_metadata, 'step%d' % self.num_timesteps)
                        else:
                            summary, losses = self._train_step(obses_t, obses_t,actions, rewards, obses_tp1, obses_tp1,
                                                               obses_tp1,dones, weights, self.act_distance_th,lr_now,
                                                               sess=self.sess)
                        writer.add_summary(summary, self.num_timesteps)
                    else:
                        _, losses = self._train_step(obses_t, obses_t, actions, rewards, obses_tp1, obses_tp1, obses_tp1,
                                                   dones, weights, self.act_distance_th, lr_now, sess=self.sess)
                    # update counters
                    self.num_timesteps += len(obses_t)
                    iter_cnt+=1
                    ts += len(obses_t)
                    for k in losses.keys():
                        tot_epoch_loss[k] += losses[k]
                    # in DBCQ, the step is training step done on a minibatch of samples.
                    if callback.on_step() is False:
                        break
                epoch += 1     # inc
                # finished going through the data. summarize the epoch:
                avg_epoch_loss = tot_epoch_loss['main']/n_minibatches
                avg_gen_loss = tot_epoch_loss['gen']/n_minibatches if self.gen_train_with_main else None
                # should_update_target = ((ts-last_updadte_target_ts)>self.target_network_update_freq)
                should_update_target = ((epoch % self.target_network_update_freq) ==0)
                if should_update_target:
                    # Update target network periodically.
                    self.update_target(sess=self.sess)
                    last_updadte_target_ts = ts

                if self.verbose >= 1 and log_interval is not None:
                    logger.record_tabular("time steps", ts)
                    logger.record_tabular("epoch", epoch)
                    logger.record_tabular("epoch_loss", avg_epoch_loss)
                    logger.record_tabular("lr ",lr_now)
                    if avg_gen_loss is not None:
                        logger.record_tabular("gen_loss",avg_gen_loss)
                    logger.dump_tabular()
        callback.on_training_end()
        return self

    def predict(self, observation, state=None, mask=None, deterministic=True, with_prob=False):
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        with self.sess.as_default():
            actions, q_values, _ = self.step_model.step(observation, deterministic=deterministic)

        if not vectorized_env:
            actions = actions[0]
            q_values = q_values[0]

        if with_prob:
            actions_prob = softmax(q_values,axis=-1)
            return actions,None,actions_prob
        else:
            return actions, None

    def predict_ope_rewards(self,observation, deterministic=True):
        observation = np.array(observation)
        observation = observation.reshape((-1,) + self.observation_space.shape)
        with self.sess.as_default():
            _, rewards, _ = self.ope_reward_model.step(observation, deterministic=deterministic)
        return rewards

    def predict_ope_qvalues(self, observation, deterministic=True):
        observation = np.array(observation)
        observation = observation.reshape((-1,) + self.observation_space.shape)
        with self.sess.as_default():
            _, q_values, _ = self.step_model.step(observation, deterministic=deterministic)

        actions_prob = softmax(q_values,axis=-1)
        return q_values,actions_prob



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
            "algorithm":'dbcq',
            "param_noise": self.param_noise,
            "batch_size": self.batch_size,
            "target_network_update_freq": self.target_network_update_freq,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "gen_act_policy": self.gen_act_policy,
            "gen_act_params": self.gen_act_params,
            "buffer_train_fraction": self.buffer_train_fraction,
            "gen_train_with_main": self.gen_train_with_main,
            # variables saved by parent class
            "verbose": self.verbose,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "policy": self.policy,
            "n_envs": self.n_envs,
            "n_cpu_tf_sess": self.n_cpu_tf_sess,
            "seed": self.seed,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs,
            "policy_name": self.__class__.__name__ + self.policy.__name__
        }

        params_to_save = self.get_parameters()
        # params_to_save = None
        # data = None

        self._save_to_file(save_path, data=data, params=params_to_save, cloudpickle=cloudpickle)
