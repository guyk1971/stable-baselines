from functools import partial

import tensorflow as tf
import numpy as np
from scipy.special import softmax
import gym

from stable_baselines import logger
from stable_baselines.common import tf_util, OffPolicyRLModel, SetVerbosity, TensorboardWriter
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common.schedules import LinearSchedule
from stable_baselines.common.buffers import ReplayBuffer, PrioritizedReplayBuffer
from stable_baselines.common.dataset import ExperienceDataset       # batch_rl mode
from stable_baselines.qrdqn.build_graph import build_train
from stable_baselines.qrdqn.policies import QRDQNPolicy
from stable_baselines.deepq.policies import MlpPolicy       # for the reward model
from tqdm import tqdm


class QRDQN(OffPolicyRLModel):
    """
    The DQN model class.
    DQN paper: https://arxiv.org/abs/1312.5602
    Dueling DQN: https://arxiv.org/abs/1511.06581
    Double-Q Learning: https://arxiv.org/abs/1509.06461
    Prioritized Experience Replay: https://arxiv.org/abs/1511.05952
    Quantile Regression DQN paper: https://arxiv.org/abs/1710.10044
    Batch mode RL -
    replay_buffer is not supposed to be provided to constructor unless we work in batch mode.
    in batch mode we learn only from the buffer and use the environment only for evaluation.
    batch mode does not support all modes. see comments in the code.

    :param policy: (DQNPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, LnMlpPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param n_atoms: (int) The number of atoms (quantiles) in the distribution approximation
    :param gamma: (float) discount factor
    :param learning_rate: (float) learning rate for adam optimizer
    :param buffer_size: (int) size of the replay buffer
    :param exploration_fraction: (float) fraction of entire training period over which the exploration rate is
            annealed
    :param exploration_final_eps: (float) final value of random action probability
    :param exploration_initial_eps: (float) initial value of random action probability
    :param train_freq: (int) update the model every `train_freq` steps. set to None to disable printing
    :param batch_size: (int) size of a batched sampled from replay buffer for training
    :param double_q: (bool) Whether to enable Double-Q learning or not.
    :param learning_starts: (int) how many steps of the model to collect transitions for before learning starts
    :param target_network_update_freq: (int) update the target network every `target_network_update_freq` steps.
    :param prioritized_replay: (bool) if True prioritized replay buffer will be used.
    :param prioritized_replay_alpha: (float)alpha parameter for prioritized replay buffer.
        It determines how much prioritization is used, with alpha=0 corresponding to the uniform case.
    :param prioritized_replay_beta0: (float) initial value of beta for prioritized replay buffer
    :param prioritized_replay_beta_iters: (int) number of iterations over which beta will be annealed from initial
            value to 1.0. If set to None equals to max_timesteps.
    :param prioritized_replay_eps: (float) epsilon to add to the TD errors when updating priorities.
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
    def __init__(self, policy, env, replay_buffer=None, n_atoms=50, gamma=0.99, learning_rate=5e-4, buffer_size=50000, exploration_fraction=0.1,
                 exploration_final_eps=0.02, exploration_initial_eps=1.0, train_freq=1, batch_size=32,
                 double_q=False, learning_starts=1000, target_network_update_freq=500, buffer_train_fraction=1.0,
                 prioritized_replay=False,prioritized_replay_alpha=0.6, prioritized_replay_beta0=0.4,
                 prioritized_replay_beta_iters=None, prioritized_replay_eps=1e-6, param_noise=False,
                 n_cpu_tf_sess=None, verbose=0, tensorboard_log=None,
                 _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False, seed=None):

        super(QRDQN, self).__init__(policy=policy, env=env, replay_buffer=replay_buffer, verbose=verbose, policy_base=QRDQNPolicy,
                                  requires_vec_env=False, policy_kwargs=policy_kwargs, seed=seed, n_cpu_tf_sess=n_cpu_tf_sess)

        self.param_noise = param_noise
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.prioritized_replay = prioritized_replay
        self.prioritized_replay_eps = prioritized_replay_eps
        self.batch_size = batch_size
        self.target_network_update_freq = target_network_update_freq
        self.prioritized_replay_alpha = prioritized_replay_alpha
        self.prioritized_replay_beta0 = prioritized_replay_beta0
        self.prioritized_replay_beta_iters = prioritized_replay_beta_iters
        self.exploration_final_eps = exploration_final_eps
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_fraction = exploration_fraction
        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.buffer_train_fraction = buffer_train_fraction      # ope - how much of the buffer will be used to train
                                                                # the reward model vs. to evaluate the policy
                                                                # this is obsolete
        self.tensorboard_log = tensorboard_log
        self.full_tensorboard_log = full_tensorboard_log
        self.double_q = double_q
        self.n_atoms = n_atoms


        self.graph = None
        self.sess = None
        self._train_step = None
        self.step_model = None
        self.ope_reward_model = None
        self.update_target = None
        self.act = None
        self.proba_step = None
        self.beta_schedule = None
        self.exploration = None
        self.params = None
        self.summary = None

        self.batch_rl_mode = True if replay_buffer else False

        if _init_setup_model:
            self.setup_model()



    def _get_pretrain_placeholders(self):
        # this function is used to assist in initializing the model pre-training with behavioral cloning
        # how to update the quantiles? all quantiles get the mean value ?
        policy = self.step_model
        return policy.obs_ph, tf.placeholder(tf.int32, [None]), policy.q_values

    def pretrain(self, dataset, n_epochs=10, learning_rate=1e-4, adam_epsilon=1e-8, val_interval=None):
        # super(QRDQN,self).pretrain(dataset, n_epochs=n_epochs, learning_rate=learning_rate,
        #          adam_epsilon=adam_epsilon, val_interval=val_interval)
        raise NotImplementedError("QRDQN pretrain not supported")
        # return self

    def _get_ope_reward_placeholders(self):
        # obs, action,reward,q_values
        policy = self.ope_reward_model
        return policy.obs_ph, tf.placeholder(tf.int32, [None]), tf.placeholder(tf.float32, [None]), policy.q_values


    def _setup_learn(self):
        # call base class to check we have an environment (currently needed for evaluation. in the future we'll do OPE
        # so we wont need an active env as long as we derive the action and observation space from the buffer)
        super(QRDQN, self)._setup_learn()

        # make sure we have a replay buffer
        # wrap the replay buffer as ExperienceDataset
        if self.batch_rl_mode:
            self.dataset = ExperienceDataset(traj_data=self.replay_buffer,train_fraction=self.buffer_train_fraction,
                                             batch_size=self.batch_size,sequential_preprocessing=True)


    def setup_model(self):
        with SetVerbosity(self.verbose):
            assert not isinstance(self.action_space, gym.spaces.Box), \
                "Error: DQN cannot output a gym.spaces.Box action space."
            assert not isinstance(self.action_space, gym.spaces.MultiDiscrete), \
                "Error: QRDQN cannot output a gym.spaces.MultiDiscrete action space."
            assert not self.param_noise, \
                "Error: QRDQN currently doesnt support Parameter Space Noise for Exploration"

            if self.prioritized_replay:
                logger.warn("qrdqn cant work with prioritied replay buffer --> disabling priority buffer")
                self.prioritized_replay = False

            reward_model = None
            if self.batch_rl_mode:
                reward_model = partial(MlpPolicy, dueling=False, **self.policy_kwargs)


            # If the policy is wrap in functool.partial (e.g. to disable dueling)
            # unwrap it to check the class type
            if isinstance(self.policy, partial):
                test_policy = self.policy.func
            else:
                test_policy = self.policy
            assert issubclass(test_policy, QRDQNPolicy), "Error: the input policy for the QRDQN model must be " \
                                                       "an instance of QRDQNPolicy."

            n_atoms = self.policy_kwargs.get('n_atoms',self.n_atoms)
            if n_atoms != self.n_atoms:
                logger.warn('n_atoms parameter conflict. overriding with constructor value {0}'.format(self.n_atoms))
            self.policy_kwargs['n_atoms']=self.n_atoms



            self.graph = tf.Graph()
            with self.graph.as_default():
                self.set_random_seed(self.seed)
                self.sess = tf_util.make_session(num_cpu=self.n_cpu_tf_sess, graph=self.graph)

                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

                self.act, self._train_step, self.update_target, self.step_model,self.ope_reward_model,_ = build_train(
                    q_func=partial(self.policy, **self.policy_kwargs),
                    ob_space=self.observation_space,
                    ac_space=self.action_space,
                    optimizer=optimizer,
                    gamma=self.gamma,
                    grad_norm_clipping=10,
                    param_noise=self.param_noise,
                    sess=self.sess,
                    reward_model=reward_model,
                    full_tensorboard_log=self.full_tensorboard_log,
                    double_q=self.double_q
                )
                self.proba_step = self.step_model.proba_step
                self.params = tf_util.get_trainable_vars("qrdqn")
                if self.batch_rl_mode:
                    self.ope_reward_trainables = tf_util.get_trainable_vars("qrdqn/ope_reward_model")

                # Initialize the parameters and copy them to the target network.
                tf_util.initialize(self.sess)
                self.update_target(sess=self.sess)

                self.summary = tf.summary.merge_all()

    def _train_ope_rew_model(self,val_interval=None):
        n_epochs = 50
        lr=1e-3     # learning rate
        batch_size=64
        train_frac=0.7

        if val_interval is None:
            # Prevent modulo by zero
            if n_epochs < 10:
                val_interval = 1
            else:
                val_interval = int(n_epochs / 10)
        dataset = ExperienceDataset(traj_data=self.replay_buffer,train_fraction=train_frac,batch_size=batch_size,
                                sequential_preprocessing=True)
        with self.graph.as_default():
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

            self.sess.run(tf.global_variables_initializer())

        if self.verbose > 0:
            logger.info("Training reward model with regression loss")

        for epoch_idx in range(int(n_epochs)):
            rew_epoch_loss = 0.0
            # Full pass on the training set
            for _ in range(len(dataset.train_loader)):
                expert_obs, expert_actions,expert_rewards,_,_,_ = dataset.get_next_batch('train')
                feed_dict = {rew_obs_ph: expert_obs,
                             rew_actions_ph: expert_actions,
                             rewards_ph: expert_rewards}
                rew_batch_loss,_ = self.sess.run([rew_loss,rew_optim_op],feed_dict)
                rew_epoch_loss += rew_batch_loss

            rew_epoch_loss /= len(dataset.train_loader)

            if self.verbose > 0 and (epoch_idx + 1) % val_interval == 0:
                rew_val_loss = 0.0
                # Full pass on the validation set
                for _ in range(len(dataset.val_loader)):
                    expert_obs, expert_actions,expert_rewards,_,_,_ = dataset.get_next_batch('val')
                    feed_dict = {rew_obs_ph: expert_obs,
                                 rew_actions_ph: expert_actions,
                                 rewards_ph: expert_rewards}
                    rew_batch_loss= self.sess.run(rew_loss, feed_dict)
                    rew_val_loss += rew_batch_loss
                rew_val_loss /= len(dataset.val_loader)
                if self.verbose > 0:
                    logger.info("================ Pre-Training epoch {0}/{1}: =============\n".format(epoch_idx,n_epochs))
                    logger.info(
                        "Reward Model: train loss {0:.6f}, val loss {1:.6f} ===\n".format(rew_epoch_loss, rew_val_loss))
            # Free memory
            del expert_obs, expert_actions, expert_rewards
        return


    def _learn_from_static_buffer(self,total_timesteps,writer, callback=None, log_interval=10, tb_log_name="DQN"):
        iter_cnt = 0  # iterations counter
        ts = 0
        epoch = 0  # epochs counter
        mean_reward = None
        last_updadte_target_ts = 0
        n_minibatches = len(self.dataset.train_loader)
        callback.on_training_start(locals(), globals())
        while ts < total_timesteps:
            # Full pass on the training set
            # frac = 1.0 - ts / total_timesteps
            # lr_now = self.learning_rate(frac)  # get the learning rate for the current epoch
            lr_now=self.learning_rate
            tot_epoch_loss=0
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
                        summary, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1,
                                                              dones, weights, sess=self.sess, options=run_options,
                                                              run_metadata=run_metadata)
                        writer.add_run_metadata(run_metadata, 'step%d' % self.num_timesteps)
                    else:
                        summary, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1,
                                                              dones, weights, sess=self.sess)
                    writer.add_summary(summary, self.num_timesteps)
                else:
                    _, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1, dones, weights,
                                                    sess=self.sess)
                # update counters
                self.num_timesteps += len(obses_t)
                iter_cnt += 1
                ts += len(obses_t)
                tot_epoch_loss += np.mean(td_errors)
                # in batch rl, the step is training step done on a minibatch of samples.
                if callback.on_step() is False:
                    break
            epoch += 1  # inc
            # finished going through the data. summarize the epoch:
            avg_epoch_loss = tot_epoch_loss / n_minibatches
            # should_update_target = ((ts-last_updadte_target_ts)>self.target_network_update_freq)
            should_update_target = ((epoch % self.target_network_update_freq) == 0)
            if should_update_target:
                # Update target network periodically.
                self.update_target(sess=self.sess)
                last_updadte_target_ts = ts


            if self.verbose >= 1 and log_interval is not None:
                logger.record_tabular("time steps", ts)
                logger.record_tabular("epoch", epoch)
                logger.record_tabular("epoch_loss", avg_epoch_loss)
                logger.record_tabular("lr ", lr_now)
                logger.dump_tabular()



    def learn(self, total_timesteps, callback=None, log_interval=10, tb_log_name="QRDQN",
              reset_num_timesteps=True, replay_wrapper=None):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        callback = self._init_callback(callback)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
            self._setup_learn()

            if self.batch_rl_mode:
                logger.info('training the reward model for off policy evaluation')
                self._train_ope_rew_model(val_interval=1)
                logger.info('finished training the reward model')
            else:
                # Create the replay buffer
                if self.prioritized_replay:
                    self.replay_buffer = PrioritizedReplayBuffer(self.buffer_size, alpha=self.prioritized_replay_alpha)
                    if self.prioritized_replay_beta_iters is None:
                        prioritized_replay_beta_iters = total_timesteps
                    else:
                        prioritized_replay_beta_iters = self.prioritized_replay_beta_iters
                    self.beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                                        initial_p=self.prioritized_replay_beta0,
                                                        final_p=1.0)
                else:
                    self.replay_buffer = ReplayBuffer(self.buffer_size)
                    self.beta_schedule = None

                if replay_wrapper is not None:
                    assert not self.prioritized_replay, "Prioritized replay buffer is not supported by HER"
                    self.replay_buffer = replay_wrapper(self.replay_buffer)

                # Create the schedule for exploration starting from 1.
                self.exploration = LinearSchedule(schedule_timesteps=int(self.exploration_fraction * total_timesteps),
                                                  initial_p=self.exploration_initial_eps,
                                                  final_p=self.exploration_final_eps)

                episode_rewards = [0.0]
                episode_successes = []

            callback.on_training_start(locals(), globals())

            if self.batch_rl_mode:
                self._learn_from_static_buffer(total_timesteps,writer, callback,log_interval,tb_log_name)
            else:
                callback.on_rollout_start()
                reset = True
                obs = self.env.reset()
                # Retrieve unnormalized observation for saving into the buffer
                if self._vec_normalize_env is not None:
                    obs_ = self._vec_normalize_env.get_original_obs().squeeze()

                for _ in tqdm(range(total_timesteps)):
                    # Take action and update exploration to the newest value
                    kwargs = {}
                    if not self.param_noise:
                        update_eps = self.exploration.value(self.num_timesteps)
                        update_param_noise_threshold = 0.
                    else:
                        update_eps = 0.
                        # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                        # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                        # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                        # for detailed explanation.
                        update_param_noise_threshold = \
                            -np.log(1. - self.exploration.value(self.num_timesteps) +
                                    self.exploration.value(self.num_timesteps) / float(self.env.action_space.n))
                        kwargs['reset'] = reset
                        kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                        kwargs['update_param_noise_scale'] = True
                    with self.sess.as_default():
                        action = self.act(np.array(obs)[None], update_eps=update_eps, **kwargs)[0]
                    env_action = action
                    reset = False
                    new_obs, rew, done, info = self.env.step(env_action)

                    self.num_timesteps += 1

                    # Stop training if return value is False
                    if callback.on_step() is False:
                        break

                    # Store only the unnormalized version
                    if self._vec_normalize_env is not None:
                        new_obs_ = self._vec_normalize_env.get_original_obs().squeeze()
                        reward_ = self._vec_normalize_env.get_original_reward().squeeze()
                    else:
                        # Avoid changing the original ones
                        obs_, new_obs_, reward_ = obs, new_obs, rew
                    # Store transition in the replay buffer.
                    self.replay_buffer.add(obs_, action, reward_, new_obs_, float(done))
                    obs = new_obs
                    # Save the unnormalized observation
                    if self._vec_normalize_env is not None:
                        obs_ = new_obs_

                    if writer is not None:
                        ep_rew = np.array([reward_]).reshape((1, -1))
                        ep_done = np.array([done]).reshape((1, -1))
                        tf_util.total_episode_reward_logger(self.episode_reward, ep_rew, ep_done, writer,
                                                            self.num_timesteps)

                    episode_rewards[-1] += reward_
                    if done:
                        maybe_is_success = info.get('is_success')
                        if maybe_is_success is not None:
                            episode_successes.append(float(maybe_is_success))
                        if not isinstance(self.env, VecEnv):
                            obs = self.env.reset()
                        episode_rewards.append(0.0)
                        reset = True

                    # Do not train if the warmup phase is not over
                    # or if there are not enough samples in the replay buffer
                    can_sample = self.replay_buffer.can_sample(self.batch_size)
                    if can_sample and self.num_timesteps > self.learning_starts \
                            and self.num_timesteps % self.train_freq == 0:

                        callback.on_rollout_end()
                        # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                        # pytype:disable=bad-unpacking
                        if self.prioritized_replay:
                            assert self.beta_schedule is not None, \
                                   "BUG: should be LinearSchedule when self.prioritized_replay True"
                            experience = self.replay_buffer.sample(self.batch_size,
                                                                   beta=self.beta_schedule.value(self.num_timesteps),
                                                                   env=self._vec_normalize_env)
                            (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                        else:
                            obses_t, actions, rewards, obses_tp1, dones = self.replay_buffer.sample(self.batch_size,
                                                                                                    env=self._vec_normalize_env)
                            weights, batch_idxes = np.ones_like(rewards), None
                        # pytype:enable=bad-unpacking

                        if writer is not None:
                            # run loss backprop with summary, but once every 100 steps save the metadata
                            # (memory, compute time, ...)
                            if (1 + self.num_timesteps) % 100 == 0:
                                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                                run_metadata = tf.RunMetadata()
                                summary, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1,
                                                                      dones, weights, sess=self.sess, options=run_options,
                                                                      run_metadata=run_metadata)
                                writer.add_run_metadata(run_metadata, 'step%d' % self.num_timesteps)
                            else:
                                summary, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1,
                                                                      dones, weights, sess=self.sess)
                            writer.add_summary(summary, self.num_timesteps)
                        else:
                            _, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1, dones, weights,
                                                            sess=self.sess)

                        if self.prioritized_replay:
                            new_priorities = np.abs(td_errors) + self.prioritized_replay_eps
                            assert isinstance(self.replay_buffer, PrioritizedReplayBuffer)
                            self.replay_buffer.update_priorities(batch_idxes, new_priorities)

                        callback.on_rollout_start()

                    if can_sample and self.num_timesteps > self.learning_starts and \
                            self.num_timesteps % self.target_network_update_freq == 0:
                        # Update target network periodically.
                        self.update_target(sess=self.sess)

                    if len(episode_rewards[-101:-1]) == 0:
                        mean_100ep_reward = -np.inf
                    else:
                        mean_100ep_reward = round(float(np.mean(episode_rewards[-101:-1])), 1)

                    num_episodes = len(episode_rewards)
                    if self.verbose >= 1 and done and log_interval is not None and len(episode_rewards) % log_interval == 0:
                        logger.record_tabular("steps", self.num_timesteps)
                        logger.record_tabular("episodes", num_episodes)
                        if len(episode_successes) > 0:
                            logger.logkv("success rate", np.mean(episode_successes[-100:]))
                        logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                        logger.record_tabular("% time spent exploring",
                                              int(100 * self.exploration.value(self.num_timesteps)))
                        logger.dump_tabular()

        callback.on_training_end()
        return self

    def predict(self, observation, state=None, mask=None, deterministic=True,with_prob=False):
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        with self.sess.as_default():
            actions, q_values, _ = self.step_model.step(observation, deterministic=deterministic)

        if not vectorized_env:
            actions = actions[0]
            q_values = q_values[0]
        if with_prob:
            actions_prob=softmax(q_values)
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
            "n_atoms": self.n_atoms,
            "double_q": self.double_q,
            "param_noise": self.param_noise,
            "learning_starts": self.learning_starts,
            "train_freq": self.train_freq,
            "prioritized_replay": self.prioritized_replay,
            "prioritized_replay_eps": self.prioritized_replay_eps,
            "batch_size": self.batch_size,
            "target_network_update_freq": self.target_network_update_freq,
            "prioritized_replay_alpha": self.prioritized_replay_alpha,
            "prioritized_replay_beta0": self.prioritized_replay_beta0,
            "prioritized_replay_beta_iters": self.prioritized_replay_beta_iters,
            "exploration_final_eps": self.exploration_final_eps,
            "exploration_fraction": self.exploration_fraction,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
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

        self._save_to_file(save_path, data=data, params=params_to_save, cloudpickle=cloudpickle)