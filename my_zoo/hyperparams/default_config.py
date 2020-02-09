import os
import logging
import numpy as np

ALGO_IDS = ['a2c','acer','dqn','ddpg','sac','td3','ppo2']
# 'a2c': A2C,
# 'acer': ACER,
# 'acktr': ACKTR,
# 'dqn': DQN,
# 'ddpg': DDPG,
# 'her': HER,
# 'sac': SAC,
# 'trpo': TRPO,
# 'td3': TD3
# 'ppo2': PPO2

CC_ENVS = {'cartpole':'CartPole-v0',
           'cartpole1':'CartPole-v1',
           'mntcar':'MountainCar-v0',
           'acrobot':'Acrobot-v1',
           'lunland':'LunarLander-v2'
           }

class EnvParams:
    def __init__(self,env_id='cartpole'):
        self.env_id=env_id
        # self.normalize = False      # can also be "{'norm_obs': True, 'norm_reward': False}"
        # consider dropping normalize and use the below directly
        self.norm_obs = False
        self.norm_reward = False

        self.env_wrapper = None     # see utils.wrappers
        self.frame_stack = 1        # 1 = no stack , >1 means how many frames to stack

    def as_dict(self):
        return vars(self)


#############################
# Agents Defaults

class AgentParams:
    def __init__(self):

        # these are parameters that should be derived from the experiment params
        self.verbose = 0
        self.tensorboard_log = None
        self._init_setup_model = True
        self.full_tensorboard_log = False
        self.seed = None
        return

    def as_dict(self):
        return vars(self)


class DQNAgentParams(AgentParams):
    """
    Parameters for DQN agent
    The agent gets the following values in its construction:
    policy,env
    gamma = 0.99, learning_rate = 5e-4, buffer_size = 50000, exploration_fraction = 0.1,
    exploration_final_eps = 0.02, exploration_initial_eps = 1.0, train_freq = 1, batch_size = 32, double_q = True,
    learning_starts = 1000, target_network_update_freq = 500, prioritized_replay = False,
    prioritized_replay_alpha = 0.6, prioritized_replay_beta0 = 0.4, prioritized_replay_beta_iters = None,
    prioritized_replay_eps = 1e-6, param_noise = False,

    n_cpu_tf_sess = None, verbose = 0, tensorboard_log = None, _init_setup_model = True, policy_kwargs = None,
    full_tensorboard_log = False, seed = None
    """
    def __init__(self):
        super(DQNAgentParams, self).__init__()
        # Default parameters for DQN Agent
        self.algorithm = 'dqn'

        self.policy = 'MlpPolicy'    # or 'CnnPolicy' or 'CustomDQNPolicy'
        self.buffer_size = 50000
        self.learning_rate = 1e-4
        self.learning_starts = 1000
        self.target_network_update_freq = 500
        self.train_freq = 1
        self.exploration_initial_eps = 1.0
        self.exploration_final_eps = 0.02
        self.exploration_fraction = 0.1
        self.prioritized_replay_alpha = 0.6
        self.prioritized_replay = False
        self.param_noise = False

        # other default params
        self.gamma = 0.99
        self.batch_size = 32
        self.double_q = True
        self.prioritized_replay_beta0 = 0.4
        self.prioritized_replay_beta_iters = None
        self.prioritized_replay_eps = 1e-6
        self.n_cpu_tf_sess = None
        self.policy_kwargs = None
        return


class PPO2AgentParams(AgentParams):
    """
    PPO2AgentParams
    gamma = 0.99, n_steps = 128, ent_coef = 0.01, learning_rate = 2.5e-4, vf_coef = 0.5,
    max_grad_norm = 0.5, lam = 0.95, nminibatches = 4, noptepochs = 4, cliprange = 0.2, cliprange_vf = None,
    verbose = 0, tensorboard_log = None, _init_setup_model = True, policy_kwargs = None,
    full_tensorboard_log = False, seed = None, n_cpu_tf_sess = None
    """
    def __init__(self):
        super(PPO2AgentParams, self).__init__()
        self.algorithm = 'ppo2'


        self.policy = 'MlpPolicy'
        self.n_steps = 128
        self.nminibatches = 4
        self.lam = 0.95
        self.gamma = 0.99
        self.noptepochs = 4
        self.ent_coef = 0.0
        self.learning_rate = 2.5e-4   # can also be 'lin_<float>' e.g. 'lin_0.001'
        self.cliprange = 0.2        # can also be 'lin_<float>' e.g. 'lin_0.1'
        self.vf_coef = 0.5
        self.ent_coef = 0.01
        self.cliprange_vf = None
        self.max_grad_norm = 0.5
        self.policy_kwargs = None
        self.n_cpu_tf_sess = None
        return


class SACAgentParams(AgentParams):
    """
    SACAgentParams:
    policy, env,
    gamma=0.99, learning_rate=3e-4, buffer_size=50000,
    learning_starts=100, train_freq=1, batch_size=64,
    tau=0.005, ent_coef='auto', target_update_interval=1,
    gradient_steps=1, target_entropy='auto', action_noise=None,
    random_exploration=0.0, verbose=0, tensorboard_log=None,
    _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False,
    seed=None, n_cpu_tf_sess=None
    """
    def __init__(self):
        super(SACAgentParams, self).__init__()
        # parameters to be parsed and removed at runtime
        self.algorithm='sac'
        self.noise_type = 'ornstein-uhlenbeck'
        self.noise_std = 0.5

        self.policy = 'MlpPolicy'       # see also 'CustomSACPolicy'
        self.learning_rate = 3e-4       # can also be 'lin_3e-4'
        self.buffer_size = 50000
        self.batch_size = 64
        self.ent_coef = 'auto'
        self.train_freq = 1
        self.gradient_steps = 1
        self.learning_starts = 100

        # other algorithm defaults
        self.gamma = 0.99
        self.tau = 0.005
        self.target_update_interval = 1
        self.target_entropy = 'auto'
        self.action_noise = None
        self.random_exploration = 0.0
        self.policy_kwargs = None,
        self.n_cpu_tf_sess = None
        return


class DDPGAgentParams(AgentParams):
    """
    DDPGAgentParams
    policy,env,
    gamma=0.99, memory_policy=None, eval_env=None, nb_train_steps=50,
    nb_rollout_steps=100, nb_eval_steps=100, param_noise=None, action_noise=None,
    normalize_observations=False, tau=0.001, batch_size=128, param_noise_adaption_interval=50,
    normalize_returns=False, enable_popart=False, observation_range=(-5., 5.), critic_l2_reg=0.,
    return_range=(-np.inf, np.inf), actor_lr=1e-4, critic_lr=1e-3, clip_norm=None, reward_scale=1.,
    render=False, render_eval=False, memory_limit=None, buffer_size=50000, random_exploration=0.0,
    verbose=0, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None,
    full_tensorboard_log=False, seed=None, n_cpu_tf_sess=1
    """
    def __init__(self):
        super(DDPGAgentParams, self).__init__()
        self.algorithm='ddpg'
        # parameters to be parsed at runtime
        self.noise_type = 'ornstein-uhlenbeck'
        self.noise_std = 0.5

        self.policy = 'MlpPolicy'
        self.memory_limit = 5000
        self.normalize_observations = False
        self.normalize_returns = False
        self.gamma = 0.99
        self.actor_lr = 1e-4
        self.critic_lr = 1e-3
        self.batch_size = 128
        self.random_exploration = 0.0
        self.policy_kwargs = None

        # other algorithm defaults
        self.memory_policy = None
        self.eval_env = None
        self.nb_train_steps = 50,
        self.nb_rollout_steps = 100
        self.nb_eval_steps = 100
        self.param_noise = None
        self.action_noise = None
        self.tau = 0.001
        self.param_noise_adaption_interval = 50
        self.enable_popart = False
        self.observation_range = (-5., 5.)
        self.critic_l2_reg = 0.,
        self.return_range = (-np.inf, np.inf)
        self.clip_norm = None
        self.reward_scale = 1.
        self.render = False
        self.render_eval = False
        self.memory_limit = None
        self.buffer_size = 50000
        self.n_cpu_tf_sess = 1
        return


class A2CAgentParams(AgentParams):
    """
    A2CAgentParams
    policy,env,
    gamma=0.99, n_steps=5, vf_coef=0.25, ent_coef=0.01, max_grad_norm=0.5,
    learning_rate=7e-4, alpha=0.99, epsilon=1e-5, lr_schedule='constant', verbose=0,
    tensorboard_log=None, _init_setup_model=True, policy_kwargs=None,
    full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None
    """
    def __init__(self):
        super(A2CAgentParams, self).__init__()
        self.algorithm='a2c'

        self.policy = 'MlpPolicy'
        self.lr_schedule = 'constant'           # can also be 'linear'
        self.ent_coef = 0.01
        self.gamma = 0.99
        self.n_steps = 5
        self.learning_rate = 7e-4
        self.vf_coef = 0.25

        # other defaults
        self.max_grad_norm = 0.5
        self.alpha = 0.99
        self.epsilon = 1e-5
        self.lr_schedule = 'constant'
        self.policy_kwargs = None
        self.n_cpu_tf_sess = None
        return


class ACERAgentParams(AgentParams):
    """
    ACERAgentParams

    policy,env,
    gamma=0.99, n_steps=20, num_procs=None, q_coef=0.5, ent_coef=0.01, max_grad_norm=10,
    learning_rate=7e-4, lr_schedule='linear', rprop_alpha=0.99, rprop_epsilon=1e-5, buffer_size=5000,
    replay_ratio=4, replay_start=1000, correction_term=10.0, trust_region=True,
    alpha=0.99, delta=1, verbose=0, tensorboard_log=None,
    _init_setup_model=True, policy_kwargs=None,
    full_tensorboard_log=False, seed=None, n_cpu_tf_sess=1
    """
    def __init__(self):
        super(ACERAgentParams, self).__init__()
        self.algorithm='acer'
        self.n_timesteps = 1e7

        self.policy = 'MlpPolicy'       # 'CnnPolicy'
        self.lr_schedule = 'linear'
        self.buffer_size = 5000
        self.ent_coef = 0.01
        self.gamma = 0.99

        # other defaults
        self.n_steps = 20
        self.num_procs = None
        self.q_coef = 0.5
        self.max_grad_norm = 10
        self.learning_rate = 7e-4
        self.rprop_alpha = 0.99
        self.rprop_epsilon = 1e-5
        self.replay_ratio = 4
        self.replay_start = 1000
        self.correction_term = 10.0
        self.trust_region = True
        self.alpha = 0.99
        self.delta = 1
        self.n_cpu_tf_sess = 1
        self.policy_kwargs = None
        return


class TD3AgentParams(AgentParams):
    """
    TD3AgentParams
    policy,env,
    gamma=0.99, learning_rate=3e-4, buffer_size=50000,
    learning_starts=100, train_freq=100, gradient_steps=100, batch_size=128,
    tau=0.005, policy_delay=2, action_noise=None,
    target_policy_noise=0.2, target_noise_clip=0.5,
    random_exploration=0.0, verbose=0, tensorboard_log=None,
    _init_setup_model=True, policy_kwargs=None,
    full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None
    """
    def __init__(self):
        super(TD3AgentParams, self).__init__()
        # parameters to be parsed at runtime and removed
        self.algorithm='td3'
        self.noise_type = 'ornstein-uhlenbeck'
        self.noise_std = 0.5

        # defaults
        self.policy = 'MlpPolicy'
        self.learning_starts = 100
        self.batch_size = 128
        self.gamma = 0.99
        self.learning_rate = 3e-4
        self.buffer_size = 50000
        self.gradient_steps = 100
        self.policy_kwargs = None
        self.train_freq = 100

        self.tau = 0.005
        self.policy_delay = 2
        self.action_noise = {'noise_type':None,'noise_std':None}
        self.target_policy_noise = 0.2
        self.target_noise_clip = 0.5
        self.random_exploration = 0.0
        self.n_cpu_tf_sess = None
        return


class DBCQAgentParams(AgentParams):
    """
    Parameters for DQN agent
    The agent gets the following values in its construction:
    policy,env
    gamma = 0.99, learning_rate = 5e-4, buffer_size = 50000, exploration_fraction = 0.1,
    exploration_final_eps = 0.02, exploration_initial_eps = 1.0, train_freq = 1, batch_size = 32, double_q = True,
    learning_starts = 1000, target_network_update_freq = 500, prioritized_replay = False,
    prioritized_replay_alpha = 0.6, prioritized_replay_beta0 = 0.4, prioritized_replay_beta_iters = None,
    prioritized_replay_eps = 1e-6, param_noise = False,

    n_cpu_tf_sess = None, verbose = 0, tensorboard_log = None, _init_setup_model = True, policy_kwargs = None,
    full_tensorboard_log = False, seed = None
    """
    def __init__(self):
        super(DBCQAgentParams, self).__init__()
        # Default parameters for DQN Agent
        self.algorithm = 'dbcq'

        self.policy = 'MlpPolicy'    # or 'CnnPolicy' or 'CustomDQNPolicy'
        self.buffer_size = 50000
        self.learning_rate = 1e-4
        self.learning_starts = 1000
        self.target_network_update_freq = 500
        self.train_freq = 1
        self.exploration_initial_eps = 1.0
        self.exploration_final_eps = 0.02
        self.exploration_fraction = 0.1
        self.prioritized_replay_alpha = 0.6
        self.prioritized_replay = False
        self.param_noise = False

        # other default params
        self.gamma = 0.99
        self.batch_size = 32
        self.double_q = True
        self.prioritized_replay_beta0 = 0.4
        self.prioritized_replay_beta_iters = None
        self.prioritized_replay_eps = 1e-6
        self.n_cpu_tf_sess = None
        self.policy_kwargs = None
        return


#################################
# Experiment Params
class ExperimentParams:
    def __init__(self):
        self.name=None      # should be overriden
        self.seed = 1
        ####### Folders #######
        self.output_root_dir = os.path.join(os.path.expanduser('~'),'share','Data','MLA','stbl','results')

        ####### Logging #######
        self.log_level = logging.INFO
        # use the %(process)d format to log the process-ID (useful in multiprocessing where each process has a different ID)
        # LOG_FORMAT = '%(asctime)s | %(process)d | %(message)s'
        self.log_format = '%(asctime)s | %(message)s'
        self.log_date_format = '%y-%m-%d %H:%M:%S'
        self.verbose = 1

        ####### Env #######
        self.n_envs = 1
        self.env_params = None

        ###### Agent #######
        self.trained_agent=None
        self.agent_params = None        # should be class of agent params e.g. DQNAgentParams()
        # training params
        self.n_timesteps = 1e5
        self.log_interval = -1  # using algorithm default

        ###### BatchRL #######
        self.batch_experience_agent_params = None       # determines how the experience buffer is created
                                                        # None = no creation of buffer --> load from file
        self.batch_experience_buffer = None     # path to experience buffer we'll learn from
                                                # name template: experience_<env-id>_<agent-id>.npy
        self.batch_n_epochs = 200               # number of epochs to train
        self.online_evaluation = True           # whether to use evaluation environment
        self.offline_evaluation_split = 0.0     # if >0 perform offline evaluation on this fraction of experience
                                                # e.g. if 0.2, train on 80%, evaluate on 20%





        ###### Hyper Parameters Optimization ######
        self.hp_optimize = False
        self.n_trials = 10
        self.n_jobs = 1
        self.hpopt_sampler = 'tpe'
        self.hpopt_pruner = 'median'

    def as_dict(self):
        return vars(self)





