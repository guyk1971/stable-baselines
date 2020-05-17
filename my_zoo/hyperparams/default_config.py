import os
import logging
import numpy as np
from my_zoo.my_envs import BENCHMARKS,PLATFORMS
ALGO_IDS = ['a2c','acer','dqn','ddpg','sac','td3','ppo2','dbcq']
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


class DTTEnvSimParams(EnvParams):
    def __init__(self):
        super(EnvParams, self).__init__()
        self.env_id = 'DTTSim'
        self.workload = [BENCHMARKS['cb15'],BENCHMARKS['cooldown']]
        self.platform = PLATFORMS['Scarlet']
        self.norm_obs = True
        self.log_output = None

#############################
# Agents Defaults

class AgentParams:
    def __init__(self):

        # these are parameters that should be derived from the experiment params
        self.verbose = 1
        self.tensorboard_log = None
        self._init_setup_model = True
        self.full_tensorboard_log = False
        self.seed = None
        return

    def as_dict(self):
        return vars(self)

class RandomAgentParams(AgentParams):
    def __init__(self):
        super(RandomAgentParams, self).__init__()
        self.algorithm = 'random'
        return

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

        # batch_rl defaults:
        self.buffer_train_fraction = 1.0        # 80% will be used for training the policy and the reward model for DM


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


class QRDQNAgentParams(AgentParams):
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
        super(QRDQNAgentParams, self).__init__()
        # Default parameters for DQN Agent
        self.algorithm = 'qrdqn'

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
        self.n_atoms = 50

        # batch_rl defaults:
        self.buffer_train_fraction = 1.0        # 80% will be used for training the policy and the reward model for DM

        # other default params
        self.gamma = 0.99
        self.batch_size = 32
        self.double_q = False
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
        self.policy = 'MlpPolicy'    # or 'CnnPolicy' or 'CustomDQNPolicy' - the main policy that we train
        self.learning_rate = 1e-4               # can also be 'lin_<float>' e.g. 'lin_0.001'
        self.target_network_update_freq = 1   # number of epochs between target network updates
        self.param_noise = False
        self.act_distance_thresh = 0.3          # if gen_act_policy is Neural Net - corresponds to the threshold tau
                                                # i.e. actions with likelihood ratio larger than threshold will be
                                                # considered as candidates
                                                # if gen_act_policy is KNN - the max distance from nearest neighbor
                                                # s.t. actions that are farther will be thrown
        # other default params
        self.gamma = 0.99
        self.batch_size = 32
        self.buffer_train_fraction = 1.0        # 100% will be used for training the policy and the reward model for DM
                                                # the rest (20%) will be used for Off policy evaluation
        # parameters of the generative model for actions
        self.gen_act_policy = None               # 'KNN' for K nearest neighbors, 'NN' for Neural Net
                                                # if 'NN' the agent will use the same type of policy for the generative model
        self.gen_act_params = {'type': 'NN', 'n_epochs': 50, 'lr': 1e-3, 'train_frac': 0.7, 'batch_size': 64}
        # self.gen_act_params = {'type':'KNN','size': 1000}  # knn parameters
        self.gen_train_with_main = False        # if True, continue to train the generative model while training the
                                                # main agent
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


        ############### Agent ################
        # pretrain with behavioral cloning
        # (pretrain_dataset,pretrain_expert_agent) are related as follows:
        # dataset = None, expert_agent = None : no pretraing
        # dataset != None, expert_agent = None : pretrain from existing buffer (behavioral cloning)
        # dataset = None, expert_agent != None : Illegal option. agent must have path to save
        # dataset != None, expert_agent != None : use expert to write to pretrain_dataset and then pretrain


        # we can either load a trained model and continue train it using the agent_params


        # given we do pretraining, use the following for pretrain (behavioral cloning)
        self.experience_dataset = None          # path to experience buffer that can be wrapped by ExpertData
                                                # if None, there's no pre-train
        self.pretrain_epochs = 0                # number of epochs to train on the expert data (as behavioral cloning)
        self.pretrain_lr = 1e-4

        ########################
        # trained agent - if we want to continue training from a saved agent
        self.trained_agent_model_file=None         # path to main pretrained agent - to continue training
        self.agent_params = None        # agent that trains the main policy.
                                        # should be class of agent params e.g. DQNAgentParams()
        # training params
        self.n_timesteps = 1e5          # number of timesteps to train main policy
        self.log_interval = -1          # using algorithm default

        self.evaluation_freq = 0        # evaluate on eval env and/or with ope every this number of timesteps
        self.online_eval_n_episodes = 10

        self.off_policy_eval_dataset_eval_fraction = 0      # fraction of the data that will be used for evaluation
                                                            # the rest will be used for training the agent and the reward model

        ###### BatchRL #######
        self.expert_model_file = None           # path to expert to generate experience for batch rl (if experience_dataset is not provided)
                                                # if not None, will load it to generate the buffer
                                                # currently not supported. SHOULD BE 'None' !
        self.expert_params = None               # parameters for further training the expert
                                                # can be any AgentParams from above (assuming coherency in obs,act)
        self.train_expert_n_timesteps = 0      # number of timesteps to train the expert before starting to record

        self.expert_steps_to_record = 0        # number of episodes to record into the experience buffer






        ###### Hyper Parameters Optimization ######
        self.hp_optimize = False
        self.n_trials = 10
        self.n_jobs = 1
        self.hpopt_sampler = 'tpe'
        self.hpopt_pruner = 'median'

    def as_dict(self):
        return vars(self)





