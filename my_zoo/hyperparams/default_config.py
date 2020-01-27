import os
import logging

ALGO_IDS = ['a2c','acer','acktr','dqn','ddpg','her','sac','trpo','td3','ppo']
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
           'acrobot':'Acrobot-v0',
           'lunland':'LunarLander-v2'
           }


class DQNAgentParams:
    def __init__(self):
        # Default parameters for DQN Agent
        # can be accessed as dict by vars(DQNAgentParams)
        self.algorithm = 'dqn'
        self.gamma = 0.99
        self.learning_rate = 5e-4
        self.buffer_size = 50000
        self.exploration_fraction = 0.1
        self.exploration_final_eps = 0.02
        self.exploration_initial_eps = 1.0
        self.train_freq = 1
        self.batch_size = 32
        self.double_q = True
        self.learning_starts = 1000
        self.target_network_update_freq = 500
        self.prioritized_replay = False,
        self.prioritized_replay_alpha = 0.6
        self.prioritized_replay_beta0 = 0.4
        self.prioritized_replay_beta_iters = None
        self.prioritized_replay_eps = 1e-6
        self.param_noise = False
        self.n_cpu_tf_sess = None
        self.verbose = 0
        self.tensorboard_log = None
        self._init_setup_model = True
        self.policy_kwargs = None
        self.full_tensorboard_log = False
        self.seed = None
    def as_dict(self):
        return vars(self)



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
        self.env_id = 'cartpole'

        ###### Agent #######
        self.agent = 'dqn'
        self.agent_params = DQNAgentParams()
        self.n_timesteps = 1e5


        ###### Hyper Parameters Optimization ######
        self.optimize = False
        self.n_trials = 10
        self.n_jobs = 1
        self.hpopt_sampler = 'tpe'
        self.hpopt_pruner = 'median'





    def as_dict(self):
        return vars(self)





