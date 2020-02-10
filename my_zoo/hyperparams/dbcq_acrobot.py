from my_zoo.hyperparams.default_config import DBCQAgentParams,DQNAgentParams,ExperimentParams, EnvParams
from stable_baselines.deepq import MlpPolicy
from zoo.utils import CustomDQNPolicy

##########################################################
# Env                                                    #
##########################################################
env_params = EnvParams()
env_params.env_id = 'acrobot'


##########################################################
# Batch Experience Agent                                 #
# Default values:
# policy = 'MlpPolicy'
# buffer_size = 50000
# learning_rate = 1e-4
# if learning_starts==buffer_size then pure random.
# learning_starts = 1000
# target_network_update_freq = 500
# train_freq = 1
# exploration_initial_eps = 1.0
# exploration_final_eps = 0.02
# exploration_fraction = 0.1
# prioritized_replay_alpha = 0.6
# prioritized_replay = False
# param_noise = False
# gamma = 0.99
# batch_size = 32
# double_q = True
# prioritized_replay_beta0 = 0.4
# prioritized_replay_beta_iters = None
# prioritized_replay_eps = 1e-6
# n_cpu_tf_sess = None
# policy_kwargs = None
##########################################################
batch_experience_agent_params = DQNAgentParams()
batch_experience_agent_params.learning_starts = batch_experience_agent_params.buffer_size   # for pure random experience



##########################################################
# Agent Params                                           #
# Default values:
# policy = 'MlpPolicy'  # or 'CnnPolicy' or 'CustomDQNPolicy'
# buffer_size = 50000
# learning_rate = 1e-4
# learning_starts = 1000
# target_network_update_freq = 500
# train_freq = 1
# exploration_initial_eps = 1.0
# exploration_final_eps = 0.02
# exploration_fraction = 0.1
# prioritized_replay_alpha = 0.6
# prioritized_replay = False
# param_noise = False
# gamma = 0.99
# batch_size = 32
# double_q = True
# prioritized_replay_beta0 = 0.4
# prioritized_replay_beta_iters = None
# prioritized_replay_eps = 1e-6
# n_cpu_tf_sess = None
# policy_kwargs = None
##########################################################
agent_params = DBCQAgentParams()
# here we can change the various parameters - for example, we can change the batch size
agent_params.policy = CustomDQNPolicy
agent_params.learning_rate = 1e-3
agent_params.exploration_final_eps= 0.1





##########################################################
# Experiment                                             #
##########################################################
experiment_params = ExperimentParams()
# for pure random agent n_timesteps=learning_starts
experiment_params.n_timesteps = batch_experience_agent_params.learning_starts   # pure random agent
experiment_params.env_params = env_params
experiment_params.agent_params = agent_params
experiment_params.batch_experience_agent_params = batch_experience_agent_params
experiment_params.batch_experience_buffer = None
experiment_params.name = __name__.split('.')[-1]




