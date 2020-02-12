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
batch_expert_params = DQNAgentParams()
# for pure random agent behavior, uncomment the below to set the epsilon scheduling accordingly
batch_expert_params.exploration_fraction=1.0      # explore for the whole timesteps
batch_expert_params.exploration_final_eps=1.0     # and always (i.e. with prob eps=1.0 ) explore



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
agent_params.exploration_final_eps = 0.1






##########################################################
# Experiment                                             #
##########################################################
experiment_params = ExperimentParams()
experiment_params.n_timesteps = 200     # its a batch agent so n_timesteps means n_epochs
experiment_params.env_params = env_params
experiment_params.agent_params = agent_params
experiment_params.batch_expert_params = batch_expert_params
experiment_params.batch_experience_buffer = None
experiment_params.batch_expert_n_timesteps = 1e5  # number of timesteps to train the expert before starting to rollout
experiment_params.batch_expert_n_episodes = 100  # number of episodes to rollout into the buffer
experiment_params.name = __name__.split('.')[-1]





