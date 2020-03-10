from my_zoo.hyperparams.default_config import *
# from zoo.utils import CustomDQNPolicy
from stable_baselines.deepq import MlpPolicy

##########################################################
# Env                                                    #
#   'cartpole':'CartPole-v0',
#   'cartpole1':'CartPole-v1',
#   'mntcar':'MountainCar-v0',
#   'acrobot':'Acrobot-v1',
#   'lunland':'LunarLander-v2'
##########################################################
env_params = EnvParams()
env_params.env_id = 'mntcar'


##########################################################
# Batch Experience-Generating Agent                                 #
# Default values for DQNAgentParams:
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
batch_expert_params = RandomAgentParams()
# for pure random agent behavior, uncomment the below to set the epsilon scheduling accordingly
# batch_expert_params = DQNAgentParams()
# batch_expert_params.exploration_fraction=1.0      # explore for the whole timesteps
# batch_expert_params.exploration_final_eps=0.01     # and always (i.e. with prob eps=1.0 ) explore
# batch_expert_params.double_q = True
# batch_expert_params.batch_size = 128





##########################################################
# Agent Params                                           #
# Default values:
# policy = 'MlpPolicy'  # or 'CnnPolicy' or 'CustomDQNPolicy'
# gen_act_model = 'NN'
# learning_rate = 1e-4
# learning_starts = 1000
# target_network_update_freq = 500
# train_freq = 1
# val_freq = 0
# prioritized_replay_alpha = 0.6
# prioritized_replay = False
# param_noise = False
# gamma = 0.99
# batch_size = 32
# prioritized_replay_beta0 = 0.4
# prioritized_replay_beta_iters = None
# prioritized_replay_eps = 1e-6
# n_cpu_tf_sess = None
# policy_kwargs = None
##########################################################
agent_params = DBCQAgentParams()
# here we can change the various parameters - for example, we can change the batch size
agent_params.policy = MlpPolicy
agent_params.verbose = 1
agent_params.learning_rate = 1e-4
agent_params.policy_kwargs = {'dueling':False,'layers': [256, 512]}
agent_params.target_network_update_freq = 1         # every 1 epoch
agent_params.val_freq = 1               # every 1 epoch
agent_params.batch_size = 128
agent_params.buffer_train_fraction = 1.0         # currently online evaluation. use all buffer for training
agent_params.gen_act_params['lr'] = 1e-4
agent_params.gen_act_params['batch_size'] = 128






##########################################################
# Experiment                                             #
##########################################################
experiment_params = ExperimentParams()
experiment_params.n_timesteps = int(5e6)
experiment_params.env_params = env_params
experiment_params.agent_params = agent_params
experiment_params.batch_expert_params = batch_expert_params
experiment_params.batch_experience_buffer = None
experiment_params.batch_expert_n_timesteps = int(1e5)       # n_timesteps to train the expert before starting to rollout
                                                            # not relevant for random
experiment_params.batch_expert_steps_to_record = 100000      # number of steps to rollout into the buffer
experiment_params.name = __name__.split('.')[-1]





