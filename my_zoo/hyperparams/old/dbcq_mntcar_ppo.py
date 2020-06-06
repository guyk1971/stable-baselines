from my_zoo.hyperparams.default_config import *
# from zoo.utils import CustomDQNPolicy
from stable_baselines.dbcq import MlpPolicy

##########################################################
# Scenario :
# loading an experience buffer (that was created while interacting with the environment) instead of generating it
# from scratch
# given this buffer, we can decide whether we want to train an a dbcq agent from scratch or to load a pretrained model.
# if we load a pretrained model, do we also want to continue to train the generative model ? or train it from scratch ?




##########################################################
# Env                                                    #
##########################################################
env_params = EnvParams()
env_params.env_id = 'mntcar'


##########################################################
# Batch Experience-Generating Agent
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
# the following 3 parameters are mutually exclusive
# batch_experience_buffer = '/home/gkoren2/share/Data/MLA/stbl/results/dbcq_acrobot-27-02-2020_15-50-55/1/er_acrobot_random.npz'
# batch_experience_buffer = '/home/gkoren2/share/Data/MLA/stbl/results/dbcq_acrobot_load_expert-27-02-2020_17-47-19/1/er_acrobot_dbcq.npz'
batch_experience_buffer = None
batch_experience_trained_agent = None       # agent to load to generate experience
# batch_expert_params = None      # this is also the default. if not None, we'll train the agent prior to recording
# batch_expert_params = RandomAgentParams()
# for a non-random agent, we need to define an agent and set its training parameters.
batch_expert_params = PPO2AgentParams()
batch_expert_params.learning_rate = 1e-3
batch_expert_params.n_steps = 16
batch_expert_params.nminibatches = 1
batch_expert_params.lam = 0.98
batch_expert_params.ent_coef=0.0
batch_expert_params.verbose = 0






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
trained_agent = None            # if we want to continue train an agent, set the path to saved model
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
experiment_params.name = __name__.split('.')[-1]
experiment_params.env_params = env_params

#####################
# experience generation
#
# the following 3 parameters are mutually exclusive - see above
experiment_params.batch_experience_buffer = batch_experience_buffer
experiment_params.batch_experience_trained_agent = batch_experience_trained_agent
experiment_params.batch_expert_params = batch_expert_params
# if generating experience: (i.e. batch_experience_buffer is None) - else the below is ignored.
# set the number of steps to record (experience buffer size)
experiment_params.batch_expert_steps_to_record = 50000      # number of steps to rollout into the buffer
# if using non-random agent (see batch_expert_params), set its training period
# else, the following parameter is ignored
experiment_params.batch_expert_n_timesteps = 100000      # n_timesteps to train the expert before starting to rollout




#####################
# main agent
experiment_params.trained_agent = trained_agent
experiment_params.agent_params = agent_params
experiment_params.n_timesteps = int(1e7)







