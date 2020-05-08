# utst_bch_acrobot_rnd2dbcq.py

from my_zoo.hyperparams.default_config import *
from stable_baselines.qrdqn import MlpPolicy
##########################################################
# Env                                                    #
##########################################################
env_params = EnvParams()
env_params.env_id = 'acrobot'

##########################################################
# Experience Buffer or Expert generator
##########################################################
experience_dataset = '/home/guy/share/Data/MLA/stbl/results/utst_onl_acrobot_rnd-28-04-2020_12-47-23/1/er_acrobot_random_100000.csv'
# load the expert model from file, without training:
expert_model_file = None       # agent to load to generate experience

##########################################################
# Agent Params                                           #
##########################################################
trained_agent_model_file = None            # if we want to continue train an agent, set the path to saved model
agent_params = QRDQNAgentParams()
# here we can change the various parameters - for example, we can change the batch size
agent_params.policy = MlpPolicy
agent_params.verbose = 1
agent_params.learning_rate = 1e-4
agent_params.policy_kwargs = {'layers': [64]}
agent_params.target_network_update_freq = 1         # every 1 epoch
agent_params.batch_size = 128
agent_params.buffer_train_fraction = 1.0         # currently online evaluation. use all buffer for training


##########################################################
# Experiment                                             #
##########################################################
experiment_params = ExperimentParams()
experiment_params.name = __name__.split('.')[-1]
experiment_params.env_params = env_params

#############################
# fill in values from above
experiment_params.experience_dataset = experience_dataset
experiment_params.expert_model_file = expert_model_file

#####################
# main agent
experiment_params.trained_agent_model_file = trained_agent_model_file
experiment_params.agent_params = agent_params
experiment_params.n_timesteps = int(1e7)

experiment_params.evaluation_freq = int(experiment_params.n_timesteps/20)  # evaluate on eval env every this number of timesteps
experiment_params.online_eval_n_episodes = 30
experiment_params.off_policy_eval_dataset_eval_fraction = 0.3

# post training the main agent - if we want to record experience with the new expert:
experiment_params.expert_steps_to_record = 0      # number of steps to rollout into the buffer







