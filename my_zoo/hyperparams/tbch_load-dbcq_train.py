# tbch_load-dbcq_train.py

from my_zoo.hyperparams.default_config import *
from stable_baselines.dbcq import MlpPolicy as dbcq_MlpPolicy

##########################################################
# Env                                                    #
##########################################################
env_params = EnvParams()
env_params.env_id = 'acrobot'


##########################################################
# Experience Buffer or Expert generator
##########################################################
experience_dataset = '/home/gkoren2/share/Data/MLA/stbl/results/tonl_random_rec-20-04-2020_19-02-38/1/er_acrobot_random_50000.csv'
# load the expert model from file, without training:
expert_model_file = None       # agent to load to generate experience

##########################################################
# Agent Params                                           #
##########################################################
# if we want to continue train an agent, set the path to saved model
trained_agent_model_file = '/home/gkoren2/share/Data/MLA/stbl/results/cfg_brl_er_rnd-05-04-2020_17-23-44/1/model_params.zip'
agent_params = DBCQAgentParams()
# here we can change the various parameters - for example, we can change the batch size
agent_params.policy = dbcq_MlpPolicy
agent_params.verbose = 1
agent_params.learning_rate = 1e-4
agent_params.policy_kwargs = {'dueling':False,'layers': [256, 512]}
agent_params.target_network_update_freq = 1         # every 1 epoch
agent_params.val_freq = 1               # every 1 epoch
agent_params.n_eval_episodes = 10
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

#############################
# fill in values from above
experiment_params.experience_dataset = experience_dataset
experiment_params.expert_model_file = expert_model_file

#####################
# main agent
experiment_params.trained_agent_model_file = trained_agent_model_file
experiment_params.agent_params = agent_params
experiment_params.n_timesteps = int(1e6)

# post training the main agent - if we want to record experience with the new expert:
experiment_params.expert_steps_to_record = 0      # number of steps to rollout into the buffer







