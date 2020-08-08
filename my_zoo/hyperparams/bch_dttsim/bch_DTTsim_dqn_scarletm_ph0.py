# bch_DTTsim_dqn_scarletm_ph0.py


from stable_baselines.deepq import MlpPolicy
from my_zoo.hyperparams.default_config import *
from my_zoo.my_envs import EPISODES
from my_zoo.dttsim_wrappers import reward_7,feature_extraction_scarlet_ns
##########################################################
# Env                                                    #
##########################################################
env_params = DTTEnvSimParams()
env_params.episode_workloads = EPISODES['cb20mr']
env_params.full_reset = True
env_params.use_wrapper = True
env_params.wrapper_params['feature_extractor'] = feature_extraction_scarlet_ns
env_params.wrapper_params['reward_calc'] = reward_7
env_params.wrapper_params['n_frames'] = 5

##########################################################
# Experience Buffer or Expert generator
##########################################################
experience_dataset = "/home/gkoren2/share/Data/MLA/DTT/results/stbl/bchsim/onlsim_rnd_ScarletM_cb20mix3_f2_r7-25-06-2020_12-48-55/1/er_DTTSim_random_175000.csv"
# load the expert model from file, without training:
expert_model_file = None       # agent to load to generate experience

##########################################################
# Agent Params                                           #
##########################################################
trained_agent_model_file = None            # if we want to continue train an agent, set the path to saved model
agent_params = DQNAgentParams()
# here we can change the various parameters - for example, we can change the batch size
agent_params.policy = MlpPolicy
agent_params.verbose = 1
agent_params.learning_rate = 1e-4
agent_params.policy_kwargs = {'layers': [64, 32]}
agent_params.target_network_update_freq = 1         # every 1 epoch
agent_params.ope_freq = 0                          # off policy evaluate and maybe save every # epochs
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
experiment_params.n_timesteps = int(5e7)        # number of epochs = n_timesteps/n_timesteps_in_csv

experiment_params.online_eval_freq = 0  # evaluate on eval env every this number of timesteps, if 0 - no online eval
experiment_params.online_eval_freq = int(experiment_params.n_timesteps/20)  # evaluate on eval env every this number of timesteps
experiment_params.online_eval_n_episodes = 30

# post training the main agent - if we want to record experience with the new expert:
experiment_params.expert_steps_to_record = 0      # number of steps to rollout into the buffer







