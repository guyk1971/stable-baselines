# from train.zoo_utils import CustomDQNPolicy
from stable_baselines.qrdqn.policies import MlpPolicy
from my_zoo.hyperparams.default_config import *
from my_zoo.my_envs import EPISODES
from my_zoo.dttsim_wrappers import reward_3,feature_extraction_scarlet

##########################################################
# Env                                                    #
##########################################################

env_params = DTTEnvSimParams()
env_params.episode_workloads = EPISODES['cb20mr']
env_params.full_reset = True
env_params.use_wrapper = True
env_params.wrapper_params['feature_extractor'] = feature_extraction_scarlet
env_params.wrapper_params['reward_calc'] = reward_3
env_params.wrapper_params['n_frames'] = 5

#################
# Policy        #
#################
policy = MlpPolicy


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
agent_params = QRDQNAgentParams()
# here we can change the various parameters - for example, we can change the batch size
agent_params.policy = policy
agent_params.learning_rate = 1e-3
agent_params.gamma = 0.975
agent_params.exploration_final_eps = 0.1
agent_params.prioritized_replay = True
agent_params.policy_kwargs = {'layers': [64,32]}


##########################################################
# Experiment                                             #
##########################################################
experiment_params = ExperimentParams()
experiment_params.n_timesteps = 2e6
experiment_params.env_params = env_params
experiment_params.agent_params = agent_params
experiment_params.expert_steps_to_record = 50000  # number of episodes to record into the experience buffer
experiment_params.online_eval_freq = int(experiment_params.n_timesteps/10)  # evaluate on eval env every this number of timesteps
# experiment_params.online_eval_freq = 0  # evaluate on eval env every this number of timesteps
experiment_params.online_eval_n_episodes = 30
experiment_params.name = __name__.split('.')[-1]





