# from train.zoo_utils import CustomDQNPolicy
from stable_baselines.deepq.policies import MlpPolicy
from my_zoo.hyperparams.default_config import *


##########################################################
# Env                                                    #
##########################################################
env_params = DTTEnvSimParams()
env_params.workload = 5*(['cb15']+[('cooldown',1)]) + [('cooldown',600)] + 5*(['cb20']+[('cooldown',1)])
env_params.full_reset = False


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
agent_params = RandomAgentParams()


##########################################################
# Experiment                                             #
##########################################################
experiment_params = ExperimentParams()
experiment_params.n_timesteps = 0
experiment_params.env_params = env_params
experiment_params.agent_params = agent_params
experiment_params.expert_steps_to_record = 100000  # number of episodes to record into the experience buffer
experiment_params.name = __name__.split('.')[-1]





