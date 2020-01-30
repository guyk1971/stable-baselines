from my_zoo.hyperparams.default_config import DQNAgentParams,ExperimentParams, EnvParams
from stable_baselines.deepq import MlpPolicy
from zoo.utils import CustomDQNPolicy


#################
# Env           #
#################
env_params = EnvParams()
env_params.env_id = 'cartpole1'



#################
# Policy        #
#################
# policy = MlpPolicy
policy = CustomDQNPolicy

#################
# Agent Params  #
#################
agent_params = DQNAgentParams()
# here we can change the various parameters - for example, we can change the batch size
agent_params.batch_size = 64
agent_params.policy = policy
agent_params.learning_rate = 1e-3
agent_params.buffer_size = 50000
agent_params.prioritized_replay = True





#################
# Experiment    #
#################
experiment_params = ExperimentParams()
experiment_params.n_timesteps = 1e5

experiment_params.env_params = env_params
experiment_params.agent_params = agent_params
experiment_params.name = __name__.split('.')[-1]





