from my_zoo.hyperparams.default_config import DQNAgentParams,ExperimentParams, EnvParams
from stable_baselines.deepq import MlpPolicy



#################
# Env           #
#################
env_params = EnvParams()
env_params.env_id = 'cartpole'



#################
# Policy        #
#################
policy = MlpPolicy


#################
# Agent Params  #
#################
agent_params = DQNAgentParams()
# here we can change the various parameters - for example, we can change the batch size
agent_params.batch_size = 64



#################
# Experiment    #
#################
experiment_params = ExperimentParams()
experiment_params.env_params = env_params
experiment_params.agent_params = agent_params
experiment_params.policy = policy
experiment_params.name = __name__.split('.')[-1]





