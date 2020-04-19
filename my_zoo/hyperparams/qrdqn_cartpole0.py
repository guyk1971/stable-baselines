from my_zoo.hyperparams.default_config import QRDQNAgentParams,ExperimentParams, EnvParams
from stable_baselines.qrdqn import MlpPolicy


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
agent_params = QRDQNAgentParams()
# here we can change the various parameters - for example, we can change the batch size
agent_params.policy = policy
agent_params.learning_rate = 1e-3
agent_params.buffer_size = 50000
agent_params.exploration_final_eps = 0.02
agent_params.exploration_fraction = 0.1
agent_params.n_atoms = 10
agent_params.policy_kwargs = {'dueling':False,'n_atoms': agent_params.n_atoms, 'layers': [64]}

#################
# Experiment    #
#################
experiment_params = ExperimentParams()
experiment_params.n_timesteps = 100000

experiment_params.env_params = env_params
experiment_params.agent_params = agent_params
experiment_params.name = __name__.split('.')[-1]





