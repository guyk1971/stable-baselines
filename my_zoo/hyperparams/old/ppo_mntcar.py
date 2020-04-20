from my_zoo.hyperparams.default_config import PPO2AgentParams,ExperimentParams, EnvParams


##########################################################
# Env                                                    #
##########################################################
env_params = EnvParams()
env_params.env_id = 'mntcar'
env_params.norm_obs = True
env_params.norm_reward = True

#################
# Policy        #
#################
policy = 'MlpPolicy'

##########################################################
# Agent Params                                           #

# Default values:
# policy = 'MlpPolicy'
# n_steps = 128
# nminibatches = 4
# lam = 0.95
# gamma = 0.99
# noptepochs = 4
# ent_coef = 0.0
# learning_rate = 2.5e-4  # can also be 'lin_<float>' e.g. 'lin_0.001'
# cliprange = 0.2  # can also be 'lin_<float>' e.g. 'lin_0.1'
# vf_coef = 0.5
# ent_coef = 0.01
# cliprange_vf = None
# max_grad_norm = 0.5
# policy_kwargs = None
# n_cpu_tf_sess = None
##########################################################
agent_params = PPO2AgentParams()
# here we can change the various parameters - for example, we can change the batch size
agent_params.policy = policy
agent_params.learning_rate = 1e-3
agent_params.n_steps = 16
agent_params.nminibatches = 1
agent_params.lam = 0.98
agent_params.ent_coef=0.0
agent_params.cliprange = 0.2

##########################################################
# Experiment                                             #
##########################################################
experiment_params = ExperimentParams()
experiment_params.n_timesteps = 1e6
experiment_params.n_envs = 16
experiment_params.env_params = env_params
experiment_params.agent_params = agent_params
experiment_params.name = __name__.split('.')[-1]





