from stable_baselines.dbcq.policies import MlpPolicy, CnnPolicy, LnMlpPolicy, LnCnnPolicy
from stable_baselines.dbcq.build_graph import build_act, build_train  # noqa
from stable_baselines.dbcq.dbcq import DBCQ
# from stable_baselines.dbcq.replay_buffer import ReplayBuffer


# def wrap_atari_dqn(env):
#     """
#     wrap the environment in atari wrappers for DQN
#
#     :param env: (Gym Environment) the environment
#     :return: (Gym Environment) the wrapped environment
#     """
#     from stable_baselines.common.atari_wrappers import wrap_deepmind
#     return wrap_deepmind(env, frame_stack=True, scale=False)
