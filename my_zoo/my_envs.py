#####################################################################
# my_envs.py
# Implementations of custom envs
# MLA Template Equivalent: train.custom_envs
import gym
from gym import spaces
import numpy as np

class L2PEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple env that imitates the L2P behaviour
    """
    # Because of google colab, we cannot implement the GUI ('human' render mode)
    metadata = {'render.modes': ['console']}

    def __init__(self, obs_dim=7,n_act=4):
        super(L2PEnv, self).__init__()

        # the observation space include obs_dim float values
        self.obs_dim = obs_dim
        # Currently assuming discrete action space with n_act actions
        self.act_dim = 1
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions, we have two: left and right
        self.action_space = spaces.Discrete(n_act)
        # The observation will be the coordinate of the agent
        # this can be described both by Discrete and Box space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,shape=(obs_dim,), dtype=np.float32)
        self.max_path_length = 40

    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        self.step_idx=0
        return self.observation_space.sample()


    def step(self, action):

        if (not isinstance(action,int)) or (action<0) or (action>=self.action_space.n):
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))
        self.step_idx += 1

        state = self.observation_space.sample()
        done = False
        if self.step_idx == self.max_path_length:
            done = True
            self.step_idx = 0
        reward = 1.0

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return state, reward, done, info

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()

    def close(self):
        pass