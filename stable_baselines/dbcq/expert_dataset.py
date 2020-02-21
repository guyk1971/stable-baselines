import os
import warnings
from typing import Dict

import cv2  # pytype:disable=import-error
import numpy as np
from gym import spaces

from stable_baselines.common.base_class import BaseRLModel
from stable_baselines.common.vec_env import VecEnv, VecFrameStack
from stable_baselines.common.base_class import _UnvecWrapper
from my_zoo.utils.common import title

from stable_baselines.dbcq.replay_buffer import ReplayBuffer

def generate_experience_traj(model, save_path=None, env=None, n_timesteps_train=0,
                         n_timesteps_record=100000, logger=None):
    """
    Train expert controller (if needed) and record expert trajectories.

    .. note::

        only Box and Discrete spaces are supported for now.
        support in images is removed.

    :param model: (RL model or callable) The expert model, if it needs to be trained,
        then you need to pass ``n_timesteps > 0``.
        note that the RL model can be also a pretrained expert that was loaded from file.
    :param save_path: (str) Path without the extension where the expert dataset will be saved
        (ex: 'expert_cartpole' -> creates 'expert_cartpole.npz').
        If not specified, it will not save, and just return the generated expert trajectories.
        This parameter must be specified for image-based environments.
    :param env: (gym.Env) The environment, if not defined then it tries to use the model
        environment.
    :param n_timesteps_train: (int) Number of training timesteps
    :param n_timesteps_record: (int) Number of trajectories (episodes) to record
    :param logger: (Logger) - if not None, use it for verbose output
    :return: (dict) the generated expert trajectories.
    """

    # Retrieve the environment using the RL model
    if env is None and isinstance(model, BaseRLModel):
        env = model.get_env()

    assert env is not None, "You must set the env in the model or pass it to the function."

    is_vec_env = False
    if isinstance(env, VecEnv) and not isinstance(env, _UnvecWrapper):
        is_vec_env = True
        if env.num_envs > 1:
            warnings.warn("You are using multiple envs, only the data from the first one will be recorded.")

    # Sanity check
    assert (isinstance(env.observation_space, spaces.Box) or
            isinstance(env.observation_space, spaces.Discrete)), "Observation space type not supported"

    assert (isinstance(env.action_space, spaces.Box) or
            isinstance(env.action_space, spaces.Discrete)), "Action space type not supported"

    # Note: support in recording image to files is omitted
    obs_space = env.observation_space
    replay_buffer = ReplayBuffer(n_timesteps_record)

    print(title("generate expert trajectory",20))
    print("training expert start - {0} timesteps".format(n_timesteps_train))
    if n_timesteps_train > 0 and isinstance(model, BaseRLModel):
        model.learn(n_timesteps_train)

    print("generate expert trajectory: training expert end")
    print("start recording {0} expert steps".format(n_timesteps_record))

    episode_returns = []
    episode_starts = []

    ep_idx = 0
    obs = env.reset()
    episode_starts.append(True)
    reward_sum = 0.0
    # state and mask for recurrent policies
    state, mask = None, None
    if is_vec_env:
        mask = [True for _ in range(env.num_envs)]
    for t in range(n_timesteps_record):
    # while ep_idx < n_episodes:
        if isinstance(model, BaseRLModel):
            action, state = model.predict(obs, state=state, mask=mask)
        else:
            action = model(obs)
        new_obs, reward, done, info = env.step(action)
        # obs, reward, done, _ = env.step(action)

        # Use only first env
        if is_vec_env:
            mask = [done[0] for _ in range(env.num_envs)]
            action = np.array([action[0]])
            reward = np.array([reward[0]])
            done = np.array([done[0]])

        # Store transition in the replay buffer.
        replay_buffer.add(obs, action, reward, new_obs, float(done))
        obs = new_obs
        # actions.append(action)
        # rewards.append(reward)
        episode_starts.append(done)
        reward_sum += reward
        if done:
            if not is_vec_env:
                obs = env.reset()
                # Reset the state in case of a recurrent policy
                state = None

            episode_returns.append(reward_sum)
            reward_sum = 0.0
            ep_idx += 1

    print("finished collecting experience data")
    numpy_dict = replay_buffer.record_buffer()
    # Note : the ReplayBuffer can not generally assume it has not circled around thus cant infer accurate episode
    # statistics. since in this context we know these details, we overwrite the corresponding fields:
    numpy_dict['episode_returns'] = np.array(episode_returns)
    numpy_dict['episode_starts'] = np.array(episode_starts)


    # for key, val in numpy_dict.items():
    #     print(key, val.shape)

    if save_path is not None:
        np.savez(save_path, **numpy_dict)

    env.close()
    return numpy_dict
