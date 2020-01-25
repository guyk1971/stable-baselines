import argparse

import gym
import numpy as np

from stable_baselines.deepq import DQN, MlpPolicy
from zoo.utils.utils import CustomDQNPolicy


def callback(lcl, _glb):
    """
    The callback function for logging and saving

    :param lcl: (dict) the local variables
    :param _glb: (dict) the global variables
    :return: (bool) is solved
    """
    # stop training if reward exceeds 199
    if len(lcl['episode_rewards'][-101:-1]) == 0:
        mean_100ep_reward = -np.inf
    else:
        mean_100ep_reward = round(float(np.mean(lcl['episode_rewards'][-101:-1])), 1)
    is_solved = lcl['self'].num_timesteps > 100 and mean_100ep_reward >= 199
    return not is_solved


def main(args):
    """
    Train and save the DQN model, for the cartpole problem

    :param args: (ArgumentParser) the input arguments
    """
    env = gym.make("Acrobot-v1")
    model = DQN(
        env=env,
        policy=CustomDQNPolicy,     # following zoo hyperparams
        learning_rate=1e-3,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        prioritized_replay=True,        # following zoo hyperparams
        verbose=1
    )
    model.learn(total_timesteps=args.max_timesteps)

    print("Saving model to acrobot_model_zoo.zip")
    model.save("acrobbot_model_zoo.zip")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train DQN on Acrobot-v1")
    parser.add_argument('--max-timesteps', default=100000, type=int, help="Maximum number of timesteps")
    args = parser.parse_args()
    main(args)
