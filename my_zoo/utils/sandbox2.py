import os
import sys
from my_zoo.utils.common import *
import numpy as np
import gym
# from stable_baselines import SAC
from stable_baselines import DQN
from stable_baselines import logger
from my_zoo.utils.global_logger import title
from my_zoo.hyperparams.default_config import ExperimentParams
import shutil


LOGGER_NAME=os.path.splitext(os.path.basename(__file__))[0]


def evaluate(model, env=None, num_episodes=100):
    # This function will only work for a single Environment
    env = env or model.get_env()
    all_episode_rewards = []
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)

        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    return mean_episode_reward


def main():
    # eval_env = gym.make('Pendulum-v0')
    # default_model = SAC('MlpPolicy', 'Pendulum-v0', verbose=1).learn(8000)
    # evaluate(default_model, eval_env, num_episodes=100)
    experiment_params=ExperimentParams()
    experiment_params.name='Sandbox2'
    # glogger = create_logger(LOGGER_NAME,experiment_params)
    # configure the agent's logger

    out_dir = os.path.join(experiment_params.output_root_dir, experiment_params.name)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    # logger.configure(out_dir,['stdout','log','csv','tensorboard'])
    logger.configure(out_dir,['stdout', 'log', 'csv'])
    logger.info(title("start logging",20))
    kwargs = {'double_q': False, 'prioritized_replay': False, 'policy_kwargs': dict(dueling=False,layers=[256,128,64]),
              'tensorboard_log': out_dir}
    dqn_model = DQN('MlpPolicy', 'CartPole-v1', verbose=1, **kwargs)
    mean_reward_before_train = evaluate(dqn_model, num_episodes=100)
    logger.info('mean reward before train {0}'.format(mean_reward_before_train))
    logger.info('starting learning...')
    dqn_model.learn(total_timesteps=10000, log_interval=10)
    logger.info('finished learning...')
    mean_reward = evaluate(dqn_model, num_episodes=100)
    logger.info("mean reward after train {0}".format(mean_reward))
    return

if __name__=='__main__':
    main()
    exit(0)
