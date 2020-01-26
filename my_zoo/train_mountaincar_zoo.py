import argparse
import os
import gym

from stable_baselines.deepq import DQN
from zoo.utils.utils import CustomDQNPolicy
from stable_baselines.bench import Monitor
from my_zoo.utils.common import suppress_tensorflow_warnings



def main(args):
    """
    Train and save the DQN model, for the mountain car problem

    :param args: (ArgumentParser) the input arguments
    """
    suppress_tensorflow_warnings()

    log_dir = args.logdir
    os.makedirs(log_dir, exist_ok=True)

    env = gym.make("MountainCar-v0")
    env = Monitor(env, log_dir, allow_early_resets=True)

    # using layer norm policy here is important for parameter space noise!
    model = DQN(
        policy="LnMlpPolicy",
        # policy=CustomDQNPolicy,
        env=env,
        learning_rate=1e-3,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.1,
        param_noise=True,
        policy_kwargs=dict(layers=[64]),
        tensorboard_log=log_dir,
        verbose=1)

    model.learn(total_timesteps=args.max_timesteps)

    model_path = os.path.join(log_dir,'mountaincar_model_zoo.zip')
    print("Saving model to "+model_path)
    model.save(model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train DQN on mountaincar")
    parser.add_argument('--logdir', help='log dir', default=os.path.expanduser('~')+'/share/Data/MLA/stbl/results', type=str)
    parser.add_argument('--max-timesteps', default=100000, type=int, help="Maximum number of timesteps")
    args = parser.parse_args()
    main(args)
