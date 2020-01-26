"""
train_gymcc.py
train an agent on gym classic control environment.
Supported environments :

"""
import os
import time
import difflib
import argparse
import importlib
from pprint import pprint
from collections import OrderedDict
import gym
import numpy as np

try:
    import mpi4py
    from mpi4py import MPI
except ImportError:
    mpi4py = None

from my_zoo.utils.common import suppress_tensorflow_warnings,set_gpu_device,Logger

DEFAULT_OUTPUT_DIR = os.path.join(os.path.expanduser('~'),'share','Data','MLA','stbl','results')
LOGGER_NAME=os.path.splitext(os.path.basename(__file__))[0]

ALGOS = {
    # 'a2c': A2C,
    # 'acer': ACER,
    # 'acktr': ACKTR,
    'dqn': DQN,
    # 'ddpg': DDPG,
    # 'her': HER,
    # 'sac': SAC,
    # 'trpo': TRPO,
    # 'td3': TD3
    'ppo2': PPO2
}

CC_ENVS = {'cartpole':'CartPole-v0',
           'cartpole1':'CartPole-v1',
           'mntcar':'MountainCar-v0',
           'acrobot':'Acrobot-v0',
           'lunland':'LunarLander-v2'
           }

def parse_cmd_line():

    parser = argparse.ArgumentParser()
    parser.add_argument('exparams', type=str, help='experiment params file path')
    parser.add_argument('--logdir',type=str,default=DEFAULT_OUTPUT_DIR,help='root directory for results')
    parser.add_argument('-d','--gpuid',type=str,default='',help='gpu id or "cpu"')
    parser.add_argument('--algo', help='RL Algorithm', default='dqn',
                        type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument('--num_exp', help='number of experiments', default=1,type=int)


    parser.add_argument('-i', '--trained-agent', help='Path to a pretrained agent to continue training',
                        default='', type=str)
    parser.add_argument('-n', '--n-timesteps', help='Overwrite the number of timesteps', default=-1,
                        type=int)
    parser.add_argument('--log-interval', help='Override log interval (default: -1, no change)', default=-1,
                        type=int)
    parser.add_argument('--seed', help='Random generator seed', type=int, default=1)
    # hyper parameter optimization - currently disabled
    parser.add_argument('-optimize', '--optimize-hyperparameters', action='store_true', default=False,
                        help='Run hyperparameters search')
    parser.add_argument('--n-trials', help='Number of trials for optimizing hyperparameters', type=int, default=10)
    parser.add_argument('--n-jobs', help='Number of parallel jobs when optimizing hyperparameters', type=int, default=1)
    parser.add_argument('--sampler', help='Sampler to use when optimizing hyperparameters', type=str,
                        default='tpe', choices=['random', 'tpe', 'skopt'])
    parser.add_argument('--pruner', help='Pruner to use when optimizing hyperparameters', type=str,
                        default='median', choices=['halving', 'median', 'none'])
    parser.add_argument('--verbose', help='Verbose mode (0: no output, 1: INFO)', default=1,type=int)
    args = parser.parse_args()
    return args



def run_main():

    args = parse_cmd_line()

    print('reading experiment params from '+args.exparams)


    # set logger
    logger=Logger(LOGGER_NAME).get_logger()

    # set compute device
    if args.gpuid=='cpu':
        set_gpu_device('')
    elif len(args.gpuid)>0:
        set_gpu_device(args.gpuid)
    else:
        pass




    # parse the experiment params - currently assume json file

    # define the experiment dir

    for e in range(args.n_experiments):
        seed = args.seed+10*e
        # update experiment seed
        run_experiment(experiment_params)

    return



if __name__ == '__main__':
    run_main()







