############################
# set the python path properly
import os
import sys
path_to_curr_file=os.path.realpath(__file__)
proj_root=os.path.dirname(os.path.dirname(path_to_curr_file))
if proj_root not in sys.path:
    sys.path.insert(0,proj_root)
############################
import time
import argparse
import importlib
from collections import OrderedDict
import gym
import yaml
from stable_baselines import logger
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import VecFrameStack
import shutil

try:
    import mpi4py
    from mpi4py import MPI
except ImportError:
    mpi4py = None

from my_zoo.utils.utils import ALGOS
from zoo.utils.hyperparams_opt import hyperparam_optimization

from my_zoo.utils.common import *
from my_zoo.utils.train import get_create_env,parse_agent_params

from my_zoo.hyperparams.default_config import *


CONFIGS_DIR = os.path.join(os.path.expanduser('~'),'share','Data','MLA','stbl','configs')
LOGGER_NAME=os.path.splitext(os.path.basename(__file__))[0]


def parse_cmd_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('agent', type=str, help='agent class e.g. DQN,random,PPO,DBCQ. if not random, need to provide checkpoint',default='random')
    parser.add_argument('env',type=str,help='string of registered environment to run on')
    # note: the checkpoint is assumed to be saved in a folder <path_to_results>/checkpoint_<iter>
    # note that inside this folder there is a file with the same name 'checkpoint_<iter>'. we expect to get the folder.
    parser.add_argument('--ckpt',type=str,help='path to checkpoint (ignored for random agent)',default=None)
    parser.add_argument('-n',"--num-steps", type=int, default=1000, help='number of steps in the env to play with')
    parser.add_argument('-o','--output_dir',type=str,help='location of output file')
    parser.add_argument('-d','--gpuid',type=str,default='',help='gpu id or "cpu"')
    parser.add_argument('--seed',type=int,default=1,help='initial value for seed')
    args = parser.parse_args()
    return args

def main():
    args = parse_cmd_line()
    os.makedirs(args.output_dir, exist_ok=True)
    logger.configure(os.path.join(args.output_dir), ['stdout', 'log'])

    # set compute device
    if args.gpuid=='cpu':
        set_gpu_device('')
    elif len(args.gpuid)>0:
        set_gpu_device(args.gpuid)
    else:
        pass
    # set global seeds
    seed=args.seed

    set_global_seeds(seed)

    rank = 0
    if mpi4py is not None and MPI.COMM_WORLD.Get_size() > 1:
        logger.info("Using MPI for multiprocessing with {} workers".format(MPI.COMM_WORLD.Get_size()))
        rank = MPI.COMM_WORLD.Get_rank()
        logger.info("Worker rank: {}".format(rank))
        # make sure that each worker has its own seed
        seed += rank
        # we allow only one worker to "speak"
        if rank != 0:
            verbose = 0

    # todo: if necessary, complete this function. not completed





if __name__ == '__main__':
    suppress_tensorflow_warnings()
    main()
    sys.path.remove(proj_root)
