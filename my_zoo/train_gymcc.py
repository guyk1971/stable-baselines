"""
train_gymcc.py
train an agent on gym classic control environment.
Supported environments :

"""
import os
import sys
from time import strftime
import difflib
import argparse
import importlib
from pprint import pprint
from collections import OrderedDict
import gym
import numpy as np
import logging

try:
    import mpi4py
    from mpi4py import MPI
except ImportError:
    mpi4py = None

from my_zoo.utils.common import suppress_tensorflow_warnings,set_gpu_device,MyLogger,StreamToLogger

CONFIGS_DIR = os.path.join(os.path.expanduser('~'),'share','Data','MLA','stbl','configs')
LOGGER_NAME=os.path.splitext(os.path.basename(__file__))[0]




def parse_cmd_line():

    parser = argparse.ArgumentParser()
    parser.add_argument('exparams', type=str, help='experiment params file path')
    parser.add_argument('-d','--gpuid',type=str,default='',help='gpu id or "cpu"')
    parser.add_argument('--num_experiments', help='number of experiments', default=1,type=int)
    parser.add_argument('--seed', help='Random generator seed', type=int, default=1)
    parser.add_argument('-i', '--trained-agent', help='Path to a pretrained agent to continue training',
                        default='', type=str)
    parser.add_argument('-n', '--n-timesteps', help='Overwrite the number of timesteps', default=-1,type=int)
    parser.add_argument('--log-interval', help='Override log interval (default: -1, no change)', default=-1,type=int)
    args = parser.parse_args()
    return args


def create_logger(exp_params,stdout_to_log=True):

    log_file_name=os.path.join(exp_params.output_root_dir,exp_params.name+'.log')
    log_level=exp_params.log_level
    log_format=exp_params.log_format
    logger = MyLogger(LOGGER_NAME,filename=log_file_name,
                      level=log_level,format=log_format).get_logger()

    if stdout_to_log:
        sys.stdout = StreamToLogger(logger, logging.INFO)
        sys.stderr = StreamToLogger(logger, logging.ERROR)

    return logger


def run_experiment(experiment_params):
    logger = logging.getLogger(LOGGER_NAME)
    seed=experiment_params.seed
    logger.info(f"\n\n**** starting experiment seed {seed} ****")
    # todo: create a working directory for the relevant seed
    #
    print("from within the experiment...")
    logger.info(f"\n\n**** completed experiment seed {seed} ****")
    return




def main():

    args = parse_cmd_line()
    print('reading experiment params from '+args.exparams)

    from my_zoo.hyperparams.exp_dqn import experiment_params

    # todo: find a way to load the parameters
    # module_path = os.path.join(CONFIGS_DIR,args.exparams)
    # exp_params_module = importlib.import_module(module_path)
    # experiment_params = getattr(exp_params_module,'experiment_params')

    # set logger and redirect stdout to logger
    # logger name should be : <experiment_config_name>_<time>_


    # set compute device
    if args.gpuid=='cpu':
        set_gpu_device('')
    elif len(args.gpuid)>0:
        set_gpu_device(args.gpuid)
    else:
        pass

    # create experiment folder and logger
    exp_folder = args.exparams + '-' + strftime("%d-%m-%Y_%H-%M-%S")
    experiment_params.output_root_dir = os.path.join(experiment_params.output_root_dir,exp_folder)
    os.makedirs(experiment_params.output_root_dir, exist_ok=True)

    logger = create_logger(experiment_params)
    logger.info("\n\n************* Starting {0} experiments **********".format(args.num_experiments))

    for e in range(args.num_experiments):
        seed = args.seed+10*e
        experiment_params.seed = seed
        # run experiment will generate its own sub folder for each seed
        # not yet clear how to support hyper parameter search...
        run_experiment(experiment_params)

    sys.stdout=sys.__stdout__
    sys.stderr=sys.__stderr__

    logging.shutdown()

    return

if __name__ == '__main__':
    # suppress_tensorflow_warnings()
    main()







