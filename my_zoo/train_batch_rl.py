"""
train_batch_rl.py
train an agent on gym classic control environment.
Supported environments :

"""
import os
import sys
import time
import argparse
import importlib
from functools import partial
from collections import OrderedDict
import gym
import numpy as np
import yaml
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import VecFrameStack, SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines.common.noise import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines.ppo2.ppo2 import constfn  # todo: consider adding it directly to the config class of ppo
from stable_baselines.dbcq.replay_buffer import ReplayBuffer
import copy

try:
    import mpi4py
    from mpi4py import MPI
except ImportError:
    mpi4py = None

from zoo.utils import make_env, ALGOS, linear_schedule, get_latest_run_id, get_wrapper_class, find_saved_model
from zoo.utils.hyperparams_opt import hyperparam_optimization
from zoo.utils.noise import LinearNormalActionNoise

from my_zoo.utils.common import *
from my_zoo.utils.train import create_logger,get_create_env,parse_agent_params
from my_zoo.hyperparams.default_config import CC_ENVS


LOGGER_NAME=os.path.splitext(os.path.basename(__file__))[0]




def parse_cmd_line():

    parser = argparse.ArgumentParser()
    parser.add_argument('exparams', type=str, help='experiment params file path')
    parser.add_argument('-d','--gpuid',type=str,default='',help='gpu id or "cpu"')
    parser.add_argument('--num_experiments', help='number of experiments', default=1,type=int)
    parser.add_argument('--seed', help='Random generator seed', type=int, default=1)
    parser.add_argument('-n', '--n_timesteps', help='Overwrite the number of timesteps', default=-1,type=int)
    parser.add_argument('--log_interval', help='Override log interval (default: -1, no change)', default=-1,type=int)
    args = parser.parse_args()
    return args


def create_experience_buffer(experiment_params,output_dir):
    experience_buffer = None
    # todo: copy the main flow from train_gymcc
    exp_agent_params = experiment_params.batch_experience_agent_params.as_dict()
    if experiment_params.verbose>0:
        experiment_params.batch_experience_agent_params.verbose = experiment_params.verbose
        experiment_params.batch_experience_agent_params.tensorboard_log = output_dir

    algo = exp_agent_params['algorithm']
    seed = experiment_params.batch_experience_agent_params.seed

    ###################
    # make the env
    n_envs = experiment_params.n_envs
    env_id = experiment_params.env_params.env_id
    logger.info('using {0} instances of {1} :'.format(n_envs,env_id))
    create_env = get_create_env(algo,seed,experiment_params.env_params,logger)
    env = create_env(n_envs)

    #####################
    # create the agent
    if ALGOS[algo] is None:
        raise ValueError('{} requires MPI to be installed'.format(algo))

    n_actions = 1 if isinstance(env.action_space,gym.spaces.Discrete) else env.action_space.shape[0]
    parse_agent_params(exp_agent_params,n_actions,experiment_params.n_timesteps,logger)

    normalize = experiment_params.env_params.norm_obs or experiment_params.env_params.norm_reward

    trained_agent = experiment_params.batch_experience_trained_agent
    if trained_agent:
        valid_extension = trained_agent.endswith('.pkl') or trained_agent.endswith('.zip')
        assert valid_extension and os.path.isfile(trained_agent), \
            "The trained_agent must be a valid path to a .zip/.pkl file"
        logger.info("loading pretrained agent to continue training")
        # if policy is defined, delete as it will be loaded with the trained agent
        del exp_agent_params['policy']
        model = ALGOS[algo].load(trained_agent, env=env,**exp_agent_params)
        exp_folder = trained_agent[:-4]

        if normalize:
            logger.info("Loading saved running average")
            env.load_running_average(exp_folder)

    else:       # create a model from scratch
        model = ALGOS[algo](env=env, **exp_agent_params)

    kwargs = {}
    if experiment_params.log_interval > -1:
        kwargs = {'log_interval': experiment_params.log_interval}

    model.learn(int(experiment_params.n_timesteps), **kwargs)
    experience_buffer = copy.deepcopy(model.replay_buffer)

    return experience_buffer

def load_or_create_experience_buffer(experiment_params,output_dir):
    # if we got an existing experience buffer, load from file and return it
    if experiment_params.batch_experience_buffer and os.path.exists(experiment_params.batch_experience_buffer):
        logger.info('loading experience buffer from '+experiment_params.batch_experience_buffer)
        experience_buffer = ReplayBuffer(experiment_params.batch_experience_agent_params.buffer_size)
        experience_buffer.load(experiment_params.batch_experience_buffer)
        return experience_buffer
    # if we got to this line, we need to generate an experience buffer
    logger.info('Generating experience buffer with ' + experiment_params.batch_experience_agent_params.algorithm)
    exp_agent_algo=experiment_params.batch_experience_agent_params.algorithm
    experience_buffer = create_experience_buffer(experiment_params,output_dir)
    # save the experience buffer
    exp_buf_filename = 'er_'+experiment_params.env_params.env_id+'_'+exp_agent_algo
    exp_buf_filename = os.path.join(output_dir,exp_buf_filename)
    experience_buffer.save(exp_buf_filename)
    logger.info('Saving experience buffer to ' + exp_buf_filename)
    return experience_buffer

##############################################################################
# run experiment
##############################################################################
def run_experiment(experiment_params):
    # logger = logging.getLogger(LOGGER_NAME)
    seed=experiment_params.seed
    # set global seeds : in numpy, random, tf and gym
    set_global_seeds(experiment_params.seed)

    rank = 0
    if mpi4py is not None and MPI.COMM_WORLD.Get_size() > 1:
        print("Using MPI for multiprocessing with {} workers".format(MPI.COMM_WORLD.Get_size()))
        rank = MPI.COMM_WORLD.Get_rank()
        print("Worker rank: {}".format(rank))
        # make sure that each worker has its own seed
        seed += rank
        # we allow only one worker to "speak"
        if rank != 0:
            experiment_params.verbose = 0

    logger.info(title("starting experiment seed {}".format(seed),30))

    # create a working directory for the relevant seed
    output_dir=os.path.join(experiment_params.output_root_dir,str(seed))
    os.makedirs(output_dir, exist_ok=True)

    # logger.info(experiment_params.as_dict())
    # set the seed for the current worker

    experiment_params.agent_params.seed = seed

    saved_exp_agent_hparams = None
    if experiment_params.batch_experience_agent_params:
        experiment_params.batch_experience_agent_params.seed = seed
        exp_agent_hparams = experiment_params.batch_experience_agent_params.as_dict()
        saved_exp_agent_hparams = OrderedDict(
            [(key, str(exp_agent_hparams[key])) for key in sorted(exp_agent_hparams.keys())])

    agent_hyperparams = experiment_params.agent_params.as_dict()
    env_params_dict = experiment_params.env_params.as_dict()
    exparams_dict = experiment_params.as_dict()

    saved_env_params = OrderedDict([(key, str(env_params_dict[key])) for key in sorted(env_params_dict.keys())])
    saved_agent_hparams = OrderedDict([(key, str(agent_hyperparams[key])) for key in sorted(agent_hyperparams.keys())])

    saved_hyperparams = OrderedDict([(key, str(exparams_dict[key])) for key in exparams_dict.keys()])
    saved_hyperparams['agent_params']=saved_agent_hparams
    saved_hyperparams['env_params'] = saved_env_params
    saved_hyperparams['batch_experience_agent_params'] = saved_exp_agent_hparams

    # load or create the experience replay buffer
    er_buf = load_or_create_experience_buffer(experiment_params,output_dir)

    # start batch rl training
    logger.info("Experience buffer is ready with {0} samples".format(er_buf.buffer_size))
    logger.info(title('start batch training',30))

    return



def main():

    args = parse_cmd_line()
    print('reading experiment params from '+args.exparams)

    module_path = 'my_zoo.hyperparams.'+args.exparams
    exp_params_module = importlib.import_module(module_path)
    experiment_params = getattr(exp_params_module,'experiment_params')

    # set compute device
    if args.gpuid=='cpu':
        set_gpu_device('')
    elif len(args.gpuid)>0:
        set_gpu_device(args.gpuid)
    else:
        pass

    # create experiment folder and logger
    exp_folder_name = args.exparams + '-' + time.strftime("%d-%m-%Y_%H-%M-%S")
    experiment_params.output_root_dir = os.path.join(experiment_params.output_root_dir,exp_folder_name)
    os.makedirs(experiment_params.output_root_dir, exist_ok=True)

    global logger
    logger = create_logger(LOGGER_NAME,experiment_params,stdout_to_log=False)


    # check if some cmd line arguments should override the experiment params
    if args.n_timesteps > -1:
        logger.info("overriding n_timesteps with {}".format(args.n_timesteps))
        experiment_params.n_timesteps=args.n_timesteps
    if args.log_interval > -1:
        logger.info("overriding log_interval with {}".format(args.log_interval))
        experiment_params.log_interval=args.log_interval

    logger.info(title('Starting {0} experiments'.format(args.num_experiments), 40))
    for e in range(args.num_experiments):
        seed = args.seed+100*e
        experiment_params.seed = seed
        # run experiment will generate its own sub folder for each seed
        run_experiment(experiment_params)

    # prepare for shutdown logger
    sys.stdout=sys.__stdout__
    sys.stderr=sys.__stderr__
    logging.shutdown()
    return




if __name__ == '__main__':
    # suppress_tensorflow_warnings()
    main()