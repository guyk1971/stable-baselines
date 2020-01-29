"""
train_gymcc.py
train an agent on gym classic control environment.
Supported environments :

"""
import os
import sys
import time
import argparse
import importlib
from collections import OrderedDict
import gym
import numpy as np
import yaml
from stable_baselines.common import set_global_seeds
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack, SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines.common.noise import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise

try:
    import mpi4py
    from mpi4py import MPI
except ImportError:
    mpi4py = None

from my_zoo.utils.common import *
from my_zoo.utils import make_env, ALGOS, linear_schedule, get_latest_run_id, get_wrapper_class, find_saved_model
from zoo.utils.hyperparams_opt import hyperparam_optimization
from zoo.utils.noise import LinearNormalActionNoise

from my_zoo.hyperparams.default_config import CC_ENVS
from stable_baselines.ppo2.ppo2 import constfn  # todo: consider adding it directly to the config class of ppo


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


def create_env(algo,seed,n_envs,env_params):
    # logger = logging.getLogger(LOGGER_NAME)
    env_id = CC_ENVS.get(env_params.env_id,None)
    assert env_id , "env {0} is not supported".format(env_id)
    is_atari= 'NoFrameskip' in env_id
    # n_envs = experiment_params.get('n_envs', 1)
    logger.info('using {0} instances of {1} :'.format(n_envs,env_id))

    normalize_kwargs = {'norm_obs':env_params.norm_obs,'norm_reward':env_params.norm_reward}
    normalize = normalize_kwargs['norm_obs'] or normalize_kwargs['norm_reward']

    # obtain a class object from a wrapper name string in hyperparams
    # and delete the entry
    env_wrapper = get_wrapper_class(env_params.as_dict())

    def _create_env(n_envs):
        """
        Create the environment and wrap it if necessary
        :param n_envs: (int)
        :return: (gym.Env)
        """
        if is_atari:
            logger.info("Using Atari wrapper")
            env = make_atari_env(env_id, num_env=n_envs, seed=seed)
            # Frame-stacking with 4 frames
            env = VecFrameStack(env, n_stack=4)
        elif algo in ['dqn', 'ddpg']:
            if normalize:
                logger.warning("WARNING: normalization not supported yet for DDPG/DQN")
            env = gym.make(env_id)
            env.seed(seed)
            if env_wrapper is not None:
                env = env_wrapper(env)
        else:
            if n_envs == 1:
                env = DummyVecEnv([make_env(env_id, 0, seed, wrapper_class=env_wrapper)])
            else:
                # env = SubprocVecEnv([make_env(env_id, i, seed) for i in range(n_envs)])
                # On most env, SubprocVecEnv does not help and is quite memory hungry
                env = DummyVecEnv([make_env(env_id, i, seed, wrapper_class=env_wrapper) for i in range(n_envs)])
            if normalize:
                logger.info("Normalization activated: {}".format(normalize_kwargs))
                env = VecNormalize(env, **normalize_kwargs)
        # Optional Frame-stacking
        if env_params.frame_stack>1:
            n_stack = env_params.frame_stack
            env = VecFrameStack(env, n_stack)
            logger.info("Stacking {} frames".format(n_stack))
        return env

    env = _create_env(n_envs)

    return env

def parse_agent_params(hyperparams,n_actions,n_timesteps):

    algo = hyperparams['algorithm']
    del hyperparams['algorithm']

    # Parse the schedule parameters for the relevant algorithms
    if algo in ["ppo2", "sac", "td3"]:
        for key in ['learning_rate', 'cliprange', 'cliprange_vf']:
            if key not in hyperparams:
                continue
            if isinstance(hyperparams[key], str):
                schedule, initial_value = hyperparams[key].split('_')
                initial_value = float(initial_value)
                hyperparams[key] = linear_schedule(initial_value)
            elif isinstance(hyperparams[key], (float, int)):
                # Negative value: ignore (ex: for clipping)
                if hyperparams[key] < 0:
                    continue
                hyperparams[key] = constfn(float(hyperparams[key]))
            else:
                raise ValueError('Invalid value for {}: {}'.format(key, hyperparams[key]))

    # Parse noise string for DDPG and SAC
    if algo in ['ddpg', 'sac', 'td3'] and hyperparams.get('noise_type') is not None:
        noise_type = hyperparams['noise_type'].strip()
        noise_std = hyperparams['noise_std']
        if 'adaptive-param' in noise_type:
            assert algo == 'ddpg', 'Parameter is not supported by SAC'
            hyperparams['param_noise'] = AdaptiveParamNoiseSpec(initial_stddev=noise_std,
                                                                desired_action_stddev=noise_std)
        elif 'normal' in noise_type:
            if 'lin' in noise_type:
                hyperparams['action_noise'] = LinearNormalActionNoise(mean=np.zeros(n_actions),
                                                                      sigma=noise_std * np.ones(n_actions),
                                                                      final_sigma=hyperparams.get('noise_std_final', 0.0) * np.ones(n_actions),
                                                                      max_steps=n_timesteps)
            else:
                hyperparams['action_noise'] = NormalActionNoise(mean=np.zeros(n_actions),
                                                                sigma=noise_std * np.ones(n_actions))
        elif 'ornstein-uhlenbeck' in noise_type:
            hyperparams['action_noise'] = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions),
                                                                       sigma=noise_std * np.ones(n_actions))
        else:
            raise RuntimeError('Unknown noise type "{}"'.format(noise_type))
        logger.info("Applying {} noise with std {}".format(noise_type, noise_std))
        del hyperparams['noise_type']
        del hyperparams['noise_std']
        if 'noise_std_final' in hyperparams:
            del hyperparams['noise_std_final']

    return

def do_hpopt(algo,experiment_params,output_dir):
    if experiment_params.verbose > 0:
        logger.info("Optimizing hyperparameters")

    def create_model(*_args, **kwargs):
        """
        Helper to create a model with different hyperparameters
        """
        return ALGOS[algo](env=create_env(experiment_params.n_envs),
                           tensorboard_log=experiment_params.agent_params.tensorboard_log,
                           verbose=0, **kwargs)

    data_frame = hyperparam_optimization(algo, create_model, create_env, n_trials=experiment_params.n_trials,
                                         n_timesteps=experiment_params.n_timesteps,
                                         hyperparams=experiment_params.agent_params,
                                         n_jobs=experiment_params.n_jobs, seed=experiment_params.seed,
                                         sampler_method=experiment_params.sampler,
                                         pruner_method=experiment_params.pruner,
                                         verbose=experiment_params.verbose)
    env_id = experiment_params.env_params.env_id
    report_name = "report_{}_{}-trials-{}-{}-{}_{}.csv".format(env_id, experiment_params.n_trials,
                                                               experiment_params.n_timesteps,
                                                               experiment_params.sampler,
                                                               experiment_params.pruner, int(time.time()))

    log_path = os.path.join(output_dir, report_name)

    if experiment_params.verbose:
        logger.info("Writing report to {}".format(log_path))

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    data_frame.to_csv(log_path)

    return



def run_experiment(experiment_params):
    # logger = logging.getLogger(LOGGER_NAME)
    seed=experiment_params.seed

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

    logger.info(title("starting experiment seed {seed}",30))

    # create a working directory for the relevant seed
    output_dir=os.path.join(experiment_params.output_root_dir,seed)
    os.makedirs(output_dir, exist_ok=True)

    # logger.info(experiment_params.as_dict())

    # set global seeds
    set_global_seeds(experiment_params.seed)

    agent_hyperparams = experiment_params.agent_params.as_dict()
    exparams_dict = experiment_params.as_dict()

    saved_agent_hyperparams = OrderedDict([(key, agent_hyperparams[key]) for key in sorted(agent_hyperparams.keys())])
    saved_hyperparams = OrderedDict([(key, exparams_dict[key]) for key in exparams_dict.keys()])
    saved_hyperparams['agent_hyperparams']=saved_agent_hyperparams

    # parse algorithm
    algo = agent_hyperparams['algorithm']

    experiment_params.agent_params.seed = seed
    if experiment_params.verbose>0:
        experiment_params.agent_params.verbose = experiment_params.verbose
        experiment_params.agent_params.tensorboard_log = output_dir

    ###################
    # make the env
    n_envs = experiment_params.get('n_envs', 1)
    env_id = experiment_params.env_params.env_id
    logger.info('using {0} instances of {1} :'.format(n_envs,env_id))
    env = create_env(algo,n_envs,experiment_params.env_params)

    # Stop env processes to free memory - not clear....
    if experiment_params.hp_optimize and n_envs > 1:
        env.close()

    #####################
    # create the agent
    if ALGOS[algo] is None:
        raise ValueError('{} requires MPI to be installed'.format(algo))
    n_actions = env.action_space.shape[0]
    parse_agent_params(agent_hyperparams,n_actions)


    if experiment_params.trained_agent != "":
        valid_extension = experiment_params.trained_agent.endswith('.pkl') or experiment_params.trained_agent.endswith('.zip')
        assert valid_extension and os.path.isfile(experiment_params.trained_agent), \
            "The trained_agent must be a valid path to a .zip/.pkl file"

    # todo: handle the case of "her" wrapper.

    if experiment_params.hp_optimize:
        # do_hpopt()
        logger.warning("hyper parameter optimization is not yet supported")
    else:
        normalize = experiment_params.env_params.norm_obs or experiment_params.env_params.norm_reward
        trained_agent = experiment_params.trained_agent
        if trained_agent != "":
            valid_extension = trained_agent.endswith('.pkl') or trained_agent.endswith('.zip')
            assert valid_extension and os.path.isfile(trained_agent), \
                "The trained_agent must be a valid path to a .zip/.pkl file"
            logger.info("loading pretrained agent to continue training")
            # if policy is defined, delete as it will be loaded with the trained agent
            del agent_hyperparams['policy']
            model = ALGOS[algo].load(trained_agent, env=env,**agent_hyperparams)
            exp_folder = trained_agent[:-4]

            if normalize:
                logger.info("Loading saved running average")
                env.load_running_average(exp_folder)

        else:       # create a model from scratch
            model = ALGOS[algo](env=env, **agent_hyperparams)

        kwargs = {}
        if experiment_params.log_interval > -1:
            kwargs = {'log_interval': experiment_params.log_interval}

        model.learn(experiment_params.n_timesteps, **kwargs)

        # Save trained model
        save_path = output_dir
        params_path = "{}/{}".format(save_path, 'model_params')
        os.makedirs(params_path, exist_ok=True)

        # Only save worker of rank 0 when using mpi
        if rank == 0:
            logger.info("Saving to {}".format(save_path))
            model.save(params_path)
            # Save hyperparams
            with open(os.path.join(params_path, 'config.yml'), 'w') as f:
                yaml.dump(saved_hyperparams, f)

            if normalize:
                # Unwrap
                if isinstance(env, VecFrameStack):
                    env = env.venv
                # Important: save the running average, for testing the agent we need that normalization
                env.save_running_average(params_path)


    # print("from within the experiment...")
    logger.info(title("completed experiment seed {seed}",30))
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
    logger = create_logger(experiment_params)
    logger.info(title('Starting {0} experiments'.format(args.num_experiments),40))

    # todo: check if some cmd line arguments should override the experiment params


    for e in range(args.num_experiments):
        seed = args.seed+10*e
        experiment_params.seed = seed
        # run experiment will generate its own sub folder for each seed
        # not yet clear how to support hyper parameter search...
        run_experiment(experiment_params)

    # prepare for shutdown logger
    sys.stdout=sys.__stdout__
    sys.stderr=sys.__stderr__
    logging.shutdown()

    return

if __name__ == '__main__':
    # suppress_tensorflow_warnings()
    main()







