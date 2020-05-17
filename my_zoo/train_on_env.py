"""
train_gymcc.py
train an agent on gym classic control environment.
Supported environments :
MLA Template Equivalent: train.train_on_env
"""
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
import numpy as np
import gym
import yaml
from stable_baselines import logger
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import VecFrameStack
import shutil
import pandas as pd

try:
    import mpi4py
    from mpi4py import MPI
except ImportError:
    mpi4py = None

from my_zoo.utils.utils import ALGOS
from zoo.utils.hyperparams_opt import hyperparam_optimization
from stable_baselines.gail import ExpertDataset
from my_zoo.utils.common import *
from my_zoo.utils.train import env_make,parse_agent_params,generate_experience_traj,load_experience_traj,\
    online_eval_results_analysis,OnlEvalTBCallback, UniformRandomModel



CONFIGS_DIR = os.path.join(os.path.expanduser('~'),'share','Data','MLA','stbl','configs')
LOGGER_NAME=os.path.splitext(os.path.basename(__file__))[0]




def parse_cmd_line():

    parser = argparse.ArgumentParser()
    parser.add_argument('exparams', type=str, help='experiment params file path')
    parser.add_argument('-d','--gpuid',type=str,default='',help='gpu id or "cpu"')
    parser.add_argument('--num_experiments', help='number of experiments', default=1,type=int)
    parser.add_argument('--seed', help='Random generator seed', type=int, default=1)
    parser.add_argument('-i', '--trained_agent', help='Path to a pretrained agent to continue training',
                        default='', type=str)
    parser.add_argument('-n', '--n_timesteps', help='Overwrite the number of timesteps', default=-1,type=int)
    parser.add_argument('--log_interval', help='Override log interval (default: -1, no change)', default=-1,type=int)
    args = parser.parse_args()
    return args


def do_hpopt(algo,create_env,experiment_params,output_dir):
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
    # set global seeds
    set_global_seeds(experiment_params.seed)

    rank = 0
    if mpi4py is not None and MPI.COMM_WORLD.Get_size() > 1:
        logger.info("Using MPI for multiprocessing with {} workers".format(MPI.COMM_WORLD.Get_size()))
        rank = MPI.COMM_WORLD.Get_rank()
        logger.info("Worker rank: {}".format(rank))
        # make sure that each worker has its own seed
        seed += rank
        # we allow only one worker to "speak"
        if rank != 0:
            experiment_params.verbose = 0

    logger.info(title("starting experiment seed {}".format(seed),30))

    # create a working directory for the relevant seed
    output_dir=os.path.join(experiment_params.output_root_dir,str(seed))
    os.makedirs(output_dir, exist_ok=True)

    with logger.ScopedOutputConfig(output_dir,['csv'],str(seed)):
        # logger.info(experiment_params.as_dict())

        # set the seed for the current worker's agent
        experiment_params.agent_params.seed = seed

        # take a snapshot of the hyper parameters to save
        agent_hyperparams = experiment_params.agent_params.as_dict()
        env_params_dict = experiment_params.env_params.as_dict()
        exparams_dict = experiment_params.as_dict()
        saved_env_params = OrderedDict([(key, str(env_params_dict[key])) for key in sorted(env_params_dict.keys())])
        saved_agent_hyperparams = OrderedDict([(key, str(agent_hyperparams[key])) for key in sorted(agent_hyperparams.keys())])
        saved_hyperparams = OrderedDict([(key, str(exparams_dict[key])) for key in exparams_dict.keys()])
        saved_hyperparams['agent_params']=saved_agent_hyperparams
        saved_hyperparams['env_params'] = saved_env_params

        # parse algorithm
        algo = agent_hyperparams['algorithm']

        if experiment_params.verbose>0:
            experiment_params.agent_params.verbose = experiment_params.verbose
            experiment_params.agent_params.tensorboard_log = output_dir

        ###################
        # make the env
        n_envs = experiment_params.n_envs
        env_id = experiment_params.env_params.env_id
        logger.info('using {0} instances of {1} :'.format(n_envs,env_id))
        env = env_make(n_envs,experiment_params.env_params,algo,seed)

        # Stop env processes to free memory - not clear....
        if experiment_params.hp_optimize and n_envs > 1:
            env.close()

        #####################
        # create the agent
        if algo=='random':      # if we simply need a random agent, we're creating a callable object for model
            model = UniformRandomModel(env)
        else:
            if ALGOS[algo] is None:
                raise ValueError('{} requires MPI to be installed'.format(algo))
            # n_actions = env.action_space.shape[0]
            n_actions = 1 if isinstance(env.action_space,gym.spaces.Discrete) else env.action_space.shape[0]
            parse_agent_params(agent_hyperparams,n_actions,experiment_params.n_timesteps)
            if experiment_params.hp_optimize:
                # do_hpopt(algo,create_env,experiment_params,output_dir)
                raise NotImplementedError('hyper parameter optimization is not yet supported')
            normalize = experiment_params.env_params.norm_obs or experiment_params.env_params.norm_reward
            trained_agent = experiment_params.trained_agent_model_file
            if trained_agent:
                valid_extension = trained_agent.endswith('.pkl') or trained_agent.endswith('.zip')
                assert valid_extension and os.path.isfile(trained_agent), \
                    "The trained_agent must be a valid path to a .zip/.pkl file"
                logger.info("loading pretrained agent to continue training")
                # if policy is defined, delete as it will be loaded with the trained agent
                del agent_hyperparams['policy']
                del agent_hyperparams['policy_kwargs']
                model = ALGOS[algo].load(trained_agent, env=env,**agent_hyperparams)
                exp_folder = trained_agent[:-4]

                if normalize:
                    logger.info("Loading saved running average")
                    env.load_running_average(exp_folder)

            else:       # create a model from scratch
                model = ALGOS[algo](env=env, **agent_hyperparams)

            # check if pretrain(with behavioral cloning) is needed
            if experiment_params.experience_dataset and experiment_params.pretrain_epochs>0:
                logger.info("doing pre-training")
                experience_buffer = load_experience_traj(experiment_params.experience_dataset)
                dataset = ExpertDataset(traj_data=experience_buffer, train_fraction=0.8, batch_size=32,
                                        sequential_preprocessing=True)
                model.pretrain(dataset, n_epochs=experiment_params.pretrain_epochs,
                               learning_rate=experiment_params.pretrain_lr,
                               val_interval=int(experiment_params.pretrain_epochs/5))

            kwargs = {}
            if experiment_params.log_interval > -1:
                kwargs = {'log_interval': experiment_params.log_interval}
            if experiment_params.n_timesteps>0:
                logger.info('training the agent')
                eval_env = create_env(n_envs)
                # in online algorithms, we call the eval callbacks every single step:
                eval_freq = int(experiment_params.evaluation_freq)
                evalcb = None
                if experiment_params.online_eval_n_episodes > 0:
                    # when training on batch we do callback.on_step() every minibatch of samples
                    # thus the online_eval_freq which is given in steps should be converted to minibatches
                    evalcb = OnlEvalTBCallback(eval_env,
                                               n_eval_episodes=experiment_params.online_eval_n_episodes,
                                               eval_freq=eval_freq,
                                               log_path=output_dir, best_model_save_path=output_dir)

                model.learn(int(experiment_params.n_timesteps),callback=evalcb, **kwargs)
                # save evaluation report if needed
                if experiment_params.online_eval_n_episodes > 0:
                    online_eval_results_analysis(os.path.join(output_dir,'evaluations.npz'))
                # Save trained model
                save_path = output_dir
                params_path = "{}/{}".format(save_path, 'final_model_params')
                # os.makedirs(params_path, exist_ok=True)
                # Only save worker of rank 0 when using mpi
                if rank == 0:
                    logger.info("Saving to {}".format(save_path))
                    model.save(params_path)
                    # Save hyperparams
                    # note that in order to save as yaml I need to avoid using classes in the definition.
                    # e.g. CustomDQNPolicy will not work. I need to put a string and parse it later
                    with open(os.path.join(output_dir, 'config.yml'), 'w') as f:
                        yaml.dump(saved_hyperparams, f)

                    if normalize:
                        # Unwrap
                        if isinstance(env, VecFrameStack):
                            env = env.venv
                        # Important: save the running average, for testing the agent we need that normalization
                        env.save_running_average(params_path)
    # from now on we treat the trained agent as expert.
    # check if we need to create expert trajectory buffer
    if experiment_params.expert_steps_to_record > 0:
        exp_agent_algo = algo
        exp_buf_filename = 'er_'+experiment_params.env_params.env_id+'_'+exp_agent_algo+'_'+str(experiment_params.expert_steps_to_record)
        exp_buf_filename = os.path.join(output_dir,exp_buf_filename)
        logger.info('Generating expert experience buffer with ' + exp_agent_algo)
        _ = generate_experience_traj(model, save_path=exp_buf_filename, env=env,
                                     n_timesteps_record=experiment_params.expert_steps_to_record)

    logger.info(title("completed experiment seed {}".format(seed),30))
    return




def main():

    args = parse_cmd_line()
    print('reading experiment params from '+args.exparams)

    module_path = 'my_zoo.hyperparams.'+args.exparams
    exp_params_module = importlib.import_module(module_path)
    experiment_params = getattr(exp_params_module,'experiment_params')
    # set the path to the config file
    hyperparams_folder_path=os.path.join(os.path.dirname(os.path.realpath(__file__)),'hyperparams')
    exparams_path=os.path.join(hyperparams_folder_path,args.exparams+'.py')

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
    # copy the configuration file
    shutil.copy(exparams_path,experiment_params.output_root_dir)

    # global logger
    # logger = create_logger(experiment_params,stdout_to_log=False)
    logger.configure(os.path.join(experiment_params.output_root_dir), ['stdout', 'log'])

    # check if some cmd line arguments should override the experiment params
    if args.n_timesteps > -1:
        logger.info("overriding n_timesteps with {}".format(args.n_timesteps))
        experiment_params.n_timesteps=args.n_timesteps
    if args.log_interval > -1:
        logger.info("overriding log_interval with {}".format(args.log_interval))
        experiment_params.log_interval=args.log_interval
    if args.trained_agent != '':
        logger.info("overriding trained_agent with {}".format(args.trained_agent))
        experiment_params.trained_agent=args.trained_agent

    logger.info(title('Starting {0} experiments'.format(args.num_experiments), 40))
    for e in range(args.num_experiments):
        seed = args.seed+100*e
        experiment_params.seed = seed
        # run experiment will generate its own sub folder for each seed
        # not yet clear how to support hyper parameter search...
        run_experiment(experiment_params)

    # prepare for shutdown logger
    logging.shutdown()

    return

if __name__ == '__main__':
    suppress_tensorflow_warnings()
    main()
    sys.path.remove(proj_root)







