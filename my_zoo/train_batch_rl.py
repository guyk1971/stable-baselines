"""
train_batch_rl.py
train an agent on gym classic control environment.
Supported environments :

"""
# import os
import sys
import time
import argparse
import importlib
# from functools import partial
from collections import OrderedDict
import gym
import numpy as np
import yaml
import ast
from stable_baselines import logger
from stable_baselines.common import set_global_seeds
from stable_baselines.dbcq.expert_dataset import generate_experience_traj
from my_zoo.utils.utils import ALGOS
from stable_baselines.dbcq.dbcq import DBCQ



try:
    import mpi4py
    from mpi4py import MPI
except ImportError:
    mpi4py = None



from my_zoo.utils.common import *
from my_zoo.utils.train import get_create_env,parse_agent_params





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

def env_make(n_envs,env_params,algo,seed):
    env_id = env_params.env_id
    logger.info('using {0} instances of {1} :'.format(n_envs,env_id))
    create_env = get_create_env(algo,seed,env_params)
    env = create_env(n_envs)
    return env


def create_experience_buffer(experiment_params,output_dir):
    '''
    uses the batch_experience_agent to create an experience buffer in a format that can be wrapped by ExpertData
    :param experiment_params: to extract tehe parameters of the batch_experience_agent
    :param output_dir: location of where to save the experience buffer
    :return: experience_buffer that can be consumed (wrapped) by ExpertData
    '''


    trained_agent_params_file = experiment_params.batch_experience_trained_agent
    if trained_agent_params_file:
        trained_agent_params_folder=os.path.dirname(os.path.realpath(trained_agent_params_file))
        yml_filename = os.path.join(trained_agent_params_folder,'config.yml')
        with open(yml_filename, 'r') as f:
             cfg = yaml.load(f, Loader=yaml.UnsafeLoader)
        exp_agent_params = cfg['agent_params']
        # parse the agent params from the loaded yaml
        for k in exp_agent_params.keys():
            try:
                exp_agent_params[k] = ast.literal_eval(exp_agent_params[k])
            except (ValueError,SyntaxError):
                pass
    else:       # no pretrained expert, expect to get the parameters from experiment_params
        exp_agent_params = experiment_params.batch_expert_params.as_dict()

    exp_agent_params['verbose'] = experiment_params.verbose
    exp_agent_params['tensorboard_log'] = output_dir
    algo = exp_agent_params['algorithm']
    seed = exp_agent_params['seed']
    ###################
    # make the env
    n_envs = experiment_params.n_envs
    env = env_make(n_envs,experiment_params.env_params,algo,seed)
    #####################
    # create the agent
    if algo=='random':      # if we simply need a random agent, we're creating a callable object for model
        try:
            _ = env.action_space.sample()
        except:
            raise NotImplementedError("random model assumes gym environment (uses its 'sample' method)")
        def model(obs,gymenv=env):
            action = gymenv.action_space.sample()
            return action
    else:
        if ALGOS[algo] is None:
            raise ValueError('{} requires MPI to be installed'.format(algo))

        n_actions = 1 if isinstance(env.action_space,gym.spaces.Discrete) else env.action_space.shape[0]
        parse_agent_params(exp_agent_params,n_actions,experiment_params.n_timesteps)
        normalize = experiment_params.env_params.norm_obs or experiment_params.env_params.norm_reward

        trained_agent = experiment_params.batch_experience_trained_agent
        if trained_agent_params_file:
            valid_extension = trained_agent_params_file.endswith('.pkl') or trained_agent_params_file.endswith('.zip')
            assert valid_extension and os.path.isfile(trained_agent_params_file), \
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

    # prepare the path to save the expert experience buffer
    exp_agent_algo = experiment_params.batch_expert_params.algorithm
    exp_buf_filename = 'er_'+experiment_params.env_params.env_id+'_'+exp_agent_algo
    exp_buf_filename = os.path.join(output_dir,exp_buf_filename)
    logger.info('Generating experience buffer with ' + exp_agent_algo)
    # at this point, we have 2 options:
    #########################
    # option 1: create it manually
    # kwargs = {}
    # if experiment_params.log_interval > -1:
    #     kwargs = {'log_interval': experiment_params.log_interval}

    # model.learn(int(experiment_params.n_timesteps), **kwargs)
    # experience_buffer = copy.deepcopy(model.replay_buffer)
    # # save the experience buffer
    # experience_buffer.save_for_bc(exp_buf_filename)
    # # experience_buffer is structured to be wrapped by ExpertData
    #########################
    # option 2: use generate_expert_data
    experience_buffer = generate_experience_traj(model, save_path=exp_buf_filename, env=env,
                                                 n_timesteps_train=int(experiment_params.batch_expert_n_timesteps),
                                                 n_timesteps_record=experiment_params.batch_expert_steps_to_record)
    logger.info('Saved experience buffer to ' + exp_buf_filename)
    env.close()
    return experience_buffer

def load_or_create_experience_buffer(experiment_params,output_dir):
    # if we got an existing experience buffer, load from file and return it
    if experiment_params.batch_experience_buffer and os.path.exists(experiment_params.batch_experience_buffer):
        logger.info('loading experience buffer from '+experiment_params.batch_experience_buffer)
        # experience_buffer = ReplayBuffer(experiment_params.batch_experience_agent_params.buffer_size)
        # experience_buffer.load(experiment_params.batch_experience_buffer)
        experience_buffer = np.load(experiment_params.batch_experience_buffer,allow_pickle=True)
        return experience_buffer
    # if we got to this line, we need to generate an experience buffer
    experience_buffer = create_experience_buffer(experiment_params,output_dir)
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
    # set the tensorboard log for the agent
    experiment_params.agent_params.tensorboard_log = output_dir
    with logger.ScopedOutputConfig(output_dir,['csv'],str(seed)):
        # logger.info(experiment_params.as_dict())
        # set the seed for the current worker
        experiment_params.agent_params.seed = seed

        saved_exp_agent_hparams = None
        if experiment_params.batch_expert_params:
            experiment_params.batch_expert_params.seed = seed
            exp_agent_hparams = experiment_params.batch_expert_params.as_dict()
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
        saved_hyperparams['batch_expert_params'] = saved_exp_agent_hparams

        # load or create the experience replay buffer
        er_buf = load_or_create_experience_buffer(experiment_params,output_dir)
        # start batch rl training
        er_buf_size = len(er_buf['obs'])
        logger.info("Experience buffer is ready with {0} samples".format(er_buf_size))
        logger.info(title('start batch training',30))

        ###################
        # make the env for evaluation
        algo = agent_hyperparams['algorithm']
        # batch mode - the envs are only for evaluation
        n_envs = experiment_params.n_envs
        env = env_make(n_envs,experiment_params.env_params,algo,seed)

        #####################
        # create the agent
        n_actions = 1 if isinstance(env.action_space,gym.spaces.Discrete) else env.action_space.shape[0]
        # since the batch algorithm is currently only dbcq, no need to parse agent params
        # but we need to drop the algorithm from the parameters (as done in parse_agent_params)
        # parse_agent_params(agent_hyperparams,n_actions,experiment_params.n_timesteps,logger)
        del agent_hyperparams['algorithm']

        normalize = experiment_params.env_params.norm_obs or experiment_params.env_params.norm_reward
        if normalize:
            logger.warning("normalize observation or reward should be handled in batch mode")

        trained_agent = experiment_params.trained_agent
        if trained_agent:
            valid_extension = trained_agent.endswith('.pkl') or trained_agent.endswith('.zip')
            assert valid_extension and os.path.isfile(trained_agent), \
                "The trained_agent must be a valid path to a .zip/.pkl file"
            logger.info("loading pretrained agent to continue training")
            # if policy is defined, delete as it will be loaded with the trained agent
            del agent_hyperparams['policy']
            # model = ALGOS[algo].load(trained_agent, env=env, replay_buffer=er_buf, **agent_hyperparams)
            model = DBCQ.load(trained_agent, env=env, replay_buffer=er_buf, **agent_hyperparams)

        else:  # create a model from scratch
            # model = ALGOS[algo](env=env, replay_buffer=er_buf, **agent_hyperparams)
            model = DBCQ(env=env, replay_buffer=er_buf, **agent_hyperparams)

        kwargs = {}
        if experiment_params.log_interval > -1:
            kwargs = {'log_interval': experiment_params.log_interval}
        model.learn(int(experiment_params.n_timesteps),tb_log_name='main_agent_train', **kwargs)

        # Save trained model
        save_path = output_dir
        params_path = "{}/{}".format(save_path, 'model_params')
        os.makedirs(params_path, exist_ok=True)

        if rank == 0:
            logger.info("Saving to {}".format(save_path))
            model.save(params_path)
            # Save hyperparams
            # note that in order to save as yaml I need to avoid using classes in the definition.
            # e.g. CustomDQNPolicy will not work. I need to put a string and parse it later
            with open(os.path.join(output_dir, 'config.yml'), 'w') as f:
                yaml.dump(saved_hyperparams, f)
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

    # glogger = create_logger(LOGGER_NAME,experiment_params)
    logger.configure(os.path.join(experiment_params.output_root_dir),['stdout', 'log'])
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
    logging.shutdown()
    return




if __name__ == '__main__':
    suppress_tensorflow_warnings()
    main()
