# train related util functions.
# equivalent to train_common.py in MLA Template

import os
import warnings
import gym
from gym import spaces
import numpy as np
import pandas as pd
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack, VecEnv, VecNormalize, DummyVecEnv
from stable_baselines.common.noise import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines.common.schedules import get_schedule_fn
from tqdm import tqdm

from stable_baselines import logger
from stable_baselines.common.base_class import BaseRLModel,_UnvecWrapper
from my_zoo.utils.common import title
from stable_baselines.common.buffers import ReplayBuffer



try:
    import mpi4py
    from mpi4py import MPI
except ImportError:
    mpi4py = None

from my_zoo.utils.utils import make_env, linear_schedule, get_wrapper_class
from zoo.utils.noise import LinearNormalActionNoise

from my_zoo.hyperparams.default_config import CC_ENVS
from my_zoo.my_envs import L2PEnv


def get_create_env(algo,seed,env_params):
    # logger = logging.getLogger(LOGGER_NAME)
    env_id = CC_ENVS.get(env_params.env_id,None)
    assert env_id , "env {0} is not supported".format(env_id)
    is_atari= 'NoFrameskip' in env_id
    # n_envs = experiment_params.get('n_envs', 1)

    normalize_kwargs = {'norm_obs':env_params.norm_obs,'norm_reward':env_params.norm_reward}
    normalize = normalize_kwargs['norm_obs'] or normalize_kwargs['norm_reward']

    # obtain a class object from a wrapper name string in hyperparams
    # and delete the entry
    env_wrapper = get_wrapper_class(env_params.as_dict()) if env_params.env_wrapper else None

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
        elif algo in ['dqn', 'ddpg', 'random','dbcq','qrdqn']:
            if normalize:
                logger.warn("WARNING: normalization not supported yet for DDPG/DQN/DBCQ/QRDQN")
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
    return _create_env
    # return partial(_create_env,env_wrapper)


def parse_agent_params(hyperparams,n_actions,n_timesteps):

    algo = hyperparams['algorithm']
    del hyperparams['algorithm']

    # Parse the schedule parameters for the relevant algorithms
    if algo in ["ppo2", "sac", "td3", "dbcq"]:
        for key in ['learning_rate', 'cliprange', 'cliprange_vf']:
            if key not in hyperparams or hyperparams[key] is None:
                continue
            if isinstance(hyperparams[key], str):       # currently supporting only linear scheduling (e.g. lin_0.0001)
                schedule, initial_value = hyperparams[key].split('_')
                initial_value = float(initial_value)
                hyperparams[key] = linear_schedule(initial_value)
            elif isinstance(hyperparams[key], (float, int)):
                # Negative value: ignore (ex: for clipping)
                if hyperparams[key] < 0:
                    continue
                hyperparams[key] = get_schedule_fn(float(hyperparams[key]))
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

def env_make(n_envs,env_params,algo,seed):
    env_id = env_params.env_id
    logger.info('using {0} instances of {1} :'.format(n_envs, env_id))
    if env_id=='L2P':
        env = L2PEnv()
    else:
        create_env = get_create_env(algo,seed,env_params)
        env = create_env(n_envs)
    return env

def online_eval_results_analysis(npz_filename):
    if not os.path.exists(npz_filename):
        logger.warn('evaluation results file not found')
        return

    eval_np = np.load(npz_filename)
    eval_dict = {k:v for k,v in eval_np.items()}
    # replace 'results' with 'mean_rew' and 'std_rew':
    eval_dict['reward_mean'] = np.squeeze(eval_dict['results']).mean(axis=1)
    eval_dict['reward_std'] = np.squeeze(eval_dict['results']).std(axis=1)
    del eval_dict['results']
    eval_dict['ep_lengths_mean'] = eval_dict['ep_lengths'].mean(axis=1)
    eval_dict['ep_lengths_std']= eval_dict['ep_lengths'].std(axis=1)
    del eval_dict['ep_lengths']
    eval_df = pd.DataFrame(eval_dict)
    eval_df.set_index('timesteps',inplace=True)
    # save csv filename
    df_filename = os.path.splitext(npz_filename)[0]+'.csv'
    eval_df.to_csv(df_filename)
    return


# batch_rl utils
def load_experience_traj(csv_path):
    # check if there's a numpy_dict version already saved (to save csv load time)
    npz_filename = os.path.splitext(csv_path)[0]+'.npz'
    if os.path.exists(npz_filename):
        logger.info("found cached version of experience csv file. loading it")
        numpy_dict = np.load(npz_filename, allow_pickle=True)
    else:
        logger.info("loading from csv and saving a cache file in "+npz_filename)
        buf=ReplayBuffer(size=1)        # size will be overrun by the csv size
        episode_starts,episode_returns=buf.load_from_csv(csv_path)
        numpy_dict = buf.record_buffer()
        numpy_dict['episode_returns'] = np.array(episode_returns)
        numpy_dict['episode_starts'] = np.array(episode_starts)
        np.savez(npz_filename, **numpy_dict)
    return numpy_dict


def generate_experience_traj(model, save_path=None, env=None, n_timesteps_train=0,
                         n_timesteps_record=100000,deterministic=True):
    """
    Train expert controller (if needed) and record expert trajectories.

    .. note::

        only Box and Discrete spaces are supported for now.
        support in images is removed.

    :param model: (RL model or callable) The expert model, if it needs to be trained,
        then you need to pass ``n_timesteps > 0``.
        note that the RL model can be also a pretrained expert that was loaded from file.
    :param save_path: (str) Path without the extension where the expert dataset will be saved
        (ex: 'expert_cartpole' -> creates 'expert_cartpole.npz').
        If not specified, it will not save, and just return the generated expert trajectories.
        This parameter must be specified for image-based environments.
    :param env: (gym.Env) The environment, if not defined then it tries to use the model
        environment.
    :param n_timesteps_train: (int) Number of training timesteps
    :param n_timesteps_record: (int) Number of trajectories (episodes) to record
    :param logger: (Logger) - if not None, use it for verbose output
    :return: (dict) the generated expert trajectories.
    """

    # Retrieve the environment using the RL model
    if env is None and isinstance(model, BaseRLModel):
        env = model.get_env()

    assert env is not None, "You must set the env in the model or pass it to the function."

    is_vec_env = False
    if isinstance(env, VecEnv) and not isinstance(env, _UnvecWrapper):
        is_vec_env = True
        if env.num_envs > 1:
            warnings.warn("You are using multiple envs, only the data from the first one will be recorded.")

    # Sanity check
    assert (isinstance(env.observation_space, spaces.Box) or
            isinstance(env.observation_space, spaces.Discrete)), "Observation space type not supported"

    assert (isinstance(env.action_space, spaces.Box) or
            isinstance(env.action_space, spaces.Discrete)), "Action space type not supported"

    # Note: support in recording image to files is omitted
    obs_space = env.observation_space
    replay_buffer = ReplayBuffer(n_timesteps_record)

    logger.info(title("generate expert trajectory",20))

    if n_timesteps_train > 0 and isinstance(model, BaseRLModel):
        logger.info("training expert start - {0} timesteps".format(n_timesteps_train))
        model.learn(n_timesteps_train,tb_log_name='exp_gen_train')
        logger.info("generate expert trajectory: training expert end")


    logger.info("start recording {0} expert steps".format(n_timesteps_record))
    episode_returns = []
    episode_starts = []

    ep_idx = 0
    obs = env.reset()
    episode_starts.append(True)
    reward_sum = 0.0
    # state and mask for recurrent policies
    state, mask = None, None
    if is_vec_env:
        mask = [True for _ in range(env.num_envs)]
    for t in tqdm(range(n_timesteps_record)):
        if isinstance(model, BaseRLModel):
            action, state = model.predict(obs, state=state, mask=mask,deterministic=deterministic)
        else:
            action = model(obs)
        new_obs, reward, done, info = env.step(action)

        # Note : we save to the experience buffer as if it is not a vectorized env since anyway we
        #        use only first env
        if is_vec_env:
            mask = [done[0] for _ in range(env.num_envs)]
            action = np.array([action[0]])
            reward = np.array(reward[0])
            done = np.array([done[0]])
            replay_buffer.add(obs[0],action[0],reward,new_obs[0],float(done[0]))
        else: # Store transition in the replay buffer.
            replay_buffer.add(obs, action, reward, new_obs, float(done))
        obs = new_obs
        episode_starts.append(done)
        reward_sum += reward
        if done:
            if not is_vec_env:
                obs = env.reset()
                # Reset the state in case of a recurrent policy
                state = None
            episode_returns.append(reward_sum)
            reward_sum = 0.0
            ep_idx += 1

    logger.info("finished collecting experience data")
    numpy_dict = replay_buffer.record_buffer()
    # Note : the ReplayBuffer can not generally assume it has not circled around thus cant infer accurate episode
    # statistics. since in this context we know these details, we overwrite the corresponding fields:
    numpy_dict['episode_returns'] = np.array(episode_returns)
    numpy_dict['episode_starts'] = np.array(episode_starts)
    logger.info("Statistics: {0} episodes, average return={1}".format(len(episode_returns),
                                                                      np.mean(numpy_dict['episode_returns'])))
    # assuming we save only the numpy arrays (not the obs_space and act_space)
    if save_path is not None:
        np.savez(save_path+'.npz', **numpy_dict)
        logger.info("saving to experience as csv file: "+save_path+'.csv')
        replay_buffer.save_to_csv(save_path+'.csv',os.path.splitext(os.path.basename(save_path))[0])

    env.close()
    return numpy_dict

