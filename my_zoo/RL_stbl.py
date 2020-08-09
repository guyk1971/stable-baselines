import os
import numpy as np
import pickle
from tqdm import tqdm
import argparse


def feature_extraction_scarlet(data, **params):
    # declare feature list
    feature_of_dict = {'power_to_curpl1_mean':None,'power_to_curpl1_std':None,'power_to_maxpl1_mean':None,
                       'power_to_maxpl1_std':None,'pl1_to_pl2':None, 'ips_mean_mean':None, 'ips_mean_std':None,
                       'tskin_slope':None,'tj_slope':None,'tmem_slope':None,'power_to_curpl1':None, 'power_to_maxpl1':None,
                       'pl1':None,'pl2':None,'tskin':None,'tj':None,'tmem':None,'ips_mean':None,'torbu':None}
    if data is not None:
        # parse params
        pl1_max = params['pl1_max']
        pl1_min = params['pl1_min']
        pl2_max = params['pl2_max']
        pl2_min = params['pl2_min']
        tskin_max = params['tskin_max']
        tskin_idle = params['tskin_idle']
        tj_max = params['tj_max']
        tj_idle = params['tj_idle']
        tmem_max = params['tmem_max']
        tmem_idle = params['tmem_idle']
        ips_max = params['ips_max']
        ewma_power = params['ewma_power']
        normlized_params = params.get('normlized_params', None)

        # extract features
        n_frames = len(data['POWER'])
        feature_of_dict['power_to_curpl1'] = [data['POWER'][i] / data['MMIO_PL1'][i] for i in range(n_frames)]
        feature_of_dict['power_to_maxpl1'] = [data['POWER'][i] / pl1_max for i in range(n_frames)]
        # feature_of_dict['PKG_C0'] = [data['PKG_C0'][i] for i in range(n_frames)]
        feature_of_dict['pl1'] = [(data['MMIO_PL1'][i] - pl1_min) / (pl1_max - pl1_min) for i in range(n_frames)]
        feature_of_dict['pl2'] = [(data['MMIO_PL2'][i] - pl2_min) / (pl2_max - pl2_min) for i in range(n_frames)]
        feature_of_dict['tskin'] = [(data['tskin'][i] - tskin_idle) / (tskin_max - tskin_idle) for i in
                                    range(n_frames)]
        feature_of_dict['tj'] = [(data['tj'][i] - tj_idle) / (tj_max - tj_idle) for i in range(n_frames)]
        feature_of_dict['tmem'] = [(data['tmem'][i] - tmem_idle) / (tmem_max - tmem_idle) for i in range(n_frames)]



        ###################################
        # feature start
        # feature_of_dict['action_pl1'] = data['MMIO_PL1'][-1] - data['MMIO_PL1'][-2]
        # feature_of_dict['action_pl2'] = data['MMIO_PL1'][-1] - data['MMIO_PL1'][-2]

        feature_of_dict['power_to_curpl1_mean'] = np.mean(feature_of_dict['power_to_curpl1'])
        feature_of_dict['power_to_curpl1_std'] = np.std(feature_of_dict['power_to_curpl1'])
        feature_of_dict['power_to_maxpl1_mean'] = np.mean(feature_of_dict['power_to_maxpl1'])
        feature_of_dict['power_to_maxpl1_std'] = np.std(feature_of_dict['power_to_maxpl1'])

        feature_of_dict['pl1_to_pl2'] = (data['MMIO_PL1'][-1] - pl1_min) / (data['MMIO_PL2'][-1] - pl1_min)

        # feature_of_dict['PKG_C0_mean'] = np.mean(feature_of_dict['PKG_C0'])
        # feature_of_dict['PKG_C0_std'] = np.std(feature_of_dict['PKG_C0'])

        # feature_of_dict['ips_max_mean'] = np.mean(data['ips_max'])
        # feature_of_dict['ips_max_std'] = np.std(data['ips_max'])

        # feature_of_dict['ips_min_mean'] = np.mean(data['ips_min'])
        # feature_of_dict['ips_min_std'] = np.std(data['ips_min'])

        feature_of_dict['ips_mean_mean'] = np.mean(data['ips_mean'])/ips_max
        feature_of_dict['ips_mean_std'] = np.std(data['ips_mean'])/ips_max      # need ^2 ? looks like not.

        # feature_of_dict['ips_std_mean'] = np.mean(data['ips_std'])
        # feature_of_dict['ips_std_std'] = np.std(data['ips_std'])

        feature_of_dict['tskin_slope'] = (feature_of_dict['tskin'][-1] - feature_of_dict['tskin'][0]) / n_frames
        feature_of_dict['tj_slope'] = (feature_of_dict['tj'][-1] - feature_of_dict['tj'][0]) / n_frames
        feature_of_dict['tmem_slope'] = (feature_of_dict['tmem'][-1] - feature_of_dict['tmem'][0]) / n_frames


        feature_of_dict['power_to_curpl1'] = feature_of_dict['power_to_curpl1'][-1]
        feature_of_dict['power_to_maxpl1'] = feature_of_dict['power_to_maxpl1'][-1]
        # feature_of_dict['PKG_C0'] = feature_of_dict['PKG_C0'][-1]
        feature_of_dict['pl1'] = feature_of_dict['pl1'][-1]
        feature_of_dict['pl2'] = feature_of_dict['pl2'][-1]
        feature_of_dict['tskin'] = feature_of_dict['tskin'][-1]
        feature_of_dict['tj'] = feature_of_dict['tj'][-1]
        feature_of_dict['tmem'] = feature_of_dict['tmem'][-1]

        # feature_of_dict['ips_max'] = data['ips_max'][-1]
        # feature_of_dict['ips_min'] = data['ips_min'][-1]
        feature_of_dict['ips_mean'] = data['ips_mean'][-1]/ips_max
        # feature_of_dict['ips_std'] = data['ips_std'][-1]

        feature_of_dict['torbu'] = ewma_power / pl1_max
        ################################## 17 features ######################################
        # z-normalizing with train parameters (assuming gaussian distribution of each of the features)
        if normlized_params:
            for feature in feature_of_dict.keys():
                feature_of_dict[feature] = (feature_of_dict[feature] - normlized_params[feature]['mean']) / \
                                           normlized_params[feature]['std']

    return feature_of_dict


def feature_extraction_scarlet_ns(data, **params):
    # declare feature list
    feature_of_dict = {'power_to_curpl1_mean':None,'power_to_curpl1_std':None,'power_to_maxpl1_mean':None,
                       'power_to_maxpl1_std':None,'pl1_to_pl2':None, 'ips_mean_mean':None, 'ips_mean_std':None,
                       'tj_slope':None,'tmem_slope':None,'power_to_curpl1':None, 'power_to_maxpl1':None,
                       'pl1':None,'pl2':None,'tj':None,'tmem':None,'ips_mean':None,'torbu':None}
    if data is not None:
        # parse params
        pl1_max = params['pl1_max']
        pl1_min = params['pl1_min']
        pl2_max = params['pl2_max']
        pl2_min = params['pl2_min']
        tj_max = params['tj_max']
        tj_idle = params['tj_idle']
        tmem_max = params['tmem_max']
        tmem_idle = params['tmem_idle']
        ips_max = params['ips_max']
        ewma_power = params['ewma_power']
        normlized_params = params.get('normlized_params', None)

        # extract features
        n_frames = len(data['POWER'])
        feature_of_dict['power_to_curpl1'] = [data['POWER'][i] / data['MMIO_PL1'][i] for i in range(n_frames)]
        feature_of_dict['power_to_maxpl1'] = [data['POWER'][i] / pl1_max for i in range(n_frames)]
        # feature_of_dict['PKG_C0'] = [data['PKG_C0'][i] for i in range(n_frames)]
        feature_of_dict['pl1'] = [(data['MMIO_PL1'][i] - pl1_min) / (pl1_max - pl1_min) for i in range(n_frames)]
        feature_of_dict['pl2'] = [(data['MMIO_PL2'][i] - pl2_min) / (pl2_max - pl2_min) for i in range(n_frames)]
        feature_of_dict['tj'] = [(data['tj'][i] - tj_idle) / (tj_max - tj_idle) for i in range(n_frames)]
        feature_of_dict['tmem'] = [(data['tmem'][i] - tmem_idle) / (tmem_max - tmem_idle) for i in range(n_frames)]



        ###################################
        # feature start
        # feature_of_dict['action_pl1'] = data['MMIO_PL1'][-1] - data['MMIO_PL1'][-2]
        # feature_of_dict['action_pl2'] = data['MMIO_PL1'][-1] - data['MMIO_PL1'][-2]

        feature_of_dict['power_to_curpl1_mean'] = np.mean(feature_of_dict['power_to_curpl1'])
        feature_of_dict['power_to_curpl1_std'] = np.std(feature_of_dict['power_to_curpl1'])
        feature_of_dict['power_to_maxpl1_mean'] = np.mean(feature_of_dict['power_to_maxpl1'])
        feature_of_dict['power_to_maxpl1_std'] = np.std(feature_of_dict['power_to_maxpl1'])

        feature_of_dict['pl1_to_pl2'] = (data['MMIO_PL1'][-1] - pl1_min) / (data['MMIO_PL2'][-1] - pl1_min)

        # feature_of_dict['PKG_C0_mean'] = np.mean(feature_of_dict['PKG_C0'])
        # feature_of_dict['PKG_C0_std'] = np.std(feature_of_dict['PKG_C0'])

        # feature_of_dict['ips_max_mean'] = np.mean(data['ips_max'])
        # feature_of_dict['ips_max_std'] = np.std(data['ips_max'])

        # feature_of_dict['ips_min_mean'] = np.mean(data['ips_min'])
        # feature_of_dict['ips_min_std'] = np.std(data['ips_min'])

        feature_of_dict['ips_mean_mean'] = np.mean(data['ips_mean'])/ips_max
        feature_of_dict['ips_mean_std'] = np.std(data['ips_mean'])/ips_max      # need ^2 ? looks like not.

        # feature_of_dict['ips_std_mean'] = np.mean(data['ips_std'])
        # feature_of_dict['ips_std_std'] = np.std(data['ips_std'])

        feature_of_dict['tj_slope'] = (feature_of_dict['tj'][-1] - feature_of_dict['tj'][0]) / n_frames
        feature_of_dict['tmem_slope'] = (feature_of_dict['tmem'][-1] - feature_of_dict['tmem'][0]) / n_frames


        feature_of_dict['power_to_curpl1'] = feature_of_dict['power_to_curpl1'][-1]
        feature_of_dict['power_to_maxpl1'] = feature_of_dict['power_to_maxpl1'][-1]
        # feature_of_dict['PKG_C0'] = feature_of_dict['PKG_C0'][-1]
        feature_of_dict['pl1'] = feature_of_dict['pl1'][-1]
        feature_of_dict['pl2'] = feature_of_dict['pl2'][-1]
        feature_of_dict['tj'] = feature_of_dict['tj'][-1]
        feature_of_dict['tmem'] = feature_of_dict['tmem'][-1]

        # feature_of_dict['ips_max'] = data['ips_max'][-1]
        # feature_of_dict['ips_min'] = data['ips_min'][-1]
        feature_of_dict['ips_mean'] = data['ips_mean'][-1]/ips_max
        # feature_of_dict['ips_std'] = data['ips_std'][-1]

        feature_of_dict['torbu'] = ewma_power / pl1_max
        ################################## 17 features ######################################
        # z-normalizing with train parameters (assuming gaussian distribution of each of the features)
        if normlized_params:
            for feature in feature_of_dict.keys():
                feature_of_dict[feature] = (feature_of_dict[feature] - normlized_params[feature]['mean']) / \
                                           normlized_params[feature]['std']

    return feature_of_dict


FEATURE_EXTRACTORS={0: feature_extraction_scarlet,2:feature_extraction_scarlet_ns}




def parse_cmd_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--n_episodes', help='number of episodes', default=30,type=int)
    parser.add_argument('-b', '--benchmark', help='benchmark to run', type=str, default='cb20mr')
    parser.add_argument('--pf', help='fixed policy', type=int, nargs=2, action='append')
    parser.add_argument('--pg', help='greedy policy', type=bool)
    parser.add_argument('--pm', help='load policy model from path', type=str)
    parser.add_argument('-fe', '--featurext', help='feature extractor', default=0, type=int)
    parser.add_argument('-r', '--reward', help='reward function',default=0, type=int)
    parser.add_argument('--platform', help='type of platform: Scarlet', default='ScarletM', type=str)
    parser.add_argument('-v','--verbose',help='verbose will create esif file', action='store_true')
    args = parser.parse_args()
    return args


def sim_calc_power_limits():
    args = parse_cmd_line()
    platform = PLATFORMS[args.platform]
    n_episodes=args.n_episodes
    episode_workloads = EPISODES[args.benchmark]
    log_output=os.path.join(os.getcwd(),'tmp') if args.verbose else None
    featurext = FEATURE_EXTRACTORS[args.featurext]
    reward=REWARD_FUNC[args.reward]
    env = DTTEnvSim(platform, episode_workloads=episode_workloads, norm_obs=False, log_output=log_output)
    env = DTTStateRewardWrapper(env=env,feature_extractor=featurext,reward_calc=reward,n_frames=5)
    dPL2act = {v: np.int64(k) for k, v in env.dPL.items()}
    feat_cols = list(featurext(None))
    print(f'running {n_episodes} of {args.benchmark} with feature {args.featurext} and reward {args.reward}')
    # define the policy
    policies = {'random':random_policy}
    policy=random_policy
    if args.pm and os.path.exists(args.pm):  # assuming args.pm is the path to where the agent_params.py and model_params.zip are
        # load model from file
        policy = load_stbl_model(args.pm,env)
        print(f'loaded policy from {args.pm}')
    else:
        print('using random policy')
    tsdf = pd.DataFrame(columns=feat_cols)
    feat_csv = os.path.join(os.getcwd(),'tmp','sim_features.csv')
    tsdf.to_csv(feat_csv,index=False)
    for ei in tqdm(range(n_episodes)):
        obs = env.reset()
        done = False
        out_df = pd.DataFrame(columns=feat_cols)
        ts = 0
        total_rew = 0
        while not done:
            act = policy(obs)
            obs, rew, done, info = env.step(act)
            total_rew += rew
            # obs = ['pl1','pl2','power','tj','tskin','tmem','ewma']
            # policy : as long as we're below the max value, aim to increase
            tsdf.loc[0] = list(obs)
            out_df = out_df.append(tsdf)
            ts += 1
        scores = env.get_scores()
        avg_score = round(np.mean(scores), 2)
        print(f'episode {ei} completed. total reward: {round(total_rew, 2)}, scores:{scores}')
        out_df.to_csv(feat_csv, mode='a', header=False, index=False)
    env.close()





# if we run this script as main file, we'll run on the simulated environment.
# else, we're running on the real platform


if __name__ == '__main__':
    import sys
    path_to_curr_file = os.path.realpath(__file__)
    proj_root = os.path.dirname(os.path.dirname(path_to_curr_file))
    print(proj_root)
    if proj_root not in sys.path:
        sys.path.insert(0, proj_root)
    from my_zoo.my_envs import PLATFORMS,DTTEnvSim,EPISODES,random_policy
    from my_zoo.dttsim_wrappers import DTTStateRewardWrapper,reward_0,reward_3,reward_6,reward_7
    from my_zoo.deploy_stbl_tf import load_stbl_model,suppress_tensorflow_warnings
    import pandas as pd
    os.makedirs('tmp', exist_ok=True)
    suppress_tensorflow_warnings()
    REWARD_FUNC = {0: reward_0, 2: reward_2, 3: reward_3,6: reward_6, 7:reward_7}
    sim_calc_power_limits()
else:
    import configparser
    from deploy_stbl_tf import load_stbl_model,suppress_tensorflow_warnings
    import gym
    from gym import spaces
    import os

    class DTTEnvReal(gym.Env):
        """
        Custom Environment that follows gym interface.
        This is a simple env that imitates the L2P behaviour
        """
        # Because of google colab, we cannot implement the GUI ('human' render mode)
        metadata = {'render.modes': ['console']}

        def __init__(self, obs_dim=31, n_act=9):
            super(DTTEnvReal, self).__init__()

            # the observation space include obs_dim float values
            self.obs_dim = obs_dim
            # Currently assuming discrete action space with n_act actions
            self.act_dim = 1
            # Define action and observation space
            # They must be gym.spaces objects
            # Example when using discrete actions, we have two: left and right
            self.action_space = spaces.Discrete(n_act)
            # The observation will be the coordinate of the agent
            # this can be described both by Discrete and Box space
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            self.max_path_length = 1000  # arbitrary number

        def reset(self):
            """
            Important: the observation must be a numpy array
            :return: (np.array)
            """
            self.step_idx = 0
            return self.observation_space.sample()

        def step(self, action):
            '''
            Currently a dummy function that should not be called.
            :param action:
            :return:
            '''
            if (not isinstance(action, int)) or (action < 0) or (action >= self.action_space.n):
                raise ValueError("Received invalid action={} which is not part of the action space".format(action))
            self.step_idx += 1

            state = self.observation_space.sample()
            done = False
            if self.step_idx == self.max_path_length:
                done = True
                self.step_idx = 0
            reward = 1.0

            # Optionally we can pass additional info, we are not using that for now
            info = {}

            return state, reward, done, info

        def render(self, mode='console'):
            if mode != 'console':
                raise NotImplementedError()

        def close(self):
            pass

    my_version = 'RL_stbl'
    g = {'index_num': 0, 'POWER': [], 'PKG_C0': [], 'tj': [], 'tskin': [],
         'ips_mean': [], 'ips_std': [], 'ips_max': [], 'ips_min': [],
         'MMIO_PL1': [], 'MMIO_PL2': []}
    config = configparser.ConfigParser()
    config.sections()
    # config.read(
    #     r"C:\Users\awagner\OneDrive - Intel Corporation\Documents\GitHub\dtt_rl\deployment\calc_power_limit_versions\my_config_hyper_test.ini")
    config.read(r"my_config_hyper_test.ini")
    model_filename = r'tmp\dqn_cb20_f2_r0.zip'
    feature_extractor = FEATURE_EXTRACTORS[2]       # if 'f2' in model name then its feature extractor 2
    n_features = len(list(feature_extractor(None)))
    env = DTTEnvReal(obs_dim=n_features)
    policy = load_stbl_model(model_filename, env)

    # extract platform parameters from config files
    pl1_min = int(config[my_version]['pl1_min'])
    pl2_max = int(config[my_version]['pl2_max'])
    pl1_max = int(config[my_version]['pl1_max'])
    pl2_min = int(config[my_version]['pl2_min'])
    tskin_max = int(config[my_version]['tskin_max'])
    tskin_idle = int(config[my_version]['tskin_idle'])
    tmem_max = int(config[my_version]['tmem_max'])
    tmem_idle = int(config[my_version]['tmem_idle'])
    tj_max = int(config[my_version]['tj_max'])
    tj_idle = int(config[my_version]['tj_idle'])
    ips_max = int(config[my_version]['ips_max'])

    epslion_of_choice = 0
    ewma_power = 0
    curr_pl1_pl2 = {'pl1': pl1_max, 'pl2': pl2_max}
    Tau = 28
    n2a_dict = {0:(-500,-500), 1:(0,-500), 2:(500,-500),
                3:(-500,0), 4:(0,0), 5:(500,0),
                6:(-500,500), 7:(0,500), 8:(500,500)}
def num_to_action(num):
    dPL=n2a_dict[num]
    return {'diffp1': float(dPL[0]), 'diffp2': float(dPL[1])}


def calc_power_limits(features):
    # note that we dont have both tskin and tmem.
    # This function assumes that the value of tmem is provided through features['tskin']

    global g, pl1_max, pl1_min, tskin_max, tskin_idle, tmem_max, tmem_idle, pl2_max, pl2_min, tj_max, tj_idle, \
        ips_max, ewma_power, normlized_params, TAU

    g['POWER'].append(features['pkgPower'])
    # g['PKG_C0'].append(features['pkgC0'])
    g['tj'].append(features['tj'] / 10.0 - 273.15)
    # g['tskin'].append(features['tskin'] / 10.0 - 273.15)
    # g['tmem'].append(features['tmem'] / 10.0 - 273.15)
    g['tmem'].append(features['tskin'] / 10.0 - 273.15)
    g['MMIO_PL1'].append(features['mmioPl1'])
    g['MMIO_PL2'].append(features['mmioPl2'])

    ewma_power = ewma_power * (np.exp(-1 / TAU)) + (1 - np.exp(-1 / TAU)) * (curr_pl1_pl2['pl1'] - features['pkgPower'])
    cpus_delta = []
    for cpu in range(8):
        cpus_delta.append(features['instructions']['cpu' + str(cpu) + '_instructions_delta'])

    g['ips_mean'].append(np.mean(cpus_delta))
    # g['ips_std'].append(np.std(cpus_delta))
    # g['ips_max'].append(np.max(cpus_delta))
    # g['ips_min'].append(np.min(cpus_delta))

    choose_random_prob = np.random.binomial(1, epslion_of_choice)

    if len(g['POWER']) < 6:
        curr_pl1_pl2['pl1'] = pl1_max
        curr_pl1_pl2['pl2'] = pl2_max
        q_action = 4
        my_features = 0
        if len(g['POWER']) == 1:
            ewma_power = (curr_pl1_pl2['pl1'] - features['pkgPower'])
    else:
        g['POWER'] = g['POWER'][1:]
        # g['PKG_C0'] = g['PKG_C0'][1:]
        g['tj'] = g['tj'][1:]
        # g['tskin'] = g['tskin'][1:]
        g['tmem'] = g['tmem'][1:]
        g['ips_mean'] = g['ips_mean'][1:]
        # g['ips_max'] = g['ips_max'][1:]
        # g['ips_min'] = g['ips_min'][1:]
        # g['ips_std'] = g['ips_std'][1:]
        g['MMIO_PL1'] = g['MMIO_PL1'][1:]
        g['MMIO_PL2'] = g['MMIO_PL2'][1:]
        my_features=None
        if choose_random_prob:
            q_action = np.random.randint(0, 9)
        else:
            features_dict = feature_extractor(g, pl1_max=pl1_max, pl1_min=pl1_min, tmem_max=tmem_max,tmem_idle=tmem_idle,
                                              pl2_max=pl2_max, pl2_min=pl2_min,tj_max=tj_max,tj_idle=tj_idle,
                                              ips_max=ips_max,ewma_power=ewma_power)
            my_features = np.fromiter(features_dict.values(),dtype=float)
            q_action = policy(my_features)

        diff_action = num_to_action(q_action)

        curr_pl1_pl2['pl1'] = g['MMIO_PL1'][-1]
        curr_pl1_pl2['pl2'] = g['MMIO_PL2'][-1]


        curr_pl1_pl2['pl1'] += diff_action['diffp1']
        curr_pl1_pl2['pl2'] += diff_action['diffp2']

    # clip pl2 to limits
    curr_pl1_pl2['pl2'] = np.min([np.max([curr_pl1_pl2['pl2'],pl2_min]),pl2_max])
    # clip pl1 to limits
    pl1_ub = np.min([pl1_max, curr_pl1_pl2['pl2']])
    curr_pl1_pl2['pl1'] = np.min([np.max([curr_pl1_pl2['pl1'],pl1_min]),pl1_ub])

    pls = {'pl1': int(curr_pl1_pl2['pl1']), 'pl2': int(curr_pl1_pl2['pl2']),
           'diff_action': q_action,
           'my_features': my_features}

    return pls
