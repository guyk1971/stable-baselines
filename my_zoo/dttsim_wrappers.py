# compare to: dtt_rl/train/dttsim_wrappers.py
from collections import deque
import os
import numpy as np
import gym
from gym import spaces
import pickle
from tqdm import tqdm

###########################################################################################################
# Feature Extraction (State representation)
# data that is expected to be received from the environment
# data = {'POWER': [], 'tj': [], 'tskin': [], 'ips_mean': [], 'MMIO_PL1': [], 'MMIO_PL2': []}
# also we expect to get the ewma which is calculated inside the DTTsim and is part of the original state
def feature_extraction(data, **params):
    # declare feature list
    feature_of_dict = {'power_to_curpl1_mean':None,'power_to_curpl1_std':None,'power_to_maxpl1_mean':None,
                       'power_to_maxpl1_std':None,'pl1_to_pl2':None, 'ips_mean_mean':None, 'ips_mean_std':None,
                       'tskin_slope':None,'tj_slope':None,'power_to_curpl1':None, 'power_to_maxpl1':None,
                       'pl1':None,'pl2':None,'tskin':None,'tj':None,'ips_mean':None,'torbu':None}
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
        ewma_power = params['ewma_power']
        normlized_params = params.get('normlized_params', None)

        # extract features
        feature_of_dict['power_to_curpl1'] = [data['POWER'][i] / data['MMIO_PL1'][i] for i in range(len(data['POWER']))]
        feature_of_dict['power_to_maxpl1'] = [data['POWER'][i] / pl1_max for i in range(len(data['POWER']))]
        # feature_of_dict['PKG_C0'] = [data['PKG_C0'][i] for i in range(len(data['PKG_C0']))]
        feature_of_dict['pl1'] = [(data['MMIO_PL1'][i] - pl1_min) / (pl1_max - pl1_min) for i in range(len(data['POWER']))]
        feature_of_dict['pl2'] = [(data['MMIO_PL2'][i] - pl1_min) / (pl2_max - pl2_min) for i in range(len(data['POWER']))]
        feature_of_dict['tskin'] = [(data['tskin'][i] - tskin_idle) / (tskin_max - tskin_idle) for i in
                                    range(len(data['POWER']))]
        feature_of_dict['tj'] = [(data['tj'][i] - tj_idle) / (tj_max - tj_idle) for i in range(len(data['POWER']))]


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

        feature_of_dict['ips_mean_mean'] = np.mean(data['ips_mean'])
        feature_of_dict['ips_mean_std'] = np.std(data['ips_mean'])

        # feature_of_dict['ips_std_mean'] = np.mean(data['ips_std'])
        # feature_of_dict['ips_std_std'] = np.std(data['ips_std'])

        feature_of_dict['tskin_slope'] = (feature_of_dict['tskin'][-1] - feature_of_dict['tskin'][0]) / 5
        feature_of_dict['tj_slope'] = (feature_of_dict['tj'][-1] - feature_of_dict['tj'][0]) / 5

        feature_of_dict['power_to_curpl1'] = feature_of_dict['power_to_curpl1'][-1]
        feature_of_dict['power_to_maxpl1'] = feature_of_dict['power_to_maxpl1'][-1]
        # feature_of_dict['PKG_C0'] = feature_of_dict['PKG_C0'][-1]
        feature_of_dict['pl1'] = feature_of_dict['pl1'][-1]
        feature_of_dict['pl2'] = feature_of_dict['pl2'][-1]
        feature_of_dict['tskin'] = feature_of_dict['tskin'][-1]
        feature_of_dict['tj'] = feature_of_dict['tj'][-1]

        # feature_of_dict['ips_max'] = data['ips_max'][-1]
        # feature_of_dict['ips_min'] = data['ips_min'][-1]
        feature_of_dict['ips_mean'] = data['ips_mean'][-1]
        # feature_of_dict['ips_std'] = data['ips_std'][-1]

        feature_of_dict['torbu'] = ewma_power
        ################################## 17 features ######################################
        # z-normalizing with train parameters (assuming gaussian distribution of each of the features)
        if normlized_params:
            for feature in feature_of_dict.keys():
                feature_of_dict[feature] = (feature_of_dict[feature] - normlized_params[feature]['mean']) / \
                                           normlized_params[feature]['std']

    return feature_of_dict


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


###########################################################################################################
# Reward Shaping
# assuming obs = ['pl1', 'pl2', 'power', 'tj', 'tskin','tmem' ,'ewma', 'ips_mean']
def reward_0(params,obs):
    reward = obs[0] - params.pl1_max + obs[1] - params.pl2_max - \
             10*((obs[4]>=(params.tskin_max-params.tskin_ofst)) | (obs[5]>=params.tmem_max))
    return reward

# Equivalent to reward_pl1_pl2_overshoot
def reward_2(params,obs):
    pl1_norm = (obs[0]-params.pl1_min)/(params.pl1_max-params.pl1_min)
    pl2_norm = (obs[1]-params.pl2_min)/(params.pl2_max-params.pl2_min)
    tskin_norm = (obs[4] - params.tskin_idle)/(params.tskin_max - params.tskin_idle)
    tj_norm = (obs[3] - params.tj_idle)/(params.tj_max - params.tj_idle)
    reward = -(1 - pl1_norm)**0.5 - (1 - pl2_norm)**0.5 - 2*(tskin_norm>=1) -2*(tj_norm>=1)
    return reward


def reward_3(params,obs):
    reward = (obs[0] - params.pl1_max) + (obs[1] - params.pl2_max) + (obs[7]/(10**9)) - \
             1000*((obs[4]>=(params.tskin_max-params.tskin_ofst)) | (obs[5]>=params.tmem_max))
    return reward

# similar to reward_0 only with higher penalty for overshoot
def reward_6(params,obs):
    reward = obs[0] - params.pl1_max + obs[1] - params.pl2_max - \
             100*((obs[4]>=(params.tskin_max-params.tskin_ofst)) | (obs[5]>=params.tmem_max))
    return reward

# reduce the variance by setting small rewards
def reward_7(params,obs):
    # ips=obs[7], pl1=obs[0], pl2=obs[1], tskin=obs[4], tmem=obs[5]
    reward = (obs[7]/(10**9))- int(obs[0] < params.pl1_max) - int(obs[1] < params.pl2_max) - \
             10*((obs[4]>=(params.tskin_max-params.tskin_ofst)) | (obs[5]>=params.tmem_max))
    return reward



###########################################################################################################
# DTTSim wrapper
class DTTStateRewardWrapper(gym.Wrapper):
    def __init__(self, env, feature_extractor,reward_calc,n_frames,norm_params_file=None):
        """extract features from n_frames last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        stable_baselines.common.atari_wrappers.LazyFrames

        :param env: (Gym Environment) the environment
        :param n_frames: (int) the number of frames to stack
        """
        gym.Wrapper.__init__(self, env)
        self.n_frames = n_frames
        self.feature_extractor = feature_extractor
        self.reward_calc = reward_calc
        self.n_features = len(self.feature_extractor(None).keys())
        self.frames = deque([], maxlen=n_frames)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=env.observation_space.low[0], high=env.observation_space.high[0],
                                            shape=(self.n_features,),
                                            dtype=env.observation_space.dtype)
        self.norm_params = None
        if self.norm_obs and norm_params_file:
            if os.path.exists(norm_params_file):
                self.norm_params = pickle.load(norm_params_file)
            else:
                raise ValueError('norm params file not found:',norm_params_file)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self._extract_features()

    def step(self, action):
        # assuming obs = ['pl1', 'pl2', 'power', 'tj', 'tskin','tmem' ,'ewma', 'ips_mean']
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        features = self._extract_features()
        reward = self._calc_reward(obs)
        return features, reward, done, info

    def _extract_features(self):
        # assuming f = ['pl1', 'pl2', 'power', 'tj', 'tskin','tmem' ,'ewma', 'ips_mean']
        data = {'MMIO_PL1': [f[0] for f in self.frames],
                'MMIO_PL2': [f[1] for f in self.frames],
                'POWER': [f[2] for f in self.frames],
                'tj': [f[3] for f in self.frames],
                'tskin': [f[4] for f in self.frames],
                'tmem': [f[5] for f in self.frames],
                'ips_mean': [f[7] for f in self.frames]}
        ewma_power = self.frames[-1][6]   # take the ewma from the environment
        # platform params:
        # 'tdp', 'pl1_min', 'pl1_max', 'pl2_min', 'pl2_max',
        # 'tj_idle', 'tj_max', 'tskin_idle', 'tskin_max',
        # 'tskin_ofst', 'tau', 'p2tj', 'tj2ts', 'p2ips']
        # required arguments :  pl1_max, pl1_min, tskin_max, tskin_idle,pl2_max, pl2_min, tj_max, tj_idle,
        #                       tmem_max, tmem_idle
        features_dict = self.feature_extractor(data,ewma_power=ewma_power,normlized_params=self.norm_params,
                                           **self.platform.params._asdict())
        return np.fromiter(features_dict.values(),dtype=float)

    def _calc_reward(self,state):
        return self.reward_calc(self.platform.params,state)




def main():
    platform = PLATFORMS['Scarlet']
    # to create the following experiment: (benchmark num_runs sec_between_runs)
    # time between iterations : 300 sec
    # - cb15 10 120  --> 10*(['cb15']+[('cooldown',120)])
    # - cb15 10 60   --> 10*(['cb15']+[('cooldown',60)])
    # - cb15 10 30   --> 10*(['cb15']+[('cooldown',30)])
    # do the following:
    n_episodes=100
    episode_workloads = EPISODES['cb20mr']
    env = DTTEnvSim(platform, episode_workloads=episode_workloads, norm_obs=False)
    env = DTTStateRewardWrapper(env=env,feature_extractor=feature_extraction_scarlet,reward_calc=reward_0,n_frames=5)
    dPL2act = {v: np.int64(k) for k, v in env.dPL.items()}
    feat_cols = list(feature_extraction_scarlet(None))
    tsdf = pd.DataFrame(columns=feat_cols)
    os.makedirs('./tmp',exist_ok=True)
    feat_csv = os.path.join(os.getcwd(),'tmp','sim_features.csv')
    tsdf.to_csv(feat_csv,index=False)
    for ei in tqdm(range(n_episodes)):
        obs = env.reset()
        done = False
        out_df = pd.DataFrame(columns=feat_cols)
        ts = 0
        total_rew = 0
        while not done:
            act = random_policy(platform.params, obs, dPL2act)
            obs, rew, done, info = env.step(act)
            total_rew += rew
            # obs = ['pl1','pl2','power','tj','tskin','tmem','ewma']
            # policy : as long as we're below the max value, aim to increase
            tsdf.loc[0] = list(obs)
            out_df = out_df.append(tsdf)
            ts += 1
        env.close()
        # print('session completed. total reward: {}'.format(total_rew))
        out_df.to_csv(feat_csv,mode='a',header=False,index=False)

    # calculate normalization params
    all_df = pd.read_csv(feat_csv)
    normlized_params_features = {}
    for col in all_df.columns:
        normlized_params_features[col] =  {'name': col, 'mean': all_df[col].mean(), 'std': all_df[col].std()}

    norm_filename=os.path.join(os.getcwd(),'tmp','sim_features_normlized_params.pkl')
    file_normlized = open(norm_filename, 'wb')
    pickle.dump(normlized_params_features, file_normlized)
    file_normlized.close()





if __name__ == '__main__':
    import sys
    path_to_curr_file = os.path.realpath(__file__)
    proj_root = os.path.dirname(os.path.dirname(path_to_curr_file))
    if proj_root not in sys.path:
        sys.path.insert(0, proj_root)
    from my_zoo.my_envs import PLATFORMS,DTTEnvSim,EPISODES,random_policy
    import pandas as pd
    main()

