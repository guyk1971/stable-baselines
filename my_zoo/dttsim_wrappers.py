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

###########################################################################################################
# Reward Shaping
def reward_pl1_pl2_overshoot(params):
    #pl1_limit = 1 if params['pl1'] > params['pl1_to_pl2'] else params['pl1_to_pl2']
    reward = -(1 - params['pl1'])**0.5 - (1 - params['pl2'])**0.5 -1000*(params['tskin']>=1) - 1000*(params['tj']>=1)
    #reward = params['ips_mean']
    return reward

def reward_ips(params):
    reward = params['ips_mean']/10**10 - 10000*(params['tskin']>=1) - 10000*params['ips_mean']*(params['tj']>1)
    return reward

def orig_reward(params):
    reward = (params['pl1']- params['pl1_max']) + (params['pl2'] - params['pl2_max']) - 1
    if params['tskin'] >= 1:
        reward -= 1000
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
        # assuming obs = ['pl1', 'pl2', 'power', 'tj', 'tskin', 'ewma', 'ips_mean']
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        features = self._extract_features()
        reward = self._calc_reward(obs)
        return features, reward, done, info

    def _extract_features(self):
        data = {'MMIO_PL1': [f[0] for f in self.frames],
                'MMIO_PL2': [f[1] for f in self.frames],
                'POWER': [f[2] for f in self.frames],
                'tj': [f[3] for f in self.frames],
                'tskin': [f[4] for f in self.frames],
                'ips_mean': [f[6] for f in self.frames]}
        ewma_power = self.frames[-1][5]   # take the ewma from the environment
        # platform params:
        # 'tdp', 'pl1_min', 'pl1_max', 'pl2_min', 'pl2_max',
        # 'tj_idle', 'tj_max', 'tskin_idle', 'tskin_max',
        # 'tskin_ofst', 'tau', 'p2tj', 'tj2ts', 'p2ips']
        # required arguments :  pl1_max, pl1_min, tskin_max, tskin_idle,pl2_max, pl2_min, tj_max, tj_idle,
        features_dict = self.feature_extractor(data,ewma_power=ewma_power,normlized_params=self.norm_params,
                                           **self.platform.params._asdict())
        return np.fromiter(features_dict.values(),dtype=float)

    def _calc_reward(self,state):
        params={'pl1':state[0],
                'pl2':state[1],
                'pl1_max':self.platform.params.pl1_max,
                'pl2_max':self.platform.params.pl2_max,
                'tskin':state[4]/(self.platform.params.tskin_max-self.platform.params.tskin_ofst),
                'tj':state[3],
                'ips_mean':state[6]}
        return self.reward_calc(params)




def main():
    platform = PLATFORMS['Scarlet']

    # workload_params = 20*([BENCHMARKS['cb15']]+[BENCHMARKS['cooldown']]*30+\
    #                   [BENCHMARKS['bursty']]+[BENCHMARKS['cooldown']]*10+ \
    #                   [BENCHMARKS['bursty']] + [BENCHMARKS['cooldown']] * 5 + \
    #                   [BENCHMARKS['bursty']] + [BENCHMARKS['cooldown']] * 5)

    # to create the following experiment: (benchmark num_runs sec_between_runs)
    # time between iterations : 300 sec
    # - cb15 10 120  --> 10*(['cb15']+['cooldown']*60)
    # - cb15 10 60   --> 10*(['cb15']+['cooldown']*30)
    # - cb15 10 30   --> 10*(['cb15']+['cooldown']*15)
    # do the following:
    # workload_params = 10*(['cb15']+['cooldown']*60) +\
    #                   ['cooldown'] * 150 + \
    #                   10*(['cb15']+['cooldown']*30) + \
    #                   ['cooldown'] * 150 + \
    #                   10*(['cb15']+['cooldown']*15)
    n_episodes=100
    # workload_params = 10*(['cb15']+['cooldown']*60) +\
    #                   ['cooldown'] * 150 + \
    #                   10*(['cb20']+['cooldown']*60) + \
    #                   ['cooldown'] * 150 + \
    #                   10*(['cb15']+['cooldown']*45) + \
    #                   ['cooldown'] * 150 + \
    #                   10 * (['cb20'] + ['cooldown'] * 45) + \
    #                   ['cooldown'] * 150 + \
    #                   10 * (['cb15'] + ['cooldown'] * 30) + \
    #                   ['cooldown'] * 150 + \
    #                   10 * (['cb20'] + ['cooldown'] * 30) + \
    #                   ['cooldown'] * 150 + \
    #                   10 * (['cb15'] + ['cooldown'] * 15) + \
    #                   ['cooldown'] * 150 + \
    #                   10 * (['cb20'] + ['cooldown'] * 15)

    workload_params = 10*(['cb15']+['cooldown']*2)
    env = DTTEnvSim(platform, workload_params=workload_params, norm_obs=False)
    env = DTTStateRewardWrapper(env=env,feature_extractor=feature_extraction,reward_calc=orig_reward,n_frames=5)
    dPL2act = {v: np.int64(k) for k, v in env.dPL.items()}
    feat_cols = list(feature_extraction(None))
    tsdf = pd.DataFrame(columns=feat_cols)
    tsdf.to_csv('sim_features.csv',index=False)
    for ei in tqdm(range(n_episodes)):
        obs = env.reset()
        done = False
        out_df = pd.DataFrame(columns=feat_cols)
        ts = 0
        a1 = 0
        a2 = 0  # start with no change
        total_rew = 0
        while not done:
            # act = dPL2act[(a1, a2)]
            act = np.random.randint(0,9)    # for random policy
            obs, rew, done, info = env.step(act)
            total_rew += rew
            # obs = ['pl1','pl2','power','tj','tskin','ewma']
            # policy : as long as we're below the max value, aim to increase
            a1 = 0.5 if obs[0] < env.platform.params.pl1_max else 0
            a2 = 0.5 if obs[1] < env.platform.params.pl2_max else 0
            tsdf.loc[0] = list(obs)
            out_df = out_df.append(tsdf)
            ts += 1
        env.close()
        print('session completed. total reward: {}'.format(total_rew))
        out_df.to_csv('sim_features.csv',mode='a',header=False,index=False)

    # calculate normalization params
    all_df = pd.read_csv('sim_features.csv')
    normlized_params_features = {}
    for col in all_df.columns:
        normlized_params_features[col] =  {'name': col, 'mean': all_df[col].mean(), 'std': all_df[col].std()}

    file_normlized = open('sim_features_normlized_params.pkl', 'wb')
    pickle.dump(normlized_params_features, file_normlized)
    file_normlized.close()





if __name__ == '__main__':
    import sys
    path_to_curr_file = os.path.realpath(__file__)
    proj_root = os.path.dirname(os.path.dirname(path_to_curr_file))
    if proj_root not in sys.path:
        sys.path.insert(0, proj_root)
    from my_zoo.my_envs import PLATFORMS,DTTEnvSim
    import pandas as pd
    main()

