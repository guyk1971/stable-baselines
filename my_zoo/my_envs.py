#####################################################################
# my_envs.py
# Implementations of custom envs
# MLA Template Equivalent: train.custom_envs
import gym
from gym import spaces
import numpy as np
import os
import pandas as pd
from abc import abstractmethod
from dataclasses import dataclass
######################################################
#region L2P
class L2PEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple env that imitates the L2P behaviour
    """
    # Because of google colab, we cannot implement the GUI ('human' render mode)
    metadata = {'render.modes': ['console']}

    def __init__(self, obs_dim=7,n_act=4):
        super(L2PEnv, self).__init__()

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
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,shape=(obs_dim,), dtype=np.float32)
        self.max_path_length = 40

    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        self.step_idx=0
        return self.observation_space.sample()


    def step(self, action):

        if (not isinstance(action,int)) or (action<0) or (action>=self.action_space.n):
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
#endregion
######################################################
#region DTTEnvReal
# this class defines the data that is extracted from the real environment. to be used when reading the csv
# experience buffer extracted from the real ESIF file

class DTTEnvReal(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple env that imitates the L2P behaviour
    """
    # Because of google colab, we cannot implement the GUI ('human' render mode)
    metadata = {'render.modes': ['console']}

    def __init__(self, obs_dim=31,n_act=9):
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
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,shape=(obs_dim,), dtype=np.float32)
        self.max_path_length = 1000     # arbitrary number

    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        self.step_idx=0
        return self.observation_space.sample()


    def step(self, action):
        '''
        Currently a dummy function that should not be called.
        :param action:
        :return:
        '''
        if (not isinstance(action,int)) or (action<0) or (action>=self.action_space.n):
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

#endregion

#############################################################################################
#region DTTEnvSim
# definition of DTT environment that is simulating the real environment.
# to be used for online training - to debug the agent's behavior
from collections import namedtuple
from enum import Enum
from copy import copy
from scipy.interpolate import interp1d
IDLE_POWER = 1.0

#########################################################################################
# Workload definitions
# the benchmarks are defined as list of tuples : (Power,Seconds)
BENCHMARKS = {'cb15': [(0, IDLE_POWER), (5, 45), (10, 45), (20, 30), (40, 30), (50, 28), (60, IDLE_POWER)],
              'cb20': [(0, IDLE_POWER), (10, 44), (13, 44), (20, 32), (35, 30), (45, 30), (46, 25), (200, 25),
                       (205, IDLE_POWER)],
              'cooldown': [IDLE_POWER],  # cooldown is parsed differently.see below
              'bursty':[(0,IDLE_POWER),(2,55),(4,20),(5,IDLE_POWER)]}

EPISODES = {'cb_long':  10 * (['cb15'] + [('cooldown',300)]) + [('cooldown',300)] + \
                        10 * (['cb20']+[('cooldown',300)]) + [('cooldown',300)] + \
                        10 * (['cb15'] + [('cooldown',180)]) + [('cooldown',300)] + \
                        10 * (['cb20']+[('cooldown',180)]) + [('cooldown',300)] + \
                        10 * (['cb15'] + [('cooldown',120)]) + [('cooldown',300)] + \
                        10 * (['cb20']+[('cooldown',120)]) + [('cooldown',300)] + \
                        10 * (['cb15'] + [('cooldown',60)]) + [('cooldown',300)] + \
                        10 * (['cb20'] + [('cooldown',60)]) + [('cooldown',300)] + \
                        10 * (['cb15'] + [('cooldown',30)]) + [('cooldown',300)] + \
                        10 * (['cb20'] + [('cooldown',30)]) + [('cooldown',300)] + \
                        10 * (['cb15'] + [('cooldown',1)]) + [('cooldown',300)] + \
                        10 * (['cb20'] + [('cooldown',1)]) + [('cooldown',300)],
            'cb15_1':   10 * (['cb15']+[('cooldown',1)]),
            'cb15_300':   10 * (['cb15']+[('cooldown',300)]),
            'cb20_300':   10 * (['cb20']+[('cooldown',300)]),
            'cb15mr':   5 * (['cb15']+[('cooldown',1)]),
            'cb20mr':   5 * (['cb20']+[('cooldown',1)]),
            'cb20mrx3': 2*(5 * (['cb20']+[('cooldown',1)])+[('cooldown',300)]) + 5 * (['cb20']+[('cooldown',1)]),
            'cb15': ['cb15']+[('cooldown',1)],
            'cb20': ['cb20']+[('cooldown',1)],
            'cb20mix1': 10 * (['cb20']+[('cooldown',300)]) + [('cooldown',300)] + 5 * (['cb20']+[('cooldown',1)]),
            'cb15mix1': 10 * (['cb15']+[('cooldown',300)]) + [('cooldown',300)] + 5 * (['cb15']+[('cooldown',1)]),
            'cb20mix2': 10 * (['cb20']+[('cooldown',300)]) + [('cooldown',300)] + \
                          10 * (['cb20']+[('cooldown',1)]) + [('cooldown',300)] + \
                          10 * (['cb20']+[('cooldown',30)]),
            'cb20mix3': 8 * (5 * (['cb20']+[('cooldown',1)])+[('cooldown',300)]) + \
                        10 * (['cb20']+[('cooldown',300)])+ \
                        8 * (5 * (['cb20']+[('cooldown',1)])+[('cooldown',300)]) + \
                        10 * (['cb20'] + [('cooldown', 180)]) + \
                        8 * (5 * (['cb20'] + [('cooldown', 1)]) + [('cooldown', 300)]) + \
                        10 * (['cb20'] + [('cooldown', 120)]) + \
                        8 * (5 * (['cb20'] + [('cooldown', 1)]) + [('cooldown', 300)]) + \
                        10 * (['cb20'] + [('cooldown', 90)]) + \
                        8 * (5 * (['cb20'] + [('cooldown', 1)]) + [('cooldown', 300)]) + \
                        10 * (['cb20'] + [('cooldown', 60)]) + \
                        8 * (5 * (['cb20'] + [('cooldown', 1)]) + [('cooldown', 300)]) + \
                        10 * (['cb20'] + [('cooldown', 30)]) + \
                        8 * (5 * (['cb20'] + [('cooldown', 1)]) + [('cooldown', 300)])
            }


WLScoreModel =  namedtuple("WLScoreModel", ['intercept', 'ips_coef'])
WorkloadScoreModels = {'cb15': WLScoreModel(-47.65,2.367e-07),
                       'cb20': WLScoreModel(-101.48,5.036e-07),
                       'cooldown': WLScoreModel(0,0), 'bursty': WLScoreModel(0,0)}
##########################################################################################
# Platforms definition

#################################################
# Base definitions
# a simplified model to predict Tskin from Tj is:
# tskin[n]=tskin[n-1]+Tj2TskinFactor*(tj[n]-tj[n-1])
# tskin[n]= intercept + tj_coef * tj[n] + tjm1_coef * tj[n-1] + tsm1_coef * tskin[n-1]
Tj2Tskin = namedtuple("Tj2Tskin", ['intercept', 'tj_coef', 'tjm1_coef', 'tsm1_coef'])

# tj[n]= intercept + p_coef * power[n] + tj_coef* tj[n-1]
P2Tj = namedtuple("P2Tj", ['intercept', 'p_coef', 'tj_coef'])

# ips[n] = intercept + p_coef * power[n] + pnm1_coef * power[n-1] + ipsnm1_coef * ips[n-1]
P2ips = namedtuple("P2ips",['intercept','p_coef','pnm1_coef','ipsnm1_coef'])

# tmem[n]= intercept + tj_coef * tj[n] + tjm1_coef * tj[n-1] + tmm1_coef * tmem[n-1]
Tj2Tmem = namedtuple("Tj2Tmem", ['intercept', 'tj_coef', 'tjm1_coef', 'tmm1_coef'])




class Platform:
    def __init__(self, platform_params):
        self.params = platform_params
        self.state = None

    @staticmethod
    def predict_tj(power: float, tj_nm1, tj_max, coefs: P2Tj):
        return np.minimum(coefs.intercept + coefs.p_coef * power + coefs.tj_coef * tj_nm1, tj_max)

    @staticmethod
    def predict_tskin(curr_tj, prev_tj, prev_tskin, tskin_max, coefs: Tj2Tskin):
        return np.minimum(
            coefs.intercept + coefs.tj_coef * curr_tj + coefs.tjm1_coef * prev_tj + coefs.tsm1_coef * prev_tskin,
            tskin_max)

    @staticmethod
    def predict_mean_ips(curr_power,prev_power,prev_ips,coefs:P2ips):
        return (coefs.intercept + coefs.p_coef*curr_power + coefs.pnm1_coef*prev_power + coefs.ipsnm1_coef * prev_ips)


    @abstractmethod
    def _run_dtt(self, req_pl1, req_pl2):
        raise NotImplementedError

    @abstractmethod
    def _run_pcode(self, act_pl1, act_pl2):
        raise NotImplementedError

    @abstractmethod
    def reset_state(self):
        raise NotImplementedError

    def set_state(self, state):
        self.state = state

    @abstractmethod
    def _norm_state(self):
        raise NotImplementedError

    def get_state(self, norm=False):
        if norm:
            return self._norm_state()
        return self.state

    def get_power_budget(self, req_pl1, req_pl2):
        act_pl1, act_pl2, clip_reason = self._run_dtt(req_pl1, req_pl2)
        power_budget, clip_reason_pcode = self._run_pcode(act_pl1, act_pl2)
        clip_reason += clip_reason_pcode
        if len(clip_reason) == 0:
            clip_reason += (IAClipReason.No_Clip,)
        self.state.pl1 = act_pl1
        self.state.pl2 = act_pl2
        return power_budget, clip_reason

    @abstractmethod
    def consume_power(self, power_consumed,wl_name):
        raise NotImplementedError

class IAClipReason(Enum):
    No_Clip = 0
    Thermal_Event = 1
    Max_Turbo_Limit = 2

#################################################
# Scarlet specifics
# trained on : ['rl_rnd_64','rl_rnd_64_3','rl_rnd_64_4','rl_rnd_64_5','rl_rnd_64_6']
# filters cb15: [('cinebench',300),('cinebench',240),('cinebench',180),('cinebench',120),('cinebench',90),('cinebench',60),('cinebench',30)]
# filter cb20: [('cb20',300),('cb20',240),('cb20',180),('cb20',120),('cb20',90),('cb20',60),('cb20',30)]
# tskin[n]= intercept + tj_coef * tj[n] + tjm1_coef * tj[n-1] + tsm1_coef * tskin[n-1]
# Tj2Tskin = namedtuple("Tj2Tskin", ['intercept', 'tj_coef', 'tjm1_coef', 'tsm1_coef'])
# Tj2TskinFactor = {'cb15X30': Tj2Tskin(0.631, 0.003, 0.007, 0.973),
#                  'cb15': Tj2Tskin(0.8, 0.0027, 0.012, 0.963),
#                  'cooldown': Tj2Tskin(0.8, 0.0027, 0.012, 0.963),   # arbitrarily = cb15
#                  'cb20': Tj2Tskin(0.675, 0.0039, 0.0085, 0.969),
#                  'cb20X30': Tj2Tskin(0.785, 0.003, 0.01, 0.968)}
Tj2TskinFactor = {'cb15': Tj2Tskin(0.6, 0.0056, 0.0085, 0.97),
                 'cooldown': Tj2Tskin(0.8, 0.0027, 0.012, 0.963),   # arbitrarily = cb15
                 'cb20': Tj2Tskin(0.596, 0.0024, 0.01, 0.97)}

# train_data=[{'folders':['psvt_at-9_25_45_64-fixed_1','psvt_at-9_25_45_64-greedy_1',],
#              'traces':[('cinebench',300),('cinebench',180),('cinebench',120),('cinebench',60),('cinebench',30),('cinebench',1),
#                        ('cb20',300),('cb20',180),('cb20',120),('cb20',60),('cb20',30),('cb20',1)]}]
# tmem[n]= intercept + tj_coef * tj[n] + tjm1_coef * tj[n-1] + tmm1_coef * tmem[n-1]
# Tj2Tmem = namedtuple("Tj2Tmem", ['intercept', 'tj_coef', 'tjm1_coef', 'tmm1_coef'])

Tj2TmemFactor = {'cb15': Tj2Tmem(0.1087, 0.0021, 0.01, 0.9836),
                 'cooldown': Tj2Tmem(0.1087, 0.0021, 0.01, 0.9836),
                 'cb20': Tj2Tmem(0.1087, 0.0021, 0.01, 0.9836)}


# tj[n]= intercept + p_coef * power[n] + tj_coef * tj[n-1]
# P2Tj = namedtuple("P2Tj", ['intercept', 'p_coef', 'tj_coef'])
# Power2TjFactor = {'cb15X30': P2Tj(47.74, 0.791, 0.277),
#                   'cb15': P2Tj(11.17, 0.419, 0.75),
#                   'cooldown': P2Tj(11.17, 0.419, 0.75),     # arbitrarily = cb15
#                   'cb20': P2Tj(14.63, 0.581, 0.68),
#                   'cb20X30': P2Tj(45.38, 0.797, 0.303)}
Power2TjFactor = {'cb15': P2Tj(9.62, 0.294, 0.81),
                  'cooldown': P2Tj(9.62, 0.294, 0.81),     # arbitrarily = cb15
                  'cb20': P2Tj(9.79, 0.33, 0.81)}

# ips[n] = intercept + p_coef * power[n] + pnm1_coef * power[n-1] + ipsnm1_coef * ips[n-1]
Power2IPSmean = {'cb15':P2ips(-12500268.871777534, 8.04534221e+07, -7.10065226e+07, 9.20609571e-01),
                 'cooldown':P2ips(-7830878.88678503, 8.42984945e+07, -7.61181444e+07,9.38108383e-01),
                 'cb20':P2ips(-6068173.380416393, 9.06084012e+07, -7.84225589e+07,  9.12361485e-01)}

PlatformParamsScarlet = namedtuple("PlatformParamsScarlet", ['tdp', 'pl1_min','pl1_max', 'pl2_min', 'pl2_max',
                                                             'tj_idle', 'tj_max', 'tskin_idle','tskin_max',
                                                             'tmem_idle','tmem_max',
                                                             'ips_idle','ips_max',
                                                             'tskin_ofst', 'tau', 'p2tj', 'tj2ts','p2ips','tj2tm'])

SCARLET_TDP = 15.0
SCARLET_PL1MAX = 25.0
SCARLET_PL1MIN = 9.0
SCARLET_PL2MAX = 64.0
SCARLET_PL2MIN = 45.0
SCARLET_TJIDLE = 45.0
SCARLET_TJMAX = 100.0
SCARLET_TSKINIDLE=35.0
SCARLET_TSKINMAX = 65.0
SCARLET_TMEMIDLE=38.0
SCARLET_TMEMMAX = 70.0
SCARLET_TSKINOFST = 0.0
SCARLET_TAU = 28.0
SCARLET_INITIAL_PL1 = 25.0      # SCARLET_PL1MAX
SCARLET_INITIAL_PL2 = 64.0      # SCARLET_PL2MAX
SCARLET_IPS_MAX = 1e10
SCARLET_IPS_IDLE = 47e6
SCARLET_TURBO_HYSTERESIS = 0.5

@dataclass
class StateScarlet:
    # add sensors specific to scarlet (that are not common to all platforms)
    pl1 : float = SCARLET_INITIAL_PL1
    pl2 : float = SCARLET_INITIAL_PL2
    power : float = IDLE_POWER
    tj : float = SCARLET_TJIDLE
    tskin : float = SCARLET_TSKINIDLE
    tmem : float = SCARLET_TMEMIDLE
    ewma : float = SCARLET_INITIAL_PL1-IDLE_POWER
    ips_mean : float = SCARLET_IPS_IDLE



class Scarlet(Platform):
    def __init__(self, platform_params):
        super(Scarlet, self).__init__(platform_params)
        self.min_turbo_budget = 0
        self.reset_state()

    @staticmethod
    def predict_tmem(curr_tj, prev_tj, prev_tmem, tmem_max, coefs: Tj2Tmem):
        return np.minimum(
            coefs.intercept + coefs.tj_coef * curr_tj + coefs.tjm1_coef * prev_tj + coefs.tmm1_coef * prev_tmem,
            tmem_max)


    def _run_dtt(self, req_pl1, req_pl2):
        '''
        tries to apply the power levels requested by the policy.
        conducts a series of checks to what are the allowed power levels
        :param req_pl1:
        :param req_pl2:
        :return: actual power levels that are allowed
        '''
        act_pl1 = req_pl1
        act_pl2 = req_pl2
        clip_reason = ()
        # check pl2 - nothing to check. simply accept it

        # check pl1
        # here we follow the passive table to see if we need to relax pl1,pl2
        if self.state.tskin >= (self.params.tskin_max - self.params.tskin_ofst):
            # thermal event. reduce act_pl1 to PL1_min
            act_pl1 = self.params.pl1_min
            clip_reason += (IAClipReason.Thermal_Event,)

        if self.state.tmem >= self.params.tmem_max:
            # thermal event. reduce act_pl1 to PL1_min
            act_pl1 = self.params.pl1_min
            clip_reason += (IAClipReason.Thermal_Event,)


        return act_pl1, act_pl2, clip_reason

    def _run_pcode(self, act_pl1, act_pl2):
        # the following depends on the way we calculate ewma
        clip_reason = ()
        power_budget = act_pl2
        # if self.state.ewma >= act_pl1:   # old calculation
        if self.state.ewma <= self.min_turbo_budget:    # new calculation
            clip_reason += (IAClipReason.Max_Turbo_Limit,)
            power_budget = act_pl1
            self.min_turbo_budget = SCARLET_TURBO_HYSTERESIS
        else:
            self.min_turbo_budget = 0
        return power_budget, clip_reason

    def reset_state(self):
        '''
        reset_state simulate a system that has been IDLE for a long time
        '''
        # note that ewma depends on the way it is calculated. currently set to IDLE_POWER.
        # with the new formula it should be SCARLET_INITIAL_PL1-IDLE_POWER
        self.state = StateScarlet(pl1=SCARLET_INITIAL_PL1, pl2=SCARLET_INITIAL_PL2, power=IDLE_POWER, tj=SCARLET_TJIDLE,
                                  tskin=SCARLET_TSKINIDLE, tmem=SCARLET_TMEMIDLE,ewma=(SCARLET_INITIAL_PL1-IDLE_POWER),
                                  ips_mean=SCARLET_IPS_IDLE)

        # inject some noise
        self.state.tj = np.abs(self.state.tj + np.random.randn())
        self.state.tskin = np.abs(self.state.tskin + np.random.randn())
        self.state.tskin = np.abs(self.state.tskin + np.random.randn())
        self.state.power = np.abs(self.state.power + np.random.randn())
        return self.get_state()

    def _norm_state(self):
        pl1 = (self.state.pl1 - self.params.pl1_min) / (self.params.pl1_max - self.params.pl1_min)
        pl2 = (self.state.pl2 - self.params.pl2_min) / (self.params.pl2_max - self.params.pl2_min)
        power = (self.state.power - self.params.pl1_min) / (self.params.pl2_max - self.params.pl2_min)
        tj = (self.state.tj - self.params.tj_idle) / (self.params.tj_max-self.params.tj_idle)
        tskin = (self.state.tskin - self.params.tskin_idle) / (self.params.tskin_max - self.params.tskin_idle)
        tmem = (self.state.tmem - self.params.tmem_idle) / (self.params.tmem_max - self.params.tmem_idle)
        ewma = self.state.ewma / self.state.pl1
        ips_mean = self.state.ips_mean / self.params.ips_max
        return StateScarlet(pl1, pl2, power, tj, tskin, tmem, ewma,ips_mean)

    def consume_power(self, power_consumed,wl_name):
        # update ewma
        prev_power = self.state.power
        self.state.power = power_consumed
        # old ewma
        # self.state.ewma = self.state.ewma + (1.0 / self.params.tau) * (self.state.power - self.state.ewma)
        # new ewma
        self.state.ewma = np.exp(-1.0/self.params.tau) * self.state.ewma + \
                          (1-np.exp(-1.0 / self.params.tau)) * (self.state.pl1 - self.state.power)

        # update thermal sensors
        prev_tj = self.state.tj
        # note : currently using the same model for all types of workloads.
        # for better fit, use the corresponding workload parameters
        self.state.tj = self.predict_tj(self.state.power, prev_tj, self.params.tj_max, self.params.p2tj[wl_name])
        prev_tskin = self.state.tskin
        self.state.tskin = self.predict_tskin(self.state.tj, prev_tj, prev_tskin, self.params.tskin_max,
                                       self.params.tj2ts[wl_name])

        self.state.ips_mean = self.predict_mean_ips(self.state.power,prev_power,self.state.ips_mean,self.params.p2ips[wl_name])

        prev_tmem = self.state.tmem
        self.state.tmem = self.predict_tmem(self.state.tj,prev_tj,prev_tmem,self.params.tmem_max,self.params.tj2tm[wl_name])



#################################################
# Billie specifics - to fill in like in scarlet
class Billie(Platform):
    def __init__(self, platform_params):
        super(Billie, self).__init__(platform_params)

    def _run_dtt(self, req_pl1, req_pl2):
        pass

    def _run_pcode(self, act_pl1, act_pl2):
        pass

################################################
# Platforms dictionary
PLATFORMS = {'Scarlet': Scarlet(PlatformParamsScarlet(tdp=SCARLET_TDP, pl1_min=SCARLET_PL1MIN, pl1_max=SCARLET_PL1MAX,
                                                      pl2_min=SCARLET_PL2MIN, pl2_max=SCARLET_PL2MAX,
                                                      tj_idle=SCARLET_TJIDLE, tj_max=SCARLET_TJMAX,
                                                      tskin_idle=SCARLET_TSKINIDLE ,tskin_max=SCARLET_TSKINMAX,
                                                      tmem_idle=SCARLET_TMEMIDLE, tmem_max=SCARLET_TMEMMAX,
                                                      ips_idle=SCARLET_IPS_IDLE,ips_max=SCARLET_IPS_MAX,
                                                      tskin_ofst=SCARLET_TSKINOFST, tau=SCARLET_TAU,
                                                      p2tj=Power2TjFactor, tj2ts=Tj2TskinFactor,
                                                      p2ips=Power2IPSmean,tj2tm=Tj2TmemFactor)),
             # ScarletX : enhanced platform where pl1_max = PL2MAX
             'ScarletX': Scarlet(PlatformParamsScarlet(tdp=SCARLET_TDP, pl1_min=SCARLET_PL1MIN, pl1_max=SCARLET_PL2MAX,
                                                      pl2_min=SCARLET_PL2MIN, pl2_max=SCARLET_PL2MAX,
                                                      tj_idle=SCARLET_TJIDLE, tj_max=SCARLET_TJMAX,
                                                      tskin_idle=SCARLET_TSKINIDLE, tskin_max=SCARLET_TSKINMAX+10,
                                                      tmem_idle=SCARLET_TMEMIDLE, tmem_max=SCARLET_TMEMMAX,
                                                      ips_idle=SCARLET_IPS_IDLE, ips_max=SCARLET_IPS_MAX,
                                                      tskin_ofst=SCARLET_TSKINOFST, tau=SCARLET_TAU,
                                                      p2tj=Power2TjFactor, tj2ts=Tj2TskinFactor,
                                                      p2ips=Power2IPSmean, tj2tm=Tj2TmemFactor)),
             # ScarletM - relax tskin constraint. rely only on TMEM
             'ScarletM': Scarlet(PlatformParamsScarlet(tdp=SCARLET_TDP, pl1_min=SCARLET_PL1MIN, pl1_max=SCARLET_PL1MAX,
                                                      pl2_min=SCARLET_PL2MIN, pl2_max=SCARLET_PL2MAX,
                                                      tj_idle=SCARLET_TJIDLE, tj_max=SCARLET_TJMAX,
                                                      tskin_idle=SCARLET_TSKINIDLE, tskin_max=SCARLET_TSKINMAX+10,
                                                      tmem_idle=SCARLET_TMEMIDLE, tmem_max=SCARLET_TMEMMAX,
                                                      ips_idle=SCARLET_IPS_IDLE, ips_max=SCARLET_IPS_MAX,
                                                      tskin_ofst=SCARLET_TSKINOFST, tau=SCARLET_TAU,
                                                      p2tj=Power2TjFactor, tj2ts=Tj2TskinFactor,
                                                      p2ips=Power2IPSmean, tj2tm=Tj2TmemFactor)),
             }




#############################################################################################
# DTTEnvSim definition
class DTTEnvSim(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple env that imitates the L2P behaviour
    """
    # Because of google colab, we cannot implement the GUI ('human' render mode)
    metadata = {'render.modes': ['console']}
    dPL = {0: (-0.5, -0.5), 1: (0, -0.5), 2: (0.5, -0.5),
           3: (-0.5, 0), 4: (0, 0), 5: (0.5, 0),
           6: (-0.5, 0.5), 7: (0, 0.5), 8: (0.5, 0.5)}

    def __init__(self, platform, episode_workloads, norm_obs=False, full_reset=True, fixed_pl=None,
                 calc_reward_fn=None,log_output=None):
        super(DTTEnvSim, self).__init__()
        self.platform = platform
        self.workload_params = episode_workloads
        self.state = self.platform.reset_state()  # assuming not normalized state as default
        # the observation space include obs_dim float values
        self.obs_dim = len(self.state.__dict__.keys())

        # Currently assuming discrete action space with n_act actions
        self.act_dim = 1
        self.norm_obs = norm_obs
        self.log_output = log_output
        if self.log_output is not None:
            self.log_output = os.path.join(self.log_output, 'DTTSim_esif.csv')
            self.esif_cols = ['timestamp'] + list(self.state.__dict__.keys()) + ['Clip','Episode_Scores']
            self.out_df = pd.DataFrame(columns=self.esif_cols)
            self.tsdf = pd.DataFrame(columns=self.esif_cols)  # include a single timestep

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions, we have two: left and right
        n_act = len(self.dPL)
        self.action_space = spaces.Discrete(n_act)
        # The observation will be the coordinate of the agent
        # this can be described both by Discrete and Box space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.max_path_length = np.inf
        self.full_reset = full_reset    # whether to reset the platform state or just the workload stack
        self.mean_ips_during_workload = []      # collect ips to infer the score
        self.fixed_pl = fixed_pl
        self.calc_reward_fn = calc_reward_fn

    def get_state(self):
        state = self.platform.get_state(norm=self.norm_obs).__dict__.values()
        return np.array([s for s in state])

    def _predict_score(self):
        coefs = WorkloadScoreModels[self.wip_name]
        mean_ips = np.mean(self.mean_ips_during_workload)
        score = round(coefs.intercept + coefs.ips_coef * mean_ips,2)
        # print(f'predicted score for {self.wip_name} : {score}')
        return score

    def _pop_workload(self):
        # calc the score of the recently completed workload
        if self.wip_name and self.wip_name != 'cooldown':
            self.last_score = self._predict_score()
            self.workload_scores.append(self.last_score)
        next_wl = self.workload_stack.pop(0)
        self.mean_ips_during_workload = []
        self.wip_name, self.wip_queue = next_wl.popitem()

    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        self.step_idx = 0
        # reset the workload
        self.workload_stack = []
        self.workload_scores = []
        for wl_name in self.workload_params:
            if 'cooldown' in wl_name:
                wlq = BENCHMARKS['cooldown']*wl_name[1]
                wln = 'cooldown'
            else:
                wl=BENCHMARKS[wl_name]
                x = [c[0] for c in wl]
                y = [c[1] for c in wl]
                f = interp1d(x, y)
                wlq = list(f(range(x[-1] + 1)))
                wln = wl_name
            self.workload_stack.append({wln:wlq})
        self.wip_queue = []
        self.wip_name = None
        self.last_score = np.nan
        self.req_power_for_curr_step = 0
        # reset the platform if full reset
        if self.full_reset:
            self.state = self.platform.reset_state()  # get unnormalized state from the platform
        return self.get_state()  # get normalized state if needed (self.norm_obs=True)

    def get_scores(self):
        return self.workload_scores

    def _calc_reward(self):
        if self.calc_reward_fn:
            reward = self.calc_reward_fn(self.platform.params,self.state)
        else:
            # calc reward - default. can be overriden by wrapper
            reward = (self.state.pl1 - self.platform.params.pl1_max) + (
                        self.state.pl2 - self.platform.params.pl2_max) - 1
            if self.state.tskin > (self.platform.params.tskin_max - self.platform.params.tskin_ofst):
                reward -= 1000
        return reward


    def _is_done(self):
        return (len(self.workload_stack) == 0) and (len(self.wip_queue) == 0) and (self.req_power_for_curr_step == 0)

    def step(self, action):
        # do one step in the system given the action and return the state of the system and the reward
        # as a response to that action
        # i.e. we assume the action was provided at time step n, so we move the system to time step n+1 and
        # return the impact on the system assuming the action was performed during the nth time step and we return
        # the state of the system and the reward at the end of this period (thus they are at time step n+1)
        if (not (isinstance(action, np.int64) or isinstance(action, int) or isinstance(action, np.int32))
                or (action < 0) or (action >= self.action_space.n)):
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))

        # add the next power to the buffer of compute we need to perform
        if (self.req_power_for_curr_step==0) and (len(self.wip_queue)==0) and (len(self.workload_stack)>0):
            self._pop_workload()
        if len(self.wip_queue) > 0:
            self.req_power_for_curr_step += self.wip_queue.pop(0)

        # calculate the requested power levels
        dpl1, dpl2 = self.dPL[action]
        # calc the power levels the agent asked to apply
        req_pl2 = max(min(self.state.pl2 + dpl2, self.platform.params.pl2_max), self.platform.params.pl2_min)
        req_pl1 = max(min(self.state.pl1 + dpl1, self.platform.params.pl1_max), self.platform.params.pl1_min)
        req_pl1 = min(req_pl1,req_pl2)

        # to test fixed values - for debug
        if self.fixed_pl:
            req_pl1=self.fixed_pl[0]
            req_pl2=self.fixed_pl[1]

        # assuming these levels are supported given system state,
        power_budget, ia_clip_reason = self.platform.get_power_budget(req_pl1, req_pl2)

        bg_power = np.abs(np.random.randn())  # random background power

        if power_budget <= (self.req_power_for_curr_step + bg_power):
            workload_power_consumed = power_budget - bg_power
            total_power_consumed = power_budget
        else:
            workload_power_consumed = self.req_power_for_curr_step
            total_power_consumed = workload_power_consumed + bg_power

        actual_power_consumed = min(power_budget, self.req_power_for_curr_step)

        self.platform.consume_power(total_power_consumed,self.wip_name)
        self.req_power_for_curr_step -= workload_power_consumed

        self.state = self.platform.get_state()
        self.mean_ips_during_workload.append(self.state.ips_mean)

        reward = self._calc_reward()
        done = self._is_done()
        # Optionally we can pass additional info, we are not using that for now
        info = {'IAClipReason': ia_clip_reason}
        # scores = self.workload_scores if done else np.nan

        if self.log_output:
            self.tsdf.loc[0] = [self.step_idx] + list(self.state.__dict__.values()) + [info['IAClipReason']] + \
                               [self.last_score]
            self.out_df = self.out_df.append(self.tsdf)
        self.step_idx += 1
        self.last_score = np.nan
        return self.get_state(), reward, done, info

    def set_episode_workloads(self,episode_workloads):
        self.workload_params=episode_workloads
        self.reset()


    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()

    def close(self):
        if self.log_output:
            self.out_df.set_index('timestamp', inplace=True)
            self.out_df.to_csv(self.log_output)


# endregion


###########################################################################################################
# Policies

def random_policy(params=None,state=None,dPL2act=None):
    act = np.random.randint(0, 9)
    return act

def greedy_policy(params,state,dPL2act):
    a1 = 0.5 if state[0] < params.pl1_max else 0
    a2 = 0.5 if state[1] < params.pl2_max else 0
    return dPL2act[(a1, a2)]

def fixed_policy(pl1,pl2,params,state,dPL2act):
    a1 = 0.5 * np.sign(pl1-state[0])
    a2 = 0.5 * np.sign(pl2-state[1])
    return dPL2act[(a1, a2)]

POLICIES = {'r': random_policy,'g':greedy_policy,'f':fixed_policy}


def main():
    platform = PLATFORMS['Scarlet']


    # to create the following experiment: (benchmark num_runs sec_between_runs)
    # time between iterations : 300 sec
    # - cb15 10 120  --> 10*(['cb15']+[('cooldown',120)])
    # - cb15 10 60   --> 10*(['cb15']+[('cooldown',60)])
    # - cb15 10 30   --> 10*(['cb15']+[('cooldown',30)])
    # do the following:
    # episode_workloads = 10*(['cb15']+[('cooldown',120)]) +\
    #                     [('cooldown',300)] + \
    #                     10*(['cb15']+[('cooldown',60)]) + \
    #                     [('cooldown',300)] + \
    #                     10*(['cb15']+[('cooldown',30)])

    # the following is worth 33020 sec = 9H:10M:20S - runs 260 sec on the server (single core)
    # episode_workloads = EPISODES['cb_long']
    episode_workloads = EPISODES['cb20mix3']
    env = DTTEnvSim(platform, episode_workloads=episode_workloads, norm_obs=False,
                    # fixed_pl=(SCARLET_INITIAL_PL1,SCARLET_INITIAL_PL2),
                    log_output=os.path.join(os.getcwd(),'tmp'))
    policy = POLICIES['r']  # (f)ixed, (g)reedy, or (r)andom
    dPL2act = {v: np.int64(k) for k, v in env.dPL.items()}
    obs = env.reset()
    done = False
    esif_cols = ['timestamp'] + list(platform.state.__dict__.keys()) + ['Clip']
    out_df = pd.DataFrame(columns=esif_cols)
    tsdf = pd.DataFrame(columns=esif_cols)
    ts = 0
    total_rew = 0
    while not done:
        act = policy(platform.params,obs,dPL2act)
        # act = np.random.randint(0,9)    # for random policy
        obs, rew, done, info = env.step(act)
        total_rew += rew
        # obs = ['pl1','pl2','power','tj','tskin','ewma']
        # policy : as long as we're below the max value, aim to increase
        tsdf.loc[0] = [ts] + list(obs) + [info['IAClipReason']]
        out_df = out_df.append(tsdf)
        ts += 1
    scores = env.get_scores()
    env.close()
    print(f'session completed. total reward: {total_rew}, average score:{np.mean(scores)}')
    print(f'episode scores: {scores}')
    out_df['ips_mean'] = out_df['ips_mean'] / 1e8
    out_df.set_index('timestamp', inplace=True)
    os.makedirs('./tmp',exist_ok=True)
    out_df.to_csv('./tmp/sim_esif.csv')
    out_df.loc[:, esif_cols[1:-1]].plot(figsize=(8, 4), grid=True)
    plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    main()
