#####################################################################
# my_envs.py
# Implementations of custom envs
# MLA Template Equivalent: train.custom_envs
import gym
from gym import spaces
import numpy as np
import os
import pandas as pd
from abc import ABC, abstractmethod
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
class DTTEnvReal(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple env that imitates the L2P behaviour
    """
    # Because of google colab, we cannot implement the GUI ('human' render mode)
    metadata = {'render.modes': ['console']}

    def __init__(self, obs_dim=6,n_act=9):
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

###########################################################
#region DTTEnvSim
from collections import namedtuple
from enum import Enum
from copy import copy
from scipy.interpolate import interp1d

IDLE_POWER = 1.0
IDLE_TSKIN = 40.0
IDLE_TJ = 45.0


#####################################
# Workload definitions
# the benchmarks are defined as list of tuples : (Power,Seconds)
BENCHMARKS={'cb15':[(0,IDLE_POWER),(5,45),(10,45),(20,30),(40,30),(50,28),(60,IDLE_POWER)],
            'cb20':[(0,IDLE_POWER),(10,44),(13,44),(20,32),(35,30),(45,30),(46,25),(200,25),(205,IDLE_POWER)],
            'cooldown':[(0,IDLE_POWER),(300,IDLE_POWER)]}


####################################
# Platform definition
# a simplified model to predict Tskin from Tj is:
# tskn[n]=tskn[n-1]+Tj2TsknFactor*(tj[n]-tj[n-1])

Tj2Tskn=namedtuple("Tj2Tskn",['intercept','tj_coef','tjm1_coef','tsm1_coef'])
# Tj2TsknFactor = 0.3
Tj2TsknFactor={'cb15X30':Tj2Tskn(0.631,0.003,0.007,0.973),
                'cb15':Tj2Tskn(0.8,0.0027,0.012,0.963),
                'cb20':Tj2Tskn(0.675,0.0039,0.085,0.969),
                'cb20X30':Tj2Tskn(0.785,0.003,0.01,0.968)}

# tj[n]= intercept + p_coef * power[n] + tj_coef* tj[n-1]
P2Tj=namedtuple("P2Tj",['intercept','p_coef','tj_coef'])
Power2TjFactor={'cb15X30':P2Tj(47.74,0.791,0.277),
                'cb15':P2Tj(11.17,0.419,0.75),
                'cb20':P2Tj(14.63,0.581,0.68),
                'cb20X30':P2Tj(45.38,0.797,0.303)}

PlatformParamsScarlet = namedtuple("PlatformParamsScarlet", ['tdp', 'PL1max','PL1min', 'PL2max', 'PL2min', 'tj_max',
                                                             'tskn_max', 'tskn_ofst','tau','p2tj','tj2ts'])
SCARLET_TDP=15.0
SCARLET_PL1MAX=44.0
SCARLET_PL1MIN=9.0
SCARLET_PL2MAX=44.0
SCARLET_PL2MIN=24.0
SCARLET_TJMAX=100.0
SCARLET_TSKINMAX=60
SCARLET_TSKINOFST=6.0
SCARLET_TAU=28.0

PLATFORMS={'Scarlet':PlatformParamsScarlet(tdp=SCARLET_TDP,PL1max=SCARLET_PL1MAX,PL1min=SCARLET_PL1MIN,PL2max=SCARLET_PL2MAX,PL2min=SCARLET_PL2MIN,
                                           tj_max=SCARLET_TJMAX,tskn_max=SCARLET_TSKINMAX,tskn_ofst=SCARLET_TSKINOFST,tau=SCARLET_TAU,
                                           p2tj=Power2TjFactor,tj2ts=Tj2TsknFactor)}




def predict_tj(power:float,tj_nm1, tj_max, coefs:P2Tj):
    return np.minimum(coefs.intercept+coefs.p_coef*power+coefs.tj_coef*tj_nm1,tj_max)


def predict_tskn(curr_tj,prev_tj,prev_tskn,tskn_max,coefs:Tj2Tskn):
    return np.minimum(coefs.intercept+coefs.tj_coef*curr_tj+coefs.tjm1_coef*prev_tj+coefs.tsm1_coef*prev_tskn,tskn_max)

class PlatformState(object):
    dim=6
    def __init__(self,pl1=0.0,pl2=0.0,power=0.0,tj=0.0,tskn=0.0,ewma=0.0):
        self.pl1=pl1
        self.pl2=pl2
        self.power=power
        self.tj=tj
        self.tskn=tskn
        self.ewma=ewma
    def __repr__(self):
        return 'State(pl1={}, pl2={}, power={}, tj={}, tskn={}, ewma={})'.\
            format(self.pl1,self.pl2,self.power,self.tj,self.tskn,self.ewma)

class Platform(object):
    def __init__(self,platform_params):
        self.params=platform_params
        self.state=None
        self.reset_state()


    @abstractmethod
    def _run_dtt(self,req_pl1,req_pl2):
        raise NotImplementedError

    @abstractmethod
    def _run_pcode(self,act_pl1,act_pl2):
        raise NotImplementedError
    
    @abstractmethod
    def reset_state(self):
        raise NotImplementedError
        
    def set_state(self,state):
        self.state = state

    @abstractmethod
    def _norm_state(self):
        raise NotImplementedError


    def get_state(self,norm=False):
        if norm:
            return self._norm_state()
        return self.state


    def get_power_budget(self,req_pl1,req_pl2):
        act_pl1,act_pl2,clip_reason = self._run_dtt(req_pl1,req_pl2)
        power_budget,clip_reason_pcode = self._run_pcode(act_pl1,act_pl2)
        clip_reason += clip_reason_pcode
        if len(clip_reason)==0:
            clip_reason+=(IAClipReason.No_Clip,)
        self.state.pl1=act_pl1
        self.state.pl2=act_pl2
        return power_budget,clip_reason

    def consume_power(self,power_consumed):
        # update ewma
        self.state.power=power_consumed
        self.state.ewma = self.state.ewma + (1.0/self.params.tau)*(self.state.power-self.state.ewma)

        # update thermal sensors
        prev_tj = self.state.tj 
        self.state.tj = predict_tj(self.state.power,prev_tj,self.params.tj_max,self.params.p2tj['cb15'])
        prev_tskn = self.state.tskn
        self.state.tskn = predict_tskn(self.state.tj,prev_tj,prev_tskn,self.params.tskn_max,self.params.tj2ts['cb15'])


class StateScarlet(PlatformState):
    def __init__(self,pl1=0.0,pl2=0.0,power=0.0,tj=0.0,tskn=0.0,ewma=0.0):
        super(StateScarlet,self).__init__(pl1,pl2,power,tj,tskn,ewma)
        self.pl1=pl1
        self.pl2=pl2
        self.power=power
        self.tj=tj
        self.tskn=tskn
        self.ewma=ewma
    def __repr__(self):
        return 'State(pl1={}, pl2={}, power={}, tj={}, tskn={}, ewma={})'.\
            format(self.pl1,self.pl2,self.power,self.tj,self.tskn,self.ewma)


class Scarlet(Platform):
    def __init__(self,platform_params):
        super(Scarlet,self).__init__(platform_params)

    def _run_dtt(self,req_pl1,req_pl2):
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
        if self.state.tskn >= (self.params.tskn_max-self.params.tskn_ofst):
            # thermal event. reduce act_pl1 to PL1_min
            act_pl1=self.params.PL1min
            clip_reason += (IAClipReason.Thermal_Event,)
        return act_pl1,act_pl2,clip_reason
        
    
    def _run_pcode(self,act_pl1,act_pl2):
        # the following depends on the way we calculate ewma
        clip_reason=()
        power_budget=self.state.pl2
        if self.state.ewma >= act_pl1:
            clip_reason += (IAClipReason.Max_Turbo_Limit,)
            power_budget = self.state.pl1
        return power_budget,clip_reason


    def reset_state(self):
        '''
        reset_state simulate a system that has been IDLE for a long time
        '''
        # note that ewma depends on the way it is calculated. currently set to IDLE_POWER.
        # with the new formula it should be SCARLET_PL1MAX-IDLE_POWER
        self.state = StateScarlet(pl1=SCARLET_PL1MAX,pl2=SCARLET_PL2MAX,power=IDLE_POWER,tj=IDLE_TJ,tskn=IDLE_TSKIN,ewma=IDLE_POWER)
        # inject some noise
        self.state.tj = np.abs(self.state.tj + np.random.randn())
        self.state.tskn = np.abs(self.state.tskn + np.random.randn())
        self.state.power = np.abs(self.state.power + np.random.randn())
        return 

    def _norm_state(self):
        pl1=self.state.pl1/self.params.PL1max
        pl2=self.state.pl2/self.params.PL2max
        power=self.state.power/self.params.PL2max
        tj=self.state.tj/self.params.tj_max
        tskn=self.state.tskn/self.params.tskn_max
        ewma=self.state.ewma/self.params.PL1max
        return StateScarlet(pl1,pl2,power,tj,tskn,ewma)




class Billie(Platform):
    def __init__(self,platform_params):
        super(Billie,self).__init__(platform_params)
    
    def _run_dtt(self,req_pl1,req_pl2):
        pass
    
    def _run_pcode(self,act_pl1,act_pl2):
        pass


class IAClipReason(Enum):
    No_Clip = 0
    Thermal_Event = 1
    Max_Turbo_Limit = 2


class DTTEnvSim(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple env that imitates the L2P behaviour
    """
    # Because of google colab, we cannot implement the GUI ('human' render mode)
    metadata = {'render.modes': ['console']}
    dPL={0:(-0.5,-0.5),1:(0,-0.5),2:(0.5,-0.5),
         3:(-0.5,0),4:(0,0),5:(0.5,0),
         6:(-0.5,0.5),7:(0,0.5),8:(0.5,0.5)}

    def __init__(self,platform,workload_params, norm_obs=True, log_output=None):
        super(DTTEnvSim, self).__init__()
        
        self.platform = platform
        self.workload_params=workload_params
        self.state = self.platform.get_state()      # assuming not normalized state as default
        # the observation space include obs_dim float values
        self.obs_dim = len(self.state.__dict__.keys())

        # Currently assuming discrete action space with n_act actions
        self.act_dim = 1
        self.norm_obs=norm_obs
        self.log_output=log_output
        if self.log_output is not None:
            self.log_output = os.path.join(self.log_output,'DTTSim_esif.csv')
            self.esif_cols = ['timestamp']+list(self.state.__dict__.keys())+['Clip']
            self.out_df = pd.DataFrame(columns=self.esif_cols)
            self.tsdf = pd.DataFrame(columns=self.esif_cols)        # include a single timestep

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions, we have two: left and right
        n_act= len(self.dPL)
        self.action_space = spaces.Discrete(n_act)
        # The observation will be the coordinate of the agent
        # this can be described both by Discrete and Box space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,shape=(self.obs_dim,), dtype=np.float32)
        self.max_path_length = np.inf

    def get_state(self):
        state = self.platform.get_state(norm=self.norm_obs).__dict__.values()
        return np.array([s for s in state])

    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        self.step_idx=0
        # reset the workload
        # fill in the queue of steps to perform. each includes the power it requires
        queue = []
        for wl in self.workload_params:
            x = [c[0] for c in wl]
            y = [c[1] for c in wl]
            f = interp1d(x, y)
            queue.append(f(range(x[-1]+1)))
        self.queue = [e for s in queue for e in s]
        self.req_power_for_curr_step = 0
        # reset the platform
        self.platform.reset_state()
        self.state=self.platform.get_state()        # get unnormalized state from the platform
        return self.get_state()

    def _is_done(self):
        return (len(self.queue)==0) and (self.req_power_for_curr_step==0)

    def step(self, action):
        # do one step in the system given the action and return the state of the system and the reward
        # as a response to that action
        # i.e. we assume the action was provided at time step n, so we move the system to time step n+1 and
        # return the impact on the system assuming the action was performed during the nth time step and we return
        # the state of the system and the reward at the end of this period (thus they are at time step n+1)
        if (not isinstance(action,np.int64)) or (action < 0) or (action >= self.action_space.n):
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))

        # add the next power to the buffer of compute we need to perform
        if len(self.queue)>0:
            self.req_power_for_curr_step += self.queue.pop(0)

        # calculate the requested power levels
        dpl1,dpl2 = self.dPL[action]
        # calc the power levels the agent asked to apply
        req_pl1 = max(min(self.state.pl1 + dpl1,self.platform.params.PL1max),self.platform.params.PL1min)
        req_pl2 = max(min(self.state.pl2 + dpl2,self.platform.params.PL2max),self.platform.params.PL2min)
        
        # assuming these levels are supported given system state,
        power_budget,ia_clip_reason = self.platform.get_power_budget(req_pl1,req_pl2)

        bg_power = np.abs(np.random.randn())        # random background power

        if power_budget<=(self.req_power_for_curr_step+bg_power):
            workload_power_consumed = power_budget-bg_power
            total_power_consumed = power_budget
        else:
            workload_power_consumed = self.req_power_for_curr_step
            total_power_consumed = workload_power_consumed + bg_power

        actual_power_consumed = min(power_budget,self.req_power_for_curr_step)

        self.platform.consume_power(total_power_consumed)
        self.req_power_for_curr_step -= workload_power_consumed

        self.state = self.platform.get_state()

        # calc reward
        reward = (self.state.pl1 - self.platform.params.PL1max) + (self.state.pl2 - self.platform.params.PL2max) - 1
        if self.state.tskn > (self.platform.params.tskn_max-self.platform.params.tskn_ofst):
            reward -= 1000

        done = self._is_done()
        # Optionally we can pass additional info, we are not using that for now
        info = {'IAClipReason':ia_clip_reason}
        if self.log_output:
            self.tsdf.loc[0] = [self.step_idx]+ list(self.state.__dict__.values()) + [info['IAClipReason']]
            self.out_df = self.out_df.append(self.tsdf)
        self.step_idx += 1

        return self.get_state(), reward, done, info



    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()

    def close(self):
        if self.log_output:
            self.out_df.set_index('timestamp', inplace=True)
            self.out_df.to_csv(self.log_output)

#endregion

def main():
    
    platform = Scarlet(PLATFORMS['Scarlet'])

    env = DTTEnvSim(platform,workload_params=[BENCHMARKS['cb15'],BENCHMARKS['cooldown']]*20,norm_obs=False,log_output=os.getcwd())
    dPL2act={v:np.int64(k) for k,v in env.dPL.items()}
    obs=env.reset()
    done = False
    esif_cols=['timestamp']+list(platform.state.__dict__.keys())+['Clip']
    out_df = pd.DataFrame(columns=esif_cols)
    tsdf=pd.DataFrame(columns=esif_cols)
    ts = 0
    a1 = 0
    a2 = 0   # start with no change
    total_rew = 0
    while not done:
        act = dPL2act[(a1,a2)]
        obs,rew,done,info=env.step(act)
        total_rew += rew
        # obs = ['pl1','pl2','power','tj','tskn','ewma']
        # policy : as long as we're below the max value, aim to increase
        a1 = 0.5 if obs[0]<env.platform.params.PL1max else 0
        a2 = 0.5 if obs[1]<env.platform.params.PL2max else 0
        tsdf.loc[0]=[ts]+list(obs)+[info['IAClipReason']]
        out_df=out_df.append(tsdf)
        ts+=1
    env.close()
    print('session completed. total reward: {}'.format(total_rew))
    out_df.set_index('timestamp',inplace=True)
    out_df.to_csv('sim_esif.csv')
    out_df.loc[:,esif_cols[1:-1]].plot(figsize=(8,4),grid=True)
    plt.show()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    main()
