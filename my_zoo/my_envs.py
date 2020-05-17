#####################################################################
# my_envs.py
# Implementations of custom envs
# MLA Template Equivalent: train.custom_envs
import gym
from gym import spaces
import numpy as np

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
Platform = namedtuple("Platform", ['tdp', 'PL1max','PL1min', 'PL2max', 'PL2min', 'tj_max', 'tskn_max', 'tskn_ofst','tau'])

IDLE_POWER=1.0

class IAClipReason(Enum):
    No_Clip = 0
    Thermal_Event = 1
    Max_Turbo_Limit = 2


# the benchmarks are defined as list of tuples : (Power,Seconds)
BENCHMARKS={'cb15':[(0,IDLE_POWER),(5,45),(10,45),(20,30),(40,30),(50,28),(60,IDLE_POWER)],
            'cb20':[(0,IDLE_POWER),(10,44),(13,44),(20,32),(35,30),(45,30),(46,25),(200,25),(205,IDLE_POWER)],
            'cooldown':[(0,IDLE_POWER),(300,IDLE_POWER)]}

PLATFORMS={'Scarlet':Platform(tdp=15.0,PL1max=44.0,PL1min=9.0,PL2max=44.0,PL2min=24.0,tj_max=100.0,
                              tskn_max=60.0,tskn_ofst=6.0,tau=28.0)}

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

def predict_tj(power:float,tj_nm1, coefs:P2Tj):
    return np.minimum(coefs.intercept+coefs.p_coef*power+coefs.tj_coef*tj_nm1,100)

def predict_tskn(curr_tj,prev_tj,prev_tskn,tskn_max,coefs:Tj2Tskn):
    return np.minimum(coefs.intercept+coefs.tj_coef*curr_tj+coefs.tjm1_coef*prev_tj+coefs.tsm1_coef*prev_tskn,tskn_max)


class State(object):
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

    def __init__(self,platform=PLATFORMS['Scarlet'],workload=[BENCHMARKS['cb15'],BENCHMARKS['cooldown']],
                 norm_obs=True, log_output=None):
        super(DTTEnvSim, self).__init__()
        # the observation space include obs_dim float values
        self.obs_dim = State.dim
        # Currently assuming discrete action space with n_act actions
        self.act_dim = 1
        self.norm_obs=norm_obs
        self.log_output=log_output

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions, we have two: left and right
        n_act= len(self.dPL)
        self.action_space = spaces.Discrete(n_act)
        # The observation will be the coordinate of the agent
        # this can be described both by Discrete and Box space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,shape=(self.obs_dim,), dtype=np.float32)
        self.max_path_length = np.inf

        self.platform=platform
        self.workload=workload
        self.initial_state = State(pl1=self.platform.PL1max,pl2=self.platform.PL2max,power=IDLE_POWER,
                                   tj=45.0,tskn=40.0,ewma=IDLE_POWER)
        self.state = self.initial_state


    def _norm_state(self,state):
        pl1=state.pl1/self.platform.PL1max
        pl2=state.pl2/self.platform.PL2max
        power=state.power/self.platform.PL2max
        tj=state.tj/self.platform.tj_max
        tskn=state.tskn/self.platform.tskn_max
        ewma=state.ewma/self.platform.PL1max
        return np.array([pl1,pl2,power,tj,tskn,ewma])

    def get_state(self,state):
        if self.norm_obs:
            return self._norm_state(state)
        else:
            return np.array([state.pl1,state.pl2,state.power,state.tj,state.tskn,state.ewma])

    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        self.step_idx=0
        # fill in the queue of steps to perform. each includes the power it requires
        queue = []
        for wl in self.workload:
            x = [c[0] for c in wl]
            y = [c[1] for c in wl]
            f = interp1d(x, y)
            queue.append(f(range(x[-1]+1)))
        self.queue = [e for s in queue for e in s]
        self.req_power_for_curr_step = 0
        # set the initial state of the system
        self.state = self.initial_state
        # inject some noise
        self.state.tj += np.random.randn()
        self.state.tskn += np.random.randn()
        self.state.power += np.random.randn()
        return self.get_state(self.state)

    def _apply_pl_req(self,req_pl1,req_pl2):
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
        if self.state.tskn >= (self.platform.tskn_max-self.platform.tskn_ofst):
            # thermal event. reduce act_pl1 to PL1_min
            act_pl1=self.platform.PL1min
            clip_reason += (IAClipReason.Thermal_Event,)
        if len(clip_reason)==0:
            clip_reason+=(IAClipReason.No_Clip,)
        return act_pl1,act_pl2,clip_reason

    def _update_state_reward(self,pl1,pl2):
        next_state = copy(self.state)
        self.req_power_for_curr_step += self.queue.pop(0)        # add the next power to the buffer of compute we need to perform
        next_state.pl1 = pl1
        next_state.pl2 = pl2
        clip_reason=()
        # check turbo budget and derive the maximum power we can support:
        power_capacity = next_state.pl2

        if next_state.ewma >= next_state.pl1:
            clip_reason += (IAClipReason.Max_Turbo_Limit,)
            power_capacity = self.state.pl1

        actual_power_consumed = min(power_capacity,self.req_power_for_curr_step)

        # now the system state will be updated according to the actual power consumed
        next_state.power = actual_power_consumed + np.abs(np.random.randn())
        self.req_power_for_curr_step -= actual_power_consumed

        # update ewma
        next_state.ewma = next_state.ewma + (1.0/self.platform.tau)*(next_state.power-next_state.ewma)

        # update thermal sensors
        next_state.tj = predict_tj(next_state.power,self.state.tj,Power2TjFactor['cb15'])
        next_state.tskn = predict_tskn(next_state.tj,self.state.tj,self.state.tskn,self.platform.tskn_max,
                                       Tj2TsknFactor['cb15'])

        # calc reward
        reward = (pl1 - self.platform.PL1max) + (pl2 - self.platform.PL2max)
        if next_state.tskn > (self.platform.tskn_max-self.platform.tskn_ofst):
            reward -= 1000
        if len(clip_reason)==0:
            clip_reason+=(IAClipReason.No_Clip,)
        return next_state,reward,clip_reason

    def _is_done(self):
        return (len(self.queue)==0) and (self.req_power_for_curr_step==0)

    def step(self, action):
        # do one step in the system given the action and return the state of the system and the reward
        # as a response to that action
        # i.e. we assume the action was provided at time step n, so we move the system to time step n+1 and
        # return the impact on the system assuming the action was performed during the nth time step and we return
        # the state of the system and the reward at the end of this period (thus they are at time step n+1)
        if (not isinstance(action,int)) or (action < 0) or (action >= self.action_space.n):
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))

        # calculate the requested power levels
        dpl1,dpl2 = self.dPL[action]
        # calc the power levels the agent asked to apply
        req_pl1 = max(min(self.state.pl1 + dpl1,self.platform.PL1max),self.platform.PL1min)
        req_pl2 = max(min(self.state.pl2 + dpl2,self.platform.PL2max),self.platform.PL2min)
        ia_clip_reason=set()
        # assuming these levels are supported given system state,
        act_pl1,act_pl2,clip_reason = self._apply_pl_req(req_pl1,req_pl2)
        ia_clip_reason.add(clip_reason)
        self.step_idx += 1
        next_state,reward,clip_reason = self._update_state_reward(act_pl1,act_pl2)
        ia_clip_reason.add(clip_reason)
        done = self._is_done()
        # Optionally we can pass additional info, we are not using that for now
        info = {'IAClipReason':ia_clip_reason}
        self.state = copy(next_state)
        return self.get_state(next_state), reward, done, info



    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()

    def close(self):
        pass
#endregion

def main():
    env = DTTEnvSim(workload=[BENCHMARKS['cb15'],BENCHMARKS['cooldown']]*20,norm_obs=False)
    dPL2act={v:k for k,v in env.dPL.items()}
    obs=env.reset()
    done = False
    esif_cols=['timestamp','POWER','tj','tskin','MMIO PL1','MMIO PL2','ewma','Clip']
    out_df = pd.DataFrame(columns=esif_cols)
    tsdf=pd.DataFrame(columns=esif_cols)
    ts=0
    a1 = 0
    a2 = 0   # start with no change
    total_rew = 0
    while not done:
        act = dPL2act[(a1,a2)]
        obs,rew,done,info=env.step(act)
        total_rew += rew
        # obs = ['pl1','pl2','power','tj','tskn','ewma']
        # policy : as long as we're below the max value, aim to increase
        a1 = 0.5 if obs[0]<env.platform.PL1max else 0
        a2 = 0.5 if obs[1]<env.platform.PL2max else 0
        tsdf.loc[0]=[ts,obs[2],obs[3],obs[4],obs[0],obs[1],obs[5],info['IAClipReason']]
        out_df=out_df.append(tsdf)
        ts+=1
    print('session completed. total reward: {}'.format(total_rew))
    out_df.set_index('timestamp',inplace=True)
    out_df.to_csv('sim_esif.csv')
    out_df.loc[:,['POWER','tj','tskin','MMIO PL1','MMIO PL2','ewma']].plot(figsize=(8,4),grid=True)
    plt.show()

if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt
    main()
