from collections import deque
import os
import numpy as np
import argparse
from tqdm import tqdm
import sys
path_to_curr_file = os.path.realpath(__file__)
proj_root = os.path.dirname(os.path.dirname(path_to_curr_file))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)
from my_zoo.my_envs import PLATFORMS, DTTEnvSim,EPISODES,random_policy,fixed_policy,greedy_policy
import pandas as pd
from functools import partial





###########################################################################################################
# Reward Shaping
def reward_pl1_pl2_overshoot(params,state):
    pl1_norm = (state.pl1-params.pl1_min)/(params.pl1_max-params.pl1_min)
    pl2_norm = (state.pl2-params.pl2_min)/(params.pl2_max-params.pl2_min)
    tskin_norm = (state.tskin - params.tskin_idle)/(params.tskin_max - params.tskin_idle)
    tj_norm = (state.tj - params.tj_idle)/(params.tj_max - params.tj_idle)
    reward = -(1 - pl1_norm)**0.5 - (1 - pl2_norm)**0.5 - 2*(tskin_norm>=1) -2*(tj_norm>=1)
    return reward


def reward_ips(params,state):
    reward = state.ips_mean/(10**8) - 1000*((state.tskin>=(params.tskin_max-params.tskin_ofst)) |
                                             (state.tmem>=params.tmem_max))
    return reward


def reward_ips_a(params,state):
    tskin_norm = (state.tskin - params.tskin_idle)/(params.tskin_max - params.tskin_idle)
    tj_norm = (state.tj - params.tj_idle)/(params.tj_max - params.tj_idle)
    reward = state.ips_mean/(10**10) - 1000*(tskin_norm>=1) - 1000*(tj_norm>=1)
    return reward


def orig_reward(params,state):
    reward = state.pl1 - params.pl1_max + state.pl2 - params.pl2_max - \
             10*((state.tskin>=(params.tskin_max-params.tskin_ofst)) | (state.tmem>=params.tmem_max))
    return reward

def reward_3(params,state):
    reward = (state.pl1 - params.pl1_max) + (state.pl2 - params.pl2_max) + (state.ips_mean/(10**9)) - \
             1000*((state.tskin>=(params.tskin_max-params.tskin_ofst)) | (state.tmem>=params.tmem_max))
    return reward

def reward_4(params,state):
    reward = (state.pl1 / params.pl1_max) + (state.pl2 / params.pl2_max) + (state.ips_mean/(10**9)) - \
             100*((state.tskin>=(params.tskin_max-params.tskin_ofst)) | (state.tmem>=params.tmem_max))
    return reward

# similar to orig_reward only with higher penalty for overshoot
def reward_6(params,state):
    reward = state.pl1 - params.pl1_max + state.pl2 - params.pl2_max - \
             100*((state.tskin>=(params.tskin_max-params.tskin_ofst)) | (state.tmem>=params.tmem_max))
    return reward


# similar to orig_reward only with higher penalty for overshoot
def reward_7(params,state):
    reward = (state.ips_mean/(10**9)) - int(state.pl1 < params.pl1_max) - int(state.pl2 < params.pl2_max) -\
             10*((state.tskin>=(params.tskin_max-params.tskin_ofst)) | (state.tmem>=params.tmem_max))
    return reward

# similar to orig_reward only with higher penalty for overshoot
def reward_8(params,state):
    reward = (state.ips_mean/(10**10)) - ((state.tskin>=(params.tskin_max-params.tskin_ofst)) | (state.tmem>=params.tmem_max))
    return reward





REWARDS={0:orig_reward,1:reward_ips,2:reward_pl1_pl2_overshoot, 3:reward_3, 4:reward_4, 5:reward_ips_a, 6: reward_6,
         7: reward_7, 8: reward_8}


def parse_cmd_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--n_episodes', help='number of episodes', default=30,type=int)
    parser.add_argument('-b', '--benchmark', help='benchmark to run', type=str)
    parser.add_argument('--pf', help='fixed policy', type=int, nargs=2, action='append')
    parser.add_argument('--pg', help='greedy policy', action='store_true')
    parser.add_argument('-r', '--reward', help='reward function',default=0, type=int)
    parser.add_argument('--platform', help='type of platform: Scarlet', default='Scarlet', type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_cmd_line()
    platform = PLATFORMS[args.platform]
    # to create the following experiment: (benchmark num_runs sec_between_runs)
    # time between iterations : 300 sec
    # - cb15 10 120  --> 10*(['cb15']+[('cooldown',120)])
    # - cb15 10 60   --> 10*(['cb15']+[('cooldown',60)])
    # - cb15 10 30   --> 10*(['cb15']+[('cooldown',30)])
    # do the following:
    os.makedirs('tmp', exist_ok=True)
    n_episodes=args.n_episodes
    if args.benchmark and (args.benchmark in EPISODES.keys()):
        benchmarks=[args.benchmark]
        out_csv= os.path.join(os.getcwd(),'tmp',f'rew_scores_{args.reward}_{args.benchmark}.csv')
    else:
        benchmarks=['cb15','cb15mr','cb20','cb20mr']
        out_csv = os.path.join(os.getcwd(),'tmp',f'rew_scores_{args.reward}_all.csv')
    print(f'running on {benchmarks}')

    # policies = {'fixed':partial(fixed_policy,args.policy[0],args.policy[1]),
    #             'random':random_policy,'greedy':greedy_policy}
    # seems like only random explores well
    policies = {'random':random_policy}
    if args.pf:
        for vals in args.pf:
            policies.update({f'fixed_{vals[0]}_{vals[1]}':partial(fixed_policy,vals[0],vals[1])})
    if args.pg:
        policies.update({'greedy':greedy_policy})

    reward_fn = REWARDS[args.reward]

    # env = DTTEnvSim(platform, episode_workloads=episode_workloads, calc_reward_fn= reward_fn, norm_obs=False)
    # env = DTTStateRewardWrapper(env=env, feature_extractor=feature_extraction, reward_calc=orig_reward, n_frames=5)

    # feat_cols = list(feature_extraction(None))
    # tsdf = pd.DataFrame(columns=feat_cols)
    # tsdf.to_csv('sim_features.csv',index=False)
    dPL2act = {v: np.int64(k) for k, v in DTTEnvSim.dPL.items()}
    esif_cols = ['policy','timestamp'] + list(platform.state.__dict__.keys()) + ['Clip']
    tsdf = pd.DataFrame(columns=esif_cols)
    rew_df = pd.DataFrame(columns=['benchmark','policy','episode id','reward','avg score'])
    rew_feat_csv_file = os.path.join(os.getcwd(),'tmp',f'rew_{args.reward}_features.csv')
    tsdf.to_csv(rew_feat_csv_file, index=False)
    ri=0
    for bench in benchmarks:
        print(f'testing benchmark {bench}')
        episode_workloads = EPISODES[bench]
        env = DTTEnvSim(platform, episode_workloads=episode_workloads, calc_reward_fn=reward_fn, norm_obs=False)
        for name,policy in policies.items():
            print(f'running {name} policy')
            for ei in tqdm(range(n_episodes)):
                obs = env.reset()
                done = False
                out_df = pd.DataFrame(columns=esif_cols)
                ts = 0
                total_rew = 0
                while not done:
                    act = policy(platform.params,obs,dPL2act)
                    obs, rew, done, info = env.step(act)
                    total_rew += rew
                    # obs = ['pl1','pl2','power','tj','tskin','ewma']
                    # policy : as long as we're below the max value, aim to increase
                    tsdf.loc[0] = [name]+[ts] + list(obs) + [info['IAClipReason']]
                    out_df = out_df.append(tsdf)
                    ts += 1
                scores = env.get_scores()
                avg_score = round(np.mean(scores),2)
                print(f'session completed. total reward: {round(total_rew,2)}, average score:{avg_score}')
                # print(f'scores: {scores}')
                rew_df.loc[ri]=[bench,name,ei,total_rew,avg_score]
                ri+=1
                out_df.to_csv(rew_feat_csv_file,mode='a',header=False,index=False)
        env.close()
    print(f'saving results to {out_csv}')
    rew_df.to_csv(out_csv,index=False)

if __name__ == '__main__':
    main()
