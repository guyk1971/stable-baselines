import random
from typing import Optional, List, Union

import numpy as np
import pandas as pd
from tqdm import tqdm
import ast
from stable_baselines.common.segment_tree import SumSegmentTree, MinSegmentTree
from stable_baselines.common.vec_env import VecNormalize
from stable_baselines import logger

class ReplayBuffer(object):
    def __init__(self, size: int):
        """
        Implements a ring buffer (FIFO).

        :param size: (int)  Max number of transitions to store in the buffer. When the buffer overflows the old
            memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self) -> int:
        return len(self._storage)

    @property
    def storage(self):
        """[(Union[np.ndarray, int], Union[np.ndarray, int], float, Union[np.ndarray, int], bool)]: content of the replay buffer"""
        return self._storage

    @property
    def buffer_size(self) -> int:
        """float: Max capacity of the buffer"""
        return self._maxsize

    def can_sample(self, n_samples: int) -> bool:
        """
        Check if n_samples samples can be sampled
        from the buffer.

        :param n_samples: (int)
        :return: (bool)
        """
        return len(self) >= n_samples

    def is_full(self) -> int:
        """
        Check whether the replay buffer is full or not.

        :return: (bool)
        """
        return len(self) == self.buffer_size

    def add(self, obs_t, action, reward, obs_tp1, done):
        """
        add a new transition to the buffer

        :param obs_t: (Union[np.ndarray, int]) the last observation
        :param action: (Union[np.ndarray, int]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Union[np.ndarray, int]) the current observation
        :param done: (bool) is the episode done
        """
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def extend(self, obs_t, action, reward, obs_tp1, done):
        """
        add a new batch of transitions to the buffer

        :param obs_t: (Union[Tuple[Union[np.ndarray, int]], np.ndarray]) the last batch of observations
        :param action: (Union[Tuple[Union[np.ndarray, int]]], np.ndarray]) the batch of actions
        :param reward: (Union[Tuple[float], np.ndarray]) the batch of the rewards of the transition
        :param obs_tp1: (Union[Tuple[Union[np.ndarray, int]], np.ndarray]) the current batch of observations
        :param done: (Union[Tuple[bool], np.ndarray]) terminal status of the batch

        Note: uses the same names as .add to keep compatibility with named argument passing
                but expects iterables and arrays with more than 1 dimensions
        """
        for data in zip(obs_t, action, reward, obs_tp1, done):
            if self._next_idx >= len(self._storage):
                self._storage.append(data)
            else:
                self._storage[self._next_idx] = data
            self._next_idx = (self._next_idx + 1) % self._maxsize

    @staticmethod
    def _normalize_obs(obs: np.ndarray,
                       env: Optional[VecNormalize] = None) -> np.ndarray:
        """
        Helper for normalizing the observation.
        """
        if env is not None:
            return env.normalize_obs(obs)
        return obs

    @staticmethod
    def _normalize_reward(reward: np.ndarray,
                          env: Optional[VecNormalize] = None) -> np.ndarray:
        """
        Helper for normalizing the reward.
        """
        if env is not None:
            return env.normalize_reward(reward)
        return reward

    def _encode_sample(self, idxes: Union[List[int], np.ndarray], env: Optional[VecNormalize] = None):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return (self._normalize_obs(np.array(obses_t), env),
                np.array(actions),
                self._normalize_reward(np.array(rewards), env),
                self._normalize_obs(np.array(obses_tp1), env),
                np.array(dones))

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None, **_kwargs):
        """
        Sample a batch of experiences.

        :param batch_size: (int) How many transitions to sample.
        :param env: (Optional[VecNormalize]) associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
            - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
            - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
                and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes, env=env)
    ##########################
    # batch_rl mode extensions
    def _reset(self):
        self._storage = []
        self._next_idx = 0

    def record_buffer(self,filename=None) -> dict:
        # saves the experience buffer s.t. it has the fields needed to be loaded as ExpertData
        # it includes the following fields as described in ExpertData class:
        #     The structure of the expert dataset is a dict, saved as an ".npz" archive.
        #     The dictionary contains the keys 'actions', 'episode_returns', 'rewards', 'obs' and 'episode_starts'.
        #     The corresponding values have data concatenated across episode: the first axis is the timestep,
        #     the remaining axes index into the data. In case of images, 'obs' contains the relative path to
        #     the images, to enable space saving from image compression.
        # Note:
        #   - we currently do not support image encoding
        #   - since we didnt necessarily used episodic buffer, we dont know the limits of episode in the buffer
        #     it could be that the buffer wrapped around and we have a mixture of episodes (concatenation of
        #     sub episodes that can look like one longer episode). since this information is anyway not needed
        #     for behavioral cloning (as we load only the observations and actions) we simply generate artificial
        #     arrays for 'episode_returns' and 'episode_starts'.
        #     The consequence of it is that ExpertDataset can't understand the episodic structure so we cant limit
        #     the dataset to maximum number of episodes.

        indxs = np.arange(self.buffer_size) if self.is_full() else np.arange(self._next_idx)
        obses_t, actions, rewards, obses_tp1, dones = self._encode_sample(indxs)
        n_samples = len(obses_t)
        # note : I chose to represent the buffer as length 1 episodes s.t. ExpertData will provide some statistics
        # about the rewards in the buffer (the statistics is related only to episode returns so representing each
        # transition as 1-step episode, we get statistics for the rewards)
        episode_returns = rewards
        episode_starts = np.ones((n_samples,))
        numpy_dict = {
            'actions': actions,
            'obs': obses_t,
            'rewards': rewards,
            'obs_tp1': obses_tp1,
            'dones': dones,
            'episode_returns': episode_returns,
            'episode_starts': episode_starts
        }
        if filename is not None:
            np.savez(filename, **numpy_dict)
        return numpy_dict

    def load_from_csv(self,filename) -> tuple:
        '''
        load_from_csv loads the replay buffer from csv to self._storage
        it expects the csv to have the following columns :
        'action' [str/int/float]- the action that was taken. if not scalar, will be fed as string to evaluate)
        'all_action_probabilities' [str] - the probabilities of the possible actions (what happens in continuous action ?)
        'episode_name' [str] - the name of the episode
        'episode_id' [int] - episode ID. (integer index )
        'reward' [float]- r_(t+1)
        'transition_number' - step index (t)
        'state_feature_i' [float] - element in the state vector (i starts from 0 to obs_dim-1)

        :param filename: name of the csv file
        :return:
        '''
        self._reset()
        try:
            df=pd.read_csv(filename)
        except FileNotFoundError:
            raise FileNotFoundError('Could not find '+filename)
        self._maxsize = max(self._maxsize,len(df))
        episode_returns = []
        episode_starts = [1]
        episode_ids = df['episode_id'].unique()
        state_columns = [col for col in df.columns if col.startswith('state_feature')]
        for e_id_i in tqdm(range(len(episode_ids))):
            # progress_bar.update(e_id)
            e_id = episode_ids[e_id_i]
            df_episode_transitions = df[df['episode_id'] == e_id]
            if len(df_episode_transitions) < 2:
                # we have to have at least 2 rows in each episode for creating a transition
                logger.warn('dropped short episode {0}'.format(e_id))
                continue
            transitions = []
            for (_, current_transition), (_, next_transition) in zip(df_episode_transitions[:-1].iterrows(),
                                                                     df_episode_transitions[1:].iterrows()):
                obs = np.array([current_transition[col] for col in state_columns])
                obs_tp1 = np.array([next_transition[col] for col in state_columns])
                action = int(current_transition['action'])
                reward = current_transition['reward']
                # info is extracted from the csv but currently not saved in the _storage
                info = {'all_action_probabilities':ast.literal_eval(current_transition['all_action_probabilities'])}
                transitions.append({'obs_t':obs,'action':action,'reward':reward,'obs_tp1':obs_tp1,'done':0})
            # set the done flag of the last transition to True
            transitions[-1]['done']=1
            reward_sum = 0.0
            for t in transitions:
                self.add(**t)
                reward_sum += t['reward']
                episode_starts.append(t['done'])
            episode_returns.append(reward_sum)
        episode_starts=episode_starts[:-1]
        return episode_starts,episode_returns

    def _find_next_episode_start(self) -> int:
        idx=self._next_idx
        # assuming the buffer is full
        found=False
        for t in range(self._maxsize):
            (_, _, _, _, done) = self._storage[idx]
            if done:
                found=True
                break
            idx = (idx+1) % self._maxsize
        if not found:
            idx = -100      # return arbitrary negative number
        return idx+1

    def save_to_csv(self,filename,episode_name='episode'):
        '''
        saves self._storage to a csv file
        it expects the csv to have the following columns :
        'action' [str/int/float]- the action that was taken. if not scalar, will be fed as string to evaluate)
        'all_action_probabilities' [str] - the probabilities of the possible actions (what happens in continuous action ?)
        'episode_name' [str] - the name of the episode
        'episode_id' [int] - episode ID. (integer index )
        'reward' [float]- r_(t+1)
        'transition_number' - step index (t)
        'state_feature_i' [float] - element in the state vector (i starts from 0 to obs_dim-1)
        ASSUMPTION: this method is called at the end of episode.
        :param filename: name of the csv file
        :return:
        '''

        # we want to save full episodes to the csv so we'll have to consider 2 scenarios:
        # if not is_full: we simply save the episode from the beginning of the buffer until _next_idx
        # if is_full: we scan from next_idx to the first row with 'done=1' which indicates end of episode
        # and we start record from the following line until _next_idx
        if len(self._storage)==0:
            logger.warn('replay buffer is empty. nothing saved')
            return
        if self.is_full():
            # handle the more complex scenario
            start_indx = self._find_next_episode_start()
        else:
            start_indx = 0
        if start_indx < 0:
            logger.warn('could not find full episode in the buffer. nothing saved')
            return
        num_transitions = self._next_idx
        if start_indx > self._next_idx:
            num_transitions += self._maxsize

        # now we build the dataframe
        obs,_,_,_,_=self._storage[start_indx]
        obs_dim = np.array(obs).size
        idx=start_indx
        ep_id = 0   # episode id generator
        df_cols=['action','all_action_probabilities','episode_name','episode_id','reward',
                                      'transition_number']+['state_feature_{0}'.format(i) for i in range(obs_dim)]
        df = pd.DataFrame([],columns=df_cols)
        act_prob = '[]'     # currently no information about action probabilities. once we get it from the agent,
                            # we'll add it to the buffer in the info field as a dict:
                            # {'all_action_probabilities': str(numpy array of probabilities)}
        ep_name=episode_name+'_{0}'.format(ep_id)
        obs_arr=np.zeros((num_transitions,obs_dim))
        act_arr=[0]*num_transitions
        rew_arr=[0]*num_transitions
        obs_tp1_arr=np.zeros((num_transitions,obs_dim))
        ep_name_arr=[episode_name]*num_transitions
        ep_id_arr=[0]*num_transitions
        t_arr=np.arange(num_transitions)
        for t in tqdm(range(num_transitions)):
            obs_t, action, reward, obs_tp1, done = self._storage[idx]
            # currently we're not saving the action probabilities. if we add it, it will be in the info field.
            # add the transition as a row in the dataset
            # {k:v for k,v in zip(df_cols,[a,aap,epn,eid,r,t]+list(obs))}
            obs_arr[t]=obs_t
            act_arr[t]=action
            rew_arr[t]=reward
            obs_tp1_arr[t]=obs_tp1
            ep_name_arr[t]=ep_name
            ep_id_arr[t]=ep_id
            # the episode name is episode_name+'_{0}'.format(ep_id)
            idx = (idx + 1) % self._maxsize
            if done:
                ep_id+=1
                ep_name = episode_name + '_{0}'.format(ep_id)
        df = pd.DataFrame({k: v for k, v in zip(df_cols, [act_arr, act_prob, ep_name_arr, ep_id_arr, rew_arr, t_arr] + [
            obs_arr[:, c] for c in range(obs_dim)])})
        # df is ready. save to csv
        df.to_csv(filename)
        return




class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha):
        """
        Create Prioritized Replay buffer.

        See Also ReplayBuffer.__init__

        :param size: (int) Max number of transitions to store in the buffer. When the buffer overflows the old memories
            are dropped.
        :param alpha: (float) how much prioritization is used (0 - no prioritization, 1 - full prioritization)
        """
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, obs_t, action, reward, obs_tp1, done):
        """
        add a new transition to the buffer

        :param obs_t: (Any) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Any) the current observation
        :param done: (bool) is the episode done
        """
        idx = self._next_idx
        super().add(obs_t, action, reward, obs_tp1, done)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def extend(self, obs_t, action, reward, obs_tp1, done):
        """
        add a new batch of transitions to the buffer

        :param obs_t: (Union[Tuple[Union[np.ndarray, int]], np.ndarray]) the last batch of observations
        :param action: (Union[Tuple[Union[np.ndarray, int]]], np.ndarray]) the batch of actions
        :param reward: (Union[Tuple[float], np.ndarray]) the batch of the rewards of the transition
        :param obs_tp1: (Union[Tuple[Union[np.ndarray, int]], np.ndarray]) the current batch of observations
        :param done: (Union[Tuple[bool], np.ndarray]) terminal status of the batch

        Note: uses the same names as .add to keep compatibility with named argument passing
            but expects iterables and arrays with more than 1 dimensions
        """
        idx = self._next_idx
        super().extend(obs_t, action, reward, obs_tp1, done)
        while idx != self._next_idx:
            self._it_sum[idx] = self._max_priority ** self._alpha
            self._it_min[idx] = self._max_priority ** self._alpha
            idx = (idx + 1) % self._maxsize

    def _sample_proportional(self, batch_size):
        mass = []
        total = self._it_sum.sum(0, len(self._storage) - 1)
        # TODO(szymon): should we ensure no repeats?
        mass = np.random.random(size=batch_size) * total
        idx = self._it_sum.find_prefixsum_idx(mass)
        return idx

    def sample(self, batch_size: int, beta: float = 0, env: Optional[VecNormalize] = None):
        """
        Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.

        :param batch_size: (int) How many transitions to sample.
        :param beta: (float) To what degree to use importance weights (0 - no corrections, 1 - full correction)
        :param env: (Optional[VecNormalize]) associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
            - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
            - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
                and 0 otherwise.
            - weights: (numpy float) Array of shape (batch_size,) and dtype np.float32 denoting importance weight of
                each sampled transition
            - idxes: (numpy int) Array of shape (batch_size,) and dtype np.int32 idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)
        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)
        p_sample = self._it_sum[idxes] / self._it_sum.sum()
        weights = (p_sample * len(self._storage)) ** (-beta) / max_weight
        encoded_sample = self._encode_sample(idxes, env=env)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        """
        Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        :param idxes: ([int]) List of idxes of sampled transitions
        :param priorities: ([float]) List of updated priorities corresponding to transitions at the sampled idxes
            denoted by variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        assert np.min(priorities) > 0
        assert np.min(idxes) >= 0
        assert np.max(idxes) < len(self.storage)
        self._it_sum[idxes] = priorities ** self._alpha
        self._it_min[idxes] = priorities ** self._alpha

        self._max_priority = max(self._max_priority, np.max(priorities))
