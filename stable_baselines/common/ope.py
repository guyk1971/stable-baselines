# ope.py
# implementation of off policy evaluation tools
# based on the Intel Coach
# Note: the ope manager will be used for evaluation on experience buffer
# it will be called via callback function
import math
import numpy as np
from typing import Dict
from collections import namedtuple
from stable_baselines.common.callbacks import EventCallback
from copy import deepcopy


OpeSharedStats = namedtuple("OpeSharedStats", ['all_reward_model_rewards', 'all_policy_probs',
                                               'all_v_values_reward_model_based', 'all_rewards', 'all_actions',
                                               'all_old_policy_probs', 'new_policy_prob', 'rho_all_dataset'])
OpeEstimation = namedtuple("OpeEstimation", ['ips', 'dm', 'dr', 'seq_dr', 'wis'])

class DoublyRobust(object):

    @staticmethod
    def evaluate(ope_shared_stats: 'OpeSharedStats') -> tuple:
        """
        Run the off-policy evaluator to get a score for the goodness of the new policy, based on the dataset,
        which was collected using other policy(ies).

        Papers:
        https://arxiv.org/abs/1103.4601
        https://arxiv.org/pdf/1612.01205 (some more clearer explanations)

        :return: the evaluation score
        """

        ips = np.mean(ope_shared_stats.rho_all_dataset * ope_shared_stats.all_rewards)
        dm = np.mean(ope_shared_stats.all_v_values_reward_model_based)
        dr = np.mean(ope_shared_stats.rho_all_dataset *
                     (ope_shared_stats.all_rewards - ope_shared_stats.all_reward_model_rewards[
                         range(len(ope_shared_stats.all_actions)), ope_shared_stats.all_actions])) + dm

        return ips, dm, dr

class SequentialDoublyRobust(object):

    @staticmethod
    def evaluate(evaluation_dataset_as_episodes: Dict, discount_factor: float) -> float:
        """
        Run the off-policy evaluator to get a score for the goodness of the new policy, based on the dataset,
        which was collected using other policy(ies).
        When the epsiodes are of changing lengths, this estimator might prove problematic due to its nature of recursion
        of adding rewards up to the end of the episode (horizon). It will probably work best with episodes of fixed
        length.
        Paper: https://arxiv.org/pdf/1511.03722.pdf

        :return: the evaluation score
        """
        episode_starts = np.where(evaluation_dataset_as_episodes['episode_starts'])
        softmax_policy_prob = evaluation_dataset_as_episodes['infos']['softmax_policy_prob']
        all_action_prob = evaluation_dataset_as_episodes['infos']['all_action_probabilities']
        q_values = evaluation_dataset_as_episodes['infos']['q_value']
        v_value_q_model_based = evaluation_dataset_as_episodes['infos']['v_value_q_model_based']
        actions = evaluation_dataset_as_episodes['actions']
        rewards = evaluation_dataset_as_episodes['rewards']
        episode_limits = [i for i in range(len(episode_starts)) if episode_starts[i]] + [len(episode_starts)]

        # Sequential Doubly Robust
        per_episode_seq_dr = []
        for start,end in zip(episode_limits[:-1],episode_limits[1:]):
            episode_indxs = range(start,end)
            episode_seq_dr = 0
            for i in reversed(episode_indxs):
                rho = softmax_policy_prob[i][actions[i]]/all_action_prob[i][actions[i]]
                episode_seq_dr = v_value_q_model_based[i]+\
                                 rho*(rewards[i]+discount_factor*episode_seq_dr-q_values[i][actions[i]])
        # for episode in evaluation_dataset_as_episodes:
        #     episode_seq_dr = 0
        #     for transition in reversed(episode.transitions):
        #         rho = transition.info['softmax_policy_prob'][transition.action] / \
        #               transition.info['all_action_probabilities'][transition.action]
        #         episode_seq_dr = transition.info['v_value_q_model_based'] + rho * (transition.reward + discount_factor
        #                                                                            * episode_seq_dr -
        #                                                                            transition.info['q_value'][
        #                                                                                transition.action])
            per_episode_seq_dr.append(episode_seq_dr)
        seq_dr = np.array(per_episode_seq_dr).mean()
        return seq_dr

class WeightedImportanceSampling(object):
# TODO add PDIS
    @staticmethod
    def evaluate(evaluation_dataset_as_episodes: Dict,discount_factor: float) -> float:
        """
        Run the off-policy evaluator to get a score for the goodness of the new policy, based on the dataset,
        which was collected using other policy(ies).

        References:
        - Sutton, R. S. & Barto, A. G. Reinforcement Learning: An Introduction. Chapter 5.5.
        - https://people.cs.umass.edu/~pthomas/papers/Thomas2015c.pdf
        - http://videolectures.net/deeplearning2017_thomas_safe_rl/

        :return: the evaluation score
        """
        episode_starts = evaluation_dataset_as_episodes['episode_starts']
        softmax_policy_prob = evaluation_dataset_as_episodes['infos']['softmax_policy_prob']
        all_action_prob = evaluation_dataset_as_episodes['infos']['all_action_probabilities']
        actions = evaluation_dataset_as_episodes['actions']
        rewards = evaluation_dataset_as_episodes['rewards']
        episode_limits = [i for i in range(len(episode_starts)) if episode_starts[i]]+[len(episode_starts)]

        # Weighted Importance Sampling
        per_episode_w_i = []
        episode_discounted_rewards=[]
        for start,end in zip(episode_limits[:-1],episode_limits[1:]):
            episode_indxs = range(start,end)
            w_i=1
            episode_discounted_reward=0
            gamma = 1.0
            for i in episode_indxs:
                episode_discounted_reward += gamma * rewards[i]
                gamma *= discount_factor
                w_i *= (softmax_policy_prob[i][actions[i]]/all_action_prob[i][actions[i]])

        # for episode in evaluation_dataset_as_episodes:
        #     w_i = 1
        #     for transition in episode.transitions:
        #         w_i *= transition.info['softmax_policy_prob'][transition.action] / \
        #               transition.info['all_action_probabilities'][transition.action]
            per_episode_w_i.append(w_i)
            episode_discounted_rewards.append(episode_discounted_reward)


        total_w_i_sum_across_episodes = sum(per_episode_w_i)

        wis = 0
        if total_w_i_sum_across_episodes != 0:
            for i in range(len(episode_starts)):
                wis += per_episode_w_i[i] * episode_discounted_rewards[i]
            # for i, episode in enumerate(evaluation_dataset_as_episodes):
            #     if len(episode.transitions) != 0:         # can we have 0 length episodes ???
            #         wis += per_episode_w_i[i] * episode.transitions[0].n_step_discounted_rewards
            wis /= total_w_i_sum_across_episodes

        return wis


class OPEManager(object):
    def __init__(self,evaluation_dataset_as_episodes):
        self.evaluation_dataset_as_episodes= deepcopy(evaluation_dataset_as_episodes)
        self.evaluation_dataset_as_transitions = None
        self.doubly_robust = DoublyRobust()
        self.sequential_doubly_robust = SequentialDoublyRobust()
        self.weighted_importance_sampling = WeightedImportanceSampling()
        self.all_reward_model_rewards = None
        self.all_old_policy_probs = None
        self.all_rewards = None
        self.all_actions = None
        self.is_gathered_static_shared_data = False

        # todo: get the evaluation_dataset_as_episodes as input and create the evaluation_dataset_as_transitions from it
        # todo: who should own the reward_model ? who defines its topology ? --> the agent.
        # in agents that support training on batch we'll add another network
        # for training the reward model and we'll train it in a similar way that DBCQ trains its generative model
        # note that we cant train it here since the OPE doesnt get the training dataset.
        # it means that at the first call to the evaluation, when the reward model is assumed to be already trained,
        # we'll call the _gather_static_shared_data

        # evaluation_dataset_as_episodes is a dict of numpy arrays with the following keys:
        # ['actions','obs','rewards','obs_tp1','dones','episode_returns','episode_starts']
        # do I want to arrange it as set of transitions ?
        # self.evaluation_dataset_as_transitions = [t for e in self.evaluation_dataset_as_episodes
        #                                           for t in e.transitions]
        self.evaluation_dataset_as_transitions = self.evaluation_dataset_as_episodes

    def _prepare_ope_shared_stats(self, batch_size,reward_model,policy_model) -> OpeSharedStats:
        """
        Do the preparations needed for different estimators.
        Some of the calcuations are shared, so we centralize all the work here.

        :param evaluation_dataset_as_transitions: The evaluation dataset in the form of transitions.
        :param batch_size: The batch size to use.
        :param reward_model: A reward model to be used by DR
        :param policy_model: The Q network whose policy we evaluate.
        :return:
        """

        if not self.is_gathered_static_shared_data:
            self._gather_static_shared_stats(batch_size,reward_model)

        # IPS
        all_policy_probs = []
        all_v_values_reward_model_based, all_v_values_q_model_based = [], []

        for i in range(math.ceil(len(self.evaluation_dataset_as_transitions) / batch_size)):
            batch_states = self.evaluation_dataset_as_transitions['obs'][i * batch_size: (i + 1) * batch_size]
            q_values, _, sm_values = policy_model.predict(batch_states,with_prob=True)
            all_policy_probs.append(sm_values)
            all_v_values_reward_model_based.append(np.sum(all_policy_probs[-1] * self.all_reward_model_rewards[i],
                                                          axis=1))
            all_v_values_q_model_based.append(np.sum(all_policy_probs[-1] * q_values, axis=1))
            batch_infos = self.evaluation_dataset_as_transitions['infos'][i * batch_size: (i + 1) * batch_size]
            for j,info in enumerate(batch_infos):
                info.update({
                    'q_value': q_values[j],
                    'softmax_policy_prob': all_policy_probs[-1][j],
                    'v_value_q_model_based': all_v_values_q_model_based[-1][j],
                })

        all_policy_probs = np.concatenate(all_policy_probs, axis=0)
        all_v_values_reward_model_based = np.concatenate(all_v_values_reward_model_based, axis=0)

        # generate model probabilities
        new_policy_prob = all_policy_probs[np.arange(self.all_actions.shape[0]), self.all_actions]
        rho_all_dataset = new_policy_prob / self.all_old_policy_probs

        return OpeSharedStats(self.all_reward_model_rewards, all_policy_probs, all_v_values_reward_model_based,
                              self.all_rewards, self.all_actions, self.all_old_policy_probs, new_policy_prob,
                              rho_all_dataset)

    def _gather_static_shared_stats(self,  batch_size: int, reward_model) -> None:
        # todo: what type is the reward_model ?
        all_reward_model_rewards = []
        all_old_policy_probs = []
        all_rewards = []
        all_actions = []

        # option 1 : process as mini batches
        for i in range(math.ceil(len(self.evaluation_dataset_as_transitions) / batch_size)):
            batch_states = self.evaluation_dataset_as_transitions['obs'][i * batch_size: (i + 1) * batch_size]
            batch_rewards = self.evaluation_dataset_as_transitions['rewards'][i * batch_size: (i + 1) * batch_size]
            batch_actions = self.evaluation_dataset_as_transitions['actions'][i * batch_size: (i + 1) * batch_size]
            batch_infos = self.evaluation_dataset_as_transitions['infos'][i * batch_size: (i + 1) * batch_size]
            batch_action_probs = np.concatenate([i['all_action_probabilities'] for i in batch_infos])
            all_reward_model_rewards.append(reward_model.predict(batch_states))
            all_rewards.append(batch_rewards)
            all_actions.append(batch_actions)
            all_old_policy_probs.append(batch_action_probs[range(len(batch_actions)),batch_actions])

        self.all_reward_model_rewards = np.concatenate(all_reward_model_rewards, axis=0)
        self.all_old_policy_probs = np.concatenate(all_old_policy_probs, axis=0)
        self.all_rewards = np.concatenate(all_rewards, axis=0)
        self.all_actions = np.concatenate(all_actions, axis=0)

        # mark that static shared data was collected and ready to be used
        self.is_gathered_static_shared_data = True

    def evaluate(self, batch_size: int,discount_factor: float, reward_model, policy_model) -> OpeEstimation:
        """
        Run all the OPEs and get estimations of the current policy performance based on the evaluation dataset.

        :param batch_size: Batch size to use for the estimators.
        :param discount_factor: The standard RL discount factor.
        :param reward_model: A reward model to be used by DR
        :param policy_model: The Q network whose its policy we evaluate.


        :return: An OpeEstimation tuple which groups together all the OPE estimations
        """
        ope_shared_stats = self._prepare_ope_shared_stats(batch_size, reward_model, policy_model)

        ips, dm, dr = self.doubly_robust.evaluate(ope_shared_stats)
        seq_dr = self.sequential_doubly_robust.evaluate(self.evaluation_dataset_as_episodes, discount_factor)
        wis = self.weighted_importance_sampling.evaluate(self.evaluation_dataset_as_episodes)

        return OpeEstimation(ips, dm, dr, seq_dr, wis)



class OffPolicyEvalCallback(EventCallback):
    """
    Callback for evaluating an agent on static buffer

    :param eval_env: (Union[gym.Env, VecEnv]) The environment used for initialization
    :param callback_on_new_best: (Optional[BaseCallback]) Callback to trigger
        when there is a new best model according to the `mean_reward`
    :param n_eval_episodes: (int) The number of episodes to test the agent
    :param eval_freq: (int) Evaluate the agent every eval_freq call of the callback.
    :param log_path: (str) Path to a folder where the evaluations (`evaluations.npz`)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: (str) Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: (bool) Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: (bool) Whether to render or not the environment during evaluation
    :param verbose: (int)
    """
    def __init__(self, eval_env: Union[gym.Env, VecEnv],        # todo: replace this with ope manager that was initialized with evaluation dataset
                 callback_on_new_best: Optional[BaseCallback] = None,
                 n_eval_episodes: int = 5,
                 eval_freq: int = 10000,
                 log_path: str = None,
                 best_model_save_path: str = None,
                 deterministic: bool = True,
                 render: bool = False,
                 verbose: int = 1):
        super(EvalCallback, self).__init__(callback_on_new_best, verbose=verbose)
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])

        assert eval_env.num_envs == 1, "You must pass only one environment for evaluation"

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        # Logs will be written in `evaluations.npz`
        if log_path is not None:
            log_path = os.path.join(log_path, 'evaluations')
        self.log_path = log_path
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []

    def _init_callback(self):
        # Does not work in some corner cases, where the wrapper is not the same
        if not type(self.training_env) is type(self.eval_env):
            warnings.warn("Training and eval env are not of the same type"
                          "{} != {}".format(self.training_env, self.eval_env))

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def _on_step(self) -> bool:

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            sync_envs_normalization(self.training_env, self.eval_env)

            episode_rewards, episode_lengths = evaluate_policy(self.model, self.eval_env,
                                                               n_eval_episodes=self.n_eval_episodes,
                                                               render=self.render,
                                                               deterministic=self.deterministic,
                                                               return_episode_rewards=True)

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)
                np.savez(self.log_path, timesteps=self.evaluations_timesteps,
                         results=self.evaluations_results, ep_lengths=self.evaluations_length)

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            # Keep track of the last evaluation, useful for classes that derive from this callback
            self.last_mean_reward = mean_reward

            if self.verbose > 0:
                print("Eval num_timesteps={}, "
                      "episode_reward={:.2f} +/- {:.2f}".format(self.num_timesteps, mean_reward, std_reward))
                print("Episode length: {:.2f} +/- {:.2f}".format(mean_ep_length, std_ep_length))

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, 'best_model'))
                self.best_mean_reward = mean_reward
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()

        return True
