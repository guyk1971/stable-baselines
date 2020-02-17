"""Discrete Batch Constrained deep Q learning graph

The functions in this file can are used to create the following functions:

======= act ========

    Function to chose an action given an observation

    :param observation: (Any) Observation that can be feed into the output of make_obs_ph
    :param stochastic: (bool) if set to False all the actions are always deterministic (default False)
    :param update_eps_ph: (float) update epsilon a new value, if negative not update happens (default: no update)
    :return: (TensorFlow Tensor) tensor of dtype tf.int64 and shape (BATCH_SIZE,) with an action to be performed for
        every element of the batch.


======= act (in case of parameter noise) ========

    Function to chose an action given an observation

    :param observation: (Any) Observation that can be feed into the output of make_obs_ph
    :param stochastic: (bool) if set to False all the actions are always deterministic (default False)
    :param update_eps_ph: (float) update epsilon a new value, if negative not update happens
        (default: no update)
    :param reset_ph: (bool) reset the perturbed policy by sampling a new perturbation
    :param update_param_noise_threshold_ph: (float) the desired threshold for the difference between
        non-perturbed and perturbed policy
    :param update_param_noise_scale_ph: (bool) whether or not to update the scale of the noise for the next time it is
        re-perturbed
    :return: (TensorFlow Tensor) tensor of dtype tf.int64 and shape (BATCH_SIZE,) with an action to be performed for
        every element of the batch.


======= train =======

    Function that takes a transition (s,a,r,s') and optimizes Bellman equation's error:

        td_error = Q(s,a) - (r + gamma * max_a' Q(s', a'))
        loss = huber_loss[td_error]

    :param obs_t: (Any) a batch of observations
    :param action: (numpy int) actions that were selected upon seeing obs_t. dtype must be int32 and shape must be
        (batch_size,)
    :param reward: (numpy float) immediate reward attained after executing those actions dtype must be float32 and
        shape must be (batch_size,)
    :param obs_tp1: (Any) observations that followed obs_t
    :param done: (numpy bool) 1 if obs_t was the last observation in the episode and 0 otherwise obs_tp1 gets ignored,
        but must be of the valid shape. dtype must be float32 and shape must be (batch_size,)
    :param weight: (numpy float) imporance weights for every element of the batch (gradient is multiplied by the
        importance weight) dtype must be float32 and shape must be (batch_size,)
    :return: (numpy float) td_error: a list of differences between Q(s,a) and the target in Bellman's equation.
        dtype is float32 and shape is (batch_size,)

======= update_target ========

    copy the parameters from optimized Q function to the target Q function.
    In Q learning we actually optimize the following error:

        Q(s,a) - (r + gamma * max_a' Q'(s', a'))

    Where Q' is lagging behind Q to stablize the learning. For example for Atari

    Q' is set to Q once every 10000 updates training steps.

"""
import tensorflow as tf
from gym.spaces import MultiDiscrete
import numpy as np

from stable_baselines.common import tf_util


def scope_vars(scope, trainable_only=False):
    """
    Get variables inside a scope
    The scope can be specified as a string

    :param scope: (str or VariableScope) scope in which the variables reside.
    :param trainable_only: (bool) whether or not to return only the variables that were marked as trainable.
    :return: ([TensorFlow Tensor]) vars: list of variables in `scope`.
    """
    return tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES if trainable_only else tf.GraphKeys.GLOBAL_VARIABLES,
        scope=scope if isinstance(scope, str) else scope.name
    )


def scope_name():
    """
    Returns the name of current scope as a string, e.g. deepq/q_func

    :return: (str) the name of current scope
    """
    return tf.get_variable_scope().name


def absolute_scope_name(relative_scope_name):
    """
    Appends parent scope name to `relative_scope_name`

    :return: (str) the absolute name of the scope
    """
    return scope_name() + "/" + relative_scope_name


def default_param_noise_filter(var):
    """
    check whether or not a variable is perturbable or not

    :param var: (TensorFlow Tensor) the variable
    :return: (bool) can be perturb
    """
    if var not in tf.trainable_variables():
        # We never perturb non-trainable vars.
        return False
    if "fully_connected" in var.name:
        # We perturb fully-connected layers.
        return True

    # The remaining layers are likely conv or layer norm layers, which we do not wish to
    # perturb (in the former case because they only extract features, in the latter case because
    # we use them for normalization purposes). If you change your network, you will likely want
    # to re-consider which layers to perturb and which to keep untouched.
    return False


def build_act(q_func, gen_act_policy, ob_space, ac_space, act_dist_thresh_ph, sess):
    """
    Creates the act function:

    :param q_func: (DQNPolicy) the policy
    :param gen_act_policy: (DQNPolicy) a generative model for the action
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param act_dist_thresh_ph: (TensorFlow Tensor) placeholder for the threshold to mask out unlikely actions
    :param sess: (TensorFlow session) The current TensorFlow session
    :return: (function (TensorFlow Tensor, bool, float): TensorFlow Tensor, (TensorFlow Tensor, TensorFlow Tensor)
        act function to select and action given observation (See the top of the file for details),
        A tuple containing the observation placeholder and the processed observation placeholder respectively.
    """
    # eps = tf.get_variable("eps", (), initializer=tf.constant_initializer(0))

    policy = q_func(sess, ob_space, ac_space, 1, 1, None)
    obs_phs = (policy.obs_ph, policy.processed_obs)

    # selecting actions subject to distance from gen_act_model:
    gen_act_logits = gen_act_policy.q_values
    max_gen_act_logit = tf.reduce_max(gen_act_logits,axis=1,keepdims=True)
    act_llr = tf.math.subtract(gen_act_logits,max_gen_act_logit)
    masked_q_values = tf.where(act_llr > tf.math.log(act_dist_thresh_ph), gen_act_logits,
                         tf.broadcast_to(tf.constant(-np.inf), shape=gen_act_logits.shape))
    deterministic_actions = tf.argmax(masked_q_values, axis=1)

    # batch_size = tf.shape(policy.obs_ph)[0]
    # n_actions = ac_space.nvec if isinstance(ac_space, MultiDiscrete) else ac_space.n
    # random_actions = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=n_actions, dtype=tf.int64)
    # chose_random = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps
    # stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)
    # output_actions = tf.cond(stochastic_ph, lambda: stochastic_actions, lambda: deterministic_actions)

    output_actions = deterministic_actions
    _act = tf_util.function(inputs=[policy.obs_ph,act_dist_thresh_ph],outputs=output_actions,
                            givens={act_dist_thresh_ph: 1e-7})

    def act(obs, act_dist_th=1e-7):
        return _act(obs, act_dist_th)

    return act, obs_phs


def build_train(q_func, gen_act_policy, ob_space, ac_space, optimizer, sess, grad_norm_clipping=None,
                gamma=1.0, double_q=False, scope="dbcq", reuse=None,param_noise=False, full_tensorboard_log=False):
    """
    Creates the train function:

    :param q_func: (DQNPolicy) the policy
    :param gen_act_policy: (DQNPolicy) a generative model for the action
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param reuse: (bool) whether or not to reuse the graph variables
    :param optimizer: (tf.train.Optimizer) optimizer to use for the Q-learning objective.
    :param sess: (TensorFlow session) The current TensorFlow session
    :param grad_norm_clipping: (float) clip gradient norms to this value. If None no clipping is performed.
    :param gamma: (float) discount rate.
    :param double_q: (bool) if true will use Double Q Learning (https://arxiv.org/abs/1509.06461). In general it is a
        good idea to keep it enabled.
    :param scope: (str or VariableScope) optional scope for variable_scope.
    :param reuse: (bool) whether or not the variables should be reused. To be able to reuse the scope must be given.
    :param param_noise: (bool) whether or not to use parameter space noise (https://arxiv.org/abs/1706.01905)
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        WARNING: this logging can take a lot of space quickly

    :return: (tuple)

        act: (function (TensorFlow Tensor, bool, float): TensorFlow Tensor) function to select and action given
            observation. See the top of the file for details.
        train: (function (Any, numpy float, numpy float, Any, numpy bool, numpy float): numpy float)
            optimize the error in Bellman's equation. See the top of the file for details.
        update_target: (function) copy the parameters from optimized Q function to the target Q function.
            See the top of the file for details.
        step_model: (DQNPolicy) Policy for evaluation
    """
    n_actions = ac_space.nvec if isinstance(ac_space, MultiDiscrete) else ac_space.n
    with tf.variable_scope("input", reuse=reuse):
        act_dist_thresh_ph = tf.placeholder(tf.float32, (), name="act_dist_thresh")

    with tf.variable_scope(scope, reuse=reuse):

        if param_noise:
            print('Error: DBCQ doesnt yet support parameter noise. aborting')
            raise NotImplementedError
        else:
            act_f, obs_phs = build_act(q_func, gen_act_policy, ob_space, ac_space, act_dist_thresh_ph, sess)


        # q network evaluation
        with tf.variable_scope("step_model", reuse=True, custom_getter=tf_util.outer_scope_getter("step_model")):
            step_model = q_func(sess, ob_space, ac_space, 1, 1, None, reuse=True, obs_phs=obs_phs)
        q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + "/model")

        # target q network evaluation
        with tf.variable_scope("target_q_func", reuse=False):
            target_policy = q_func(sess, ob_space, ac_space, 1, 1, None, reuse=False)
        target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                               scope=tf.get_variable_scope().name + "/target_q_func")

        # compute estimate of best possible value starting from state at t + 1
        if double_q:
            print('Error: DBCQ doesnt support double q learning. aborting')
            raise NotImplementedError

    with tf.variable_scope("loss", reuse=reuse):
        # set up placeholders
        act_t_ph = tf.placeholder(tf.int32, [None], name="action")
        rew_t_ph = tf.placeholder(tf.float32, [None], name="reward")
        done_mask_ph = tf.placeholder(tf.float32, [None], name="done")
        importance_weights_ph = tf.placeholder(tf.float32, [None], name="weight")

        # q scores for actions which we know were selected in the given state.
        q_t_selected = tf.reduce_sum(step_model.q_values * tf.one_hot(act_t_ph, n_actions), axis=1)

        # compute estimate of best possible value starting from state at t + 1
        if double_q:
            raise NotImplementedError
            # q_tp1_best_using_online_net = tf.argmax(double_q_values, axis=1)
            # q_tp1_best = tf.reduce_sum(target_policy.q_values * tf.one_hot(q_tp1_best_using_online_net, n_actions), axis=1)
        else:
            q_tp1_best = tf.reduce_max(target_policy.q_values, axis=1)

        q_tp1_best_masked = (1.0 - done_mask_ph) * q_tp1_best

        # compute RHS of bellman equation
        q_t_selected_target = rew_t_ph + gamma * q_tp1_best_masked

        # compute the error (potentially clipped)
        td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)
        errors = tf_util.huber_loss(td_error)
        loss = tf.reduce_mean(errors)
        weighted_error = tf.reduce_mean(importance_weights_ph * errors)

        tf.summary.scalar("td_error", tf.reduce_mean(td_error))
        tf.summary.scalar("wgt_error", weighted_error)
        tf.summary.scalar("loss",loss)

        if full_tensorboard_log:
            tf.summary.histogram("td_error", td_error)

        # update_target_fn will be called periodically to copy Q network to target Q network
        update_target_expr = []
        for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                                   sorted(target_q_func_vars, key=lambda v: v.name)):
            update_target_expr.append(var_target.assign(var))
        update_target_expr = tf.group(*update_target_expr)

        # todo: create update operation for the generative model
        # in case we want to learn it while training the agent...
        update_gen_model = None


        # compute optimization op (potentially with gradient clipping)
        gradients = optimizer.compute_gradients(weighted_error, var_list=q_func_vars)
        if grad_norm_clipping is not None:
            for i, (grad, var) in enumerate(gradients):
                if grad is not None:
                    gradients[i] = (tf.clip_by_norm(grad, grad_norm_clipping), var)

    with tf.variable_scope("input_info", reuse=False):
        tf.summary.scalar('rewards', tf.reduce_mean(rew_t_ph))
        tf.summary.scalar('importance_weights', tf.reduce_mean(importance_weights_ph))

        if full_tensorboard_log:
            tf.summary.histogram('rewards', rew_t_ph)
            tf.summary.histogram('importance_weights', importance_weights_ph)
            if tf_util.is_image(obs_phs[0]):
                tf.summary.image('observation', obs_phs[0])
            elif len(obs_phs[0].shape) == 1:
                tf.summary.histogram('observation', obs_phs[0])

    optimize_expr = optimizer.apply_gradients(gradients)

    summary = tf.summary.merge_all()

    # Create callable functions
    train = tf_util.function(
        inputs=[
            obs_phs[0],
            act_t_ph,
            rew_t_ph,
            target_policy.obs_ph,
            # double_obs_ph,        # double q learning is not supported in dbcq
            done_mask_ph,
            importance_weights_ph
        ],
        outputs=[summary, loss],
        updates=[optimize_expr]
    )
    update_target = tf_util.function([], [], updates=[update_target_expr])

    return act_f, train, update_target, step_model, update_gen_model
