# compare to: dtt_rl/deployment/calc_power_limit_versions/deploy_stbl_tf.py
import os
import numpy as np
from tqdm import tqdm
import argparse

###################################################################################################
import json
import pickle
from collections import OrderedDict, deque
from abc import abstractmethod,ABC
import cloudpickle
import zipfile
import warnings
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete
import multiprocessing
################################################
# Platform Agnostic Utils
import base64
import io

def json_to_data(json_string, custom_objects=None):
    """
    Turn JSON serialization of class-parameters back into dictionary.

    :param json_string: (str) JSON serialization of the class-parameters
        that should be loaded.
    :param custom_objects: (dict) Dictionary of objects to replace
        upon loading. If a variable is present in this dictionary as a
        key, it will not be deserialized and the corresponding item
        will be used instead. Similar to custom_objects in
        `keras.models.load_model`. Useful when you have an object in
        file that can not be deserialized.
    :return: (dict) Loaded class parameters.
    """
    if custom_objects is not None and not isinstance(custom_objects, dict):
        raise ValueError("custom_objects argument must be a dict or None")

    json_dict = json.loads(json_string)
    # This will be filled with deserialized data
    return_data = {}
    for data_key, data_item in json_dict.items():
        if (data_key == 'policy') or (data_key == 'gen_act_policy'):     # ignore the policy as it will be overwritten
            continue
        if custom_objects is not None and data_key in custom_objects.keys():
            # If item is provided in custom_objects, replace
            # the one from JSON with the one in custom_objects
            return_data[data_key] = custom_objects[data_key]
        elif isinstance(data_item, dict) and ":serialized:" in data_item.keys():
            # If item is dictionary with ":serialized:"
            # key, this means it is serialized with cloudpickle.
            serialization = data_item[":serialized:"]
            # Try-except deserialization in case we run into
            # errors. If so, we can tell bit more information to
            # user.
            try:
                deserialized_object = cloudpickle.loads(
                    base64.b64decode(serialization.encode())
                )
            except pickle.UnpicklingError:
                raise RuntimeError(
                    "Could not deserialize object {}. ".format(data_key) +
                    "Consider using `custom_objects` argument to replace " +
                    "this object."
                )
            return_data[data_key] = deserialized_object
        else:
            # Read as it is
            return_data[data_key] = data_item
    return return_data

def bytes_to_params(serialized_params, param_list):
    """
    Turn serialized parameters (bytes) back into OrderedDictionary.

    :param serialized_params: (byte) Serialized parameters
        with `numpy.savez`.
    :param param_list: (list) List of strings, representing
        the order of parameters in which they should be returned
    :return: (OrderedDict) Dictionary mapping variable name to
        numpy array of the parameters.
    """
    byte_file = io.BytesIO(serialized_params)
    params = np.load(byte_file)
    return_dictionary = OrderedDict()
    # Assign parameters to return_dictionary
    # in the order specified by param_list
    for param_name in param_list:
        return_dictionary[param_name] = params[param_name]
    return return_dictionary
###############################################
# Tensorflow 1.15 utils
from typing import Set
import logging

ALREADY_INITIALIZED = set()  # type: Set[tf.Variable]

def suppress_tensorflow_warnings():
    #----------- Supress Tensorflow version warnings----------------------
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

    # https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=Warning)

    tf.get_logger().setLevel('INFO')
    tf.autograph.set_verbosity(0)

    tf.get_logger().setLevel(logging.ERROR)
    #-----------------------------------------------------------------------
    return

def observation_input(ob_space, batch_size=None, name='Ob', scale=False):
    """
    Build observation input with encoding depending on the observation space type

    When using Box ob_space, the input will be normalized between [1, 0] on the bounds ob_space.low and ob_space.high.

    :param ob_space: (Gym Space) The observation space
    :param batch_size: (int) batch size for input
                       (default is None, so that resulting input placeholder can take tensors with any batch size)
    :param name: (str) tensorflow variable name for input placeholder
    :param scale: (bool) whether or not to scale the input
    :return: (TensorFlow Tensor, TensorFlow Tensor) input_placeholder, processed_input_tensor
    """
    if isinstance(ob_space, Discrete):
        observation_ph = tf.placeholder(shape=(batch_size,), dtype=tf.int32, name=name)
        processed_observations = tf.cast(tf.one_hot(observation_ph, ob_space.n), tf.float32)
        return observation_ph, processed_observations

    elif isinstance(ob_space, Box):
        observation_ph = tf.placeholder(shape=(batch_size,) + ob_space.shape, dtype=ob_space.dtype, name=name)
        processed_observations = tf.cast(observation_ph, tf.float32)
        # rescale to [1, 0] if the bounds are defined
        if (scale and
           not np.any(np.isinf(ob_space.low)) and not np.any(np.isinf(ob_space.high)) and
           np.any((ob_space.high - ob_space.low) != 0)):

            # equivalent to processed_observations / 255.0 when bounds are set to [255, 0]
            processed_observations = ((processed_observations - ob_space.low) / (ob_space.high - ob_space.low))
        return observation_ph, processed_observations

    elif isinstance(ob_space, MultiBinary):
        observation_ph = tf.placeholder(shape=(batch_size, ob_space.n), dtype=tf.int32, name=name)
        processed_observations = tf.cast(observation_ph, tf.float32)
        return observation_ph, processed_observations

    elif isinstance(ob_space, MultiDiscrete):
        observation_ph = tf.placeholder(shape=(batch_size, len(ob_space.nvec)), dtype=tf.int32, name=name)
        processed_observations = tf.concat([
            tf.cast(tf.one_hot(input_split, ob_space.nvec[i]), tf.float32) for i, input_split
            in enumerate(tf.split(observation_ph, len(ob_space.nvec), axis=-1))
        ], axis=-1)
        return observation_ph, processed_observations

    else:
        raise NotImplementedError("Error: the model does not support input space of type {}".format(
            type(ob_space).__name__))


def tf_initialize(sess=None):
    """
    Initialize all the uninitialized variables in the global scope.

    :param sess: (TensorFlow Session)
    """
    if sess is None:
        sess = tf.get_default_session()
    new_variables = set(tf.global_variables()) - ALREADY_INITIALIZED
    sess.run(tf.variables_initializer(new_variables))
    ALREADY_INITIALIZED.update(new_variables)

def tf_make_session(num_cpu=None, make_default=False, graph=None):
    """
    Returns a session that will use <num_cpu> CPU's only

    :param num_cpu: (int) number of CPUs to use for TensorFlow
    :param make_default: (bool) if this should return an InteractiveSession or a normal Session
    :param graph: (TensorFlow Graph) the graph of the session
    :return: (TensorFlow session)
    """
    if num_cpu is None:
        num_cpu = int(os.getenv('RCALL_NUM_CPU', multiprocessing.cpu_count()))
    tf_config = tf.ConfigProto(
        allow_soft_placement=True,
        inter_op_parallelism_threads=num_cpu,
        intra_op_parallelism_threads=num_cpu)
    # Prevent tensorflow from taking all the gpu memory
    tf_config.gpu_options.allow_growth = True
    if make_default:
        return tf.InteractiveSession(config=tf_config, graph=graph)
    else:
        return tf.Session(config=tf_config, graph=graph)

def tf_get_trainable_vars(name):
    """
    returns the trainable variables

    :param name: (str) the scope
    :return: ([TensorFlow Variable])
    """
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

###############################################
#region Policies
class BasePolicy(ABC):
    """
    The base policy object

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batches to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param scale: (bool) whether or not to scale the input
    :param obs_phs: (TensorFlow Tensor, TensorFlow Tensor) a tuple containing an override for observation placeholder
        and the processed observation placeholder respectively
    :param add_action_ph: (bool) whether or not to create an action placeholder
    """

    recurrent = False

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, scale=False,
                 obs_phs=None, add_action_ph=False):
        self.n_env = n_env
        self.n_steps = n_steps
        self.n_batch = n_batch
        with tf.variable_scope("input", reuse=False):
            if obs_phs is None:
                self._obs_ph, self._processed_obs = observation_input(ob_space, n_batch, scale=scale)
            else:
                self._obs_ph, self._processed_obs = obs_phs

            self._action_ph = None
            if add_action_ph:
                self._action_ph = tf.placeholder(dtype=ac_space.dtype, shape=(n_batch,) + ac_space.shape,
                                                 name="action_ph")
        self.sess = sess
        self.reuse = reuse
        self.ob_space = ob_space
        self.ac_space = ac_space

    @property
    def is_discrete(self):
        """bool: is action space discrete."""
        return isinstance(self.ac_space, Discrete)

    @property
    def initial_state(self):
        """
        The initial state of the policy. For feedforward policies, None. For a recurrent policy,
        a NumPy array of shape (self.n_env, ) + state_shape.
        """
        assert not self.recurrent, "When using recurrent policies, you must overwrite `initial_state()` method"
        return None

    @property
    def obs_ph(self):
        """tf.Tensor: placeholder for observations, shape (self.n_batch, ) + self.ob_space.shape."""
        return self._obs_ph

    @property
    def processed_obs(self):
        """tf.Tensor: processed observations, shape (self.n_batch, ) + self.ob_space.shape.

        The form of processing depends on the type of the observation space, and the parameters
        whether scale is passed to the constructor; see observation_input for more information."""
        return self._processed_obs

    @property
    def action_ph(self):
        """tf.Tensor: placeholder for actions, shape (self.n_batch, ) + self.ac_space.shape."""
        return self._action_ph

    @staticmethod
    def _kwargs_check(feature_extraction, kwargs):
        """
        Ensure that the user is not passing wrong keywords
        when using policy_kwargs.

        :param feature_extraction: (str)
        :param kwargs: (dict)
        """
        # When using policy_kwargs parameter on model creation,
        # all keywords arguments must be consumed by the policy constructor except
        # the ones for the cnn_extractor network (cf nature_cnn()), where the keywords arguments
        # are not passed explicitly (using **kwargs to forward the arguments)
        # that's why there should be not kwargs left when using the mlp_extractor
        # (in that case the keywords arguments are passed explicitly)
        if feature_extraction == 'mlp' and len(kwargs) > 0:
            raise ValueError("Unknown keywords for policy: {}".format(kwargs))

    @abstractmethod
    def step(self, obs, state=None, mask=None):
        """
        Returns the policy for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :return: ([float], [float], [float], [float]) actions, values, states, neglogp
        """
        raise NotImplementedError

    @abstractmethod
    def proba_step(self, obs, state=None, mask=None):
        """
        Returns the action probability for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :return: ([float]) the action probability
        """
        raise NotImplementedError


class DQNFeedForwardPolicy(BasePolicy):
    """
    Policy object that implements a DQN policy, using a feed forward neural network.

    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, layers=None,
                 cnn_extractor=None, feature_extraction="cnn",
                 obs_phs=None, layer_norm=False, dueling=True, act_fun=tf.nn.relu, **kwargs):

        super(DQNFeedForwardPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
                                                   scale=False, obs_phs=obs_phs)
        self.n_actions = ac_space.n
        self.value_fn = None
        self.q_values = None
        self.dueling = dueling

        if layers is None:
            layers = [64, 64]
        # IMPORTANT: Make sure the variable scope is identical to as defined in the agent
        with tf.variable_scope("model", reuse=reuse):
            with tf.variable_scope("action_value"):
                if feature_extraction == "cnn":
                    extracted_features = cnn_extractor(self.processed_obs, **kwargs)
                    action_out = extracted_features
                else:
                    extracted_features = tf.layers.flatten(self.processed_obs)
                    action_out = extracted_features
                    for layer_size in layers:
                        action_out = tf_layers.fully_connected(action_out, num_outputs=layer_size, activation_fn=None)
                        if layer_norm:
                            action_out = tf_layers.layer_norm(action_out, center=True, scale=True)
                        action_out = act_fun(action_out)

                action_scores = tf_layers.fully_connected(action_out, num_outputs=self.n_actions, activation_fn=None)

            if self.dueling:
                with tf.variable_scope("state_value"):
                    state_out = extracted_features
                    for layer_size in layers:
                        state_out = tf_layers.fully_connected(state_out, num_outputs=layer_size, activation_fn=None)
                        if layer_norm:
                            state_out = tf_layers.layer_norm(state_out, center=True, scale=True)
                        state_out = act_fun(state_out)
                    state_score = tf_layers.fully_connected(state_out, num_outputs=1, activation_fn=None)
                action_scores_mean = tf.reduce_mean(action_scores, axis=1)
                action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, axis=1)
                q_out = state_score + action_scores_centered
            else:
                q_out = action_scores

        self.q_values = q_out
        self._setup_init()

    def _setup_init(self):
        """
        Set up action probability
        """
        with tf.variable_scope("output", reuse=True):
            assert self.q_values is not None
            self.policy_proba = tf.nn.softmax(self.q_values)


    def step(self, obs, state=None, mask=None, deterministic=True):
        '''
        step method is used to derive actions to do step in the environment *in evaluation mode*
        Note that as opposed to the act_f function created in the agent, is for training.
        During training we might want to support epsilon greedy with eps decay. this is what the act_f does.
        with probability eps it chooses uniformly random , with probability 1-eps it chooses argmax (q_values)

        This function is deterministic by default (taking argmax(q_values)) but also supports stochastic mode
        in which it takes np.random.choice(n_actions, p=actions_proba)
        where actions_proba is softmax(q_values)

        :param obs:
        :param state:
        :param mask:
        :param deterministic:
        :return:
        '''
        q_values, actions_proba = self.sess.run([self.q_values, self.policy_proba], {self.obs_ph: obs})
        if deterministic:
            actions = np.argmax(q_values, axis=1)
        else:
            # Unefficient sampling
            # TODO: replace the loop
            # maybe with Gumbel-max trick ? (http://amid.fish/humble-gumbel)
            actions = np.zeros((len(obs),), dtype=np.int64)
            for action_idx in range(len(obs)):
                actions[action_idx] = np.random.choice(self.n_actions, p=actions_proba[action_idx])

        return actions, q_values, None

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})


class QRDQNFeedForwardPolicy(BasePolicy):
    """
    Policy object that implements a QRDQN policy, using a feed forward neural network.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_atoms: (int) The number of atoms (quantiles) in the distribution approximation
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) The size of the Neural network for the policy (if None, default to [64, 64])
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param obs_phs: (TensorFlow Tensor, TensorFlow Tensor) a tuple containing an override for observation placeholder
        and the processed observation placeholder respectively
    :param layer_norm: (bool) enable layer normalisation
    :param dueling: (bool) if true double the output MLP to compute a baseline for action scores
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, layers=None,
                 cnn_extractor=None, feature_extraction="cnn",
                 obs_phs=None, layer_norm=False, dueling=False, act_fun=tf.nn.relu, n_atoms=50, **kwargs):

        super(QRDQNFeedForwardPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
                                                     scale=False, obs_phs=obs_phs)
        self.n_actions = ac_space.n
        self.n_atoms = n_atoms
        self.value_fn = None
        self.q_values = None
        self.dueling = dueling      # currently not supported by QRDQN
        assert not dueling, "Dueling is not aupported with QRDQN"

        if layers is None:
            layers = [64, 64]
        # IMPORTANT: Make sure the variable scope is identical to as defined in the agent
        with tf.variable_scope("model", reuse=reuse):
            with tf.variable_scope("action_value"):
                if feature_extraction == "cnn":
                    extracted_features = cnn_extractor(self.processed_obs, **kwargs)
                    action_out = extracted_features
                else:
                    extracted_features = tf.layers.flatten(self.processed_obs)
                    action_out = extracted_features
                    for layer_size in layers:
                        action_out = tf_layers.fully_connected(action_out, num_outputs=layer_size, activation_fn=None)
                        if layer_norm:
                            action_out = tf_layers.layer_norm(action_out, center=True, scale=True)
                        action_out = act_fun(action_out)

                action_scores_flat = tf_layers.fully_connected(action_out, num_outputs=self.n_actions*self.n_atoms,
                                                          activation_fn=None)
                action_scores = tf.reshape(action_scores_flat,shape=[-1,self.n_actions,self.n_atoms])
            # todo: add support in dueling. not sure how to handle the state value with quantiles
            if self.dueling:
                with tf.variable_scope("state_value"):
                    state_out = extracted_features
                    for layer_size in layers:
                        state_out = tf_layers.fully_connected(state_out, num_outputs=layer_size, activation_fn=None)
                        if layer_norm:
                            state_out = tf_layers.layer_norm(state_out, center=True, scale=True)
                        state_out = act_fun(state_out)
                    state_score = tf_layers.fully_connected(state_out, num_outputs=n_atoms, activation_fn=None)
                action_scores = tf.reduce_mean(action_scores, axis=-1)
                action_scores_mean = tf.reduce_mean(action_scores, axis=1)
                action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, axis=1)
                q_out = state_score + action_scores_centered
            else:
                q_out = tf.identity(action_scores,name='quantiles')

        self.q_values = q_out
        self._setup_init()

    def _setup_init(self):
        """
        Set up action probability
        """
        with tf.variable_scope("output", reuse=True):
            assert self.q_values is not None
            self.policy_proba = tf.nn.softmax(tf.reduce_mean(self.q_values, axis=-1))


    def step(self, obs, state=None, mask=None, deterministic=True):
        '''
        step method is used to derive actions to do step in the environment *in evaluation mode*
        Note that as opposed to the act_f function created in the agent, is for training.
        During training we might want to support epsilon greedy with eps decay. this is what the act_f does.
        with probability eps it chooses uniformly random , with probability 1-eps it chooses argmax (q_values)

        This function is deterministic by default (taking argmax(q_values)) but also supports stochastic mode
        in which it takes np.random.choice(n_actions, p=actions_proba)
        where actions_proba is softmax(q_values)

        :param obs:
        :param state:
        :param mask:
        :param deterministic:
        :return:
        '''
        q_values, actions_proba = self.sess.run([self.q_values, self.policy_proba], {self.obs_ph: obs})
        q_values = np.mean(q_values,axis=-1)
        if deterministic:  # i.e. we pick action deterministically - using q_values
            actions = np.argmax(q_values, axis=1)
        else:  # if stochastic, use action_proba
            n_actions = q_values.shape[-1]
            # Inefficient sampling
            # maybe with Gumbel-max trick ? (http://amid.fish/humble-gumbel)
            actions = np.zeros((len(q_values),), dtype=np.int32)
            for action_idx in range(len(q_values)):
                actions[action_idx] = np.random.choice(n_actions, p=actions_proba[action_idx])
        return actions, q_values, None

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

#####################
# DQN Policies
class DQNCnnPolicy(DQNFeedForwardPolicy):
    """
    Policy object that implements DQN policy, using a CNN (the nature CNN)
    """
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch,
                 reuse=False, obs_phs=None, dueling=True, **_kwargs):
        super(DQNCnnPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                        feature_extraction="cnn", obs_phs=obs_phs, dueling=dueling,
                                        layer_norm=False, **_kwargs)


class DQNLnCnnPolicy(DQNFeedForwardPolicy):
    """
    Policy object that implements DQN policy, using a CNN (the nature CNN), with layer normalisation
    """
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch,
                 reuse=False, obs_phs=None, dueling=True, **_kwargs):
        super(DQNLnCnnPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                          feature_extraction="cnn", obs_phs=obs_phs, dueling=dueling,
                                          layer_norm=True, **_kwargs)


class DQNMlpPolicy(DQNFeedForwardPolicy):
    """
    Policy object that implements DQN policy, using a MLP (2 layers of 64)
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch,
                 reuse=False, obs_phs=None, dueling=True, **_kwargs):
        super(DQNMlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                        feature_extraction="mlp", obs_phs=obs_phs, dueling=dueling,
                                        layer_norm=False, **_kwargs)


class DQNLnMlpPolicy(DQNFeedForwardPolicy):
    """
    Policy object that implements DQN policy, using a MLP (2 layers of 64), with layer normalisation
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch,
                 reuse=False, obs_phs=None, dueling=True, **_kwargs):
        super(DQNLnMlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                          feature_extraction="mlp", obs_phs=obs_phs,
                                          layer_norm=True, dueling=dueling, **_kwargs)

#####################
# DBCQ Policies
class DBCQCnnPolicy(DQNFeedForwardPolicy):
    """
    Policy object that implements DBCQ policy, using a CNN (the nature CNN)
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch,
                 reuse=False, obs_phs=None, dueling=False, **_kwargs):
        super(DBCQCnnPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                        feature_extraction="cnn", obs_phs=obs_phs, dueling=dueling,
                                        layer_norm=False, **_kwargs)


class DBCQLnCnnPolicy(DQNFeedForwardPolicy):
    """
    Policy object that implements DBCQ policy, using a CNN (the nature CNN), with layer normalisation
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch,
                 reuse=False, obs_phs=None, dueling=False, **_kwargs):
        super(DBCQLnCnnPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                          feature_extraction="cnn", obs_phs=obs_phs, dueling=dueling,
                                          layer_norm=True, **_kwargs)


class DBCQMlpPolicy(DQNFeedForwardPolicy):
    """
    Policy object that implements DQN policy, using a MLP (2 layers of 64)
    """
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch,
                 reuse=False, obs_phs=None, dueling=False, **_kwargs):
        super(DBCQMlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                        feature_extraction="mlp", obs_phs=obs_phs, dueling=dueling,
                                        layer_norm=False, **_kwargs)


class DBCQLnMlpPolicy(DQNFeedForwardPolicy):
    """
    Policy object that implements DBCQ policy, using a MLP (2 layers of 64), with layer normalisation
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch,
                 reuse=False, obs_phs=None, dueling=False, **_kwargs):
        super(DBCQLnMlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                          feature_extraction="mlp", obs_phs=obs_phs,
                                          layer_norm=True, dueling=dueling, **_kwargs)

#####################
# QRDQN Policies

class QRDQNCnnPolicy(QRDQNFeedForwardPolicy):
    """
    Policy object that implements QRDQN policy, using a CNN (the nature CNN)
    """
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch,
                 reuse=False, obs_phs=None, dueling=False, n_atoms=50, **_kwargs):
        super(QRDQNCnnPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                        feature_extraction="cnn", obs_phs=obs_phs, dueling=dueling,n_atoms=n_atoms,
                                        layer_norm=False, **_kwargs)


class QRDQNLnCnnPolicy(QRDQNFeedForwardPolicy):
    """
    Policy object that implements DQN policy, using a CNN (the nature CNN), with layer normalisation
    """
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch,
                 reuse=False, obs_phs=None, dueling=False, n_atoms=50,**_kwargs):
        super(QRDQNLnCnnPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                          feature_extraction="cnn", obs_phs=obs_phs, dueling=dueling,
                                          layer_norm=True, n_atoms=n_atoms,**_kwargs)


class QRDQNMlpPolicy(QRDQNFeedForwardPolicy):
    """
    Policy object that implements QRDQN policy, using a MLP (2 layers of 64)
    """
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch,
                 reuse=False, obs_phs=None, dueling=False, n_atoms=50, **_kwargs):
        super(QRDQNMlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                        feature_extraction="mlp", obs_phs=obs_phs, dueling=dueling,
                                        layer_norm=False, n_atoms=n_atoms, **_kwargs)


class QRDQNLnMlpPolicy(QRDQNFeedForwardPolicy):
    """
    Policy object that implements DQN policy, using a MLP (2 layers of 64), with layer normalisation
    """
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch,
                 reuse=False, obs_phs=None, dueling=False, n_atoms=50, **_kwargs):
        super(QRDQNLnMlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                          feature_extraction="mlp", obs_phs=obs_phs,
                                          layer_norm=True, dueling=dueling, n_atoms=n_atoms,**_kwargs)


# dict of policies that are supported for deployment
POLICIES={'DQNMlpPolicy':DQNMlpPolicy,
          'DQNLnMlpPolicy':DQNLnMlpPolicy,
          'DQNCnnPolicy':DQNCnnPolicy,
          'DQNLnCnnPolicy':DQNLnCnnPolicy,
          'DBCQMlpPolicy': DBCQMlpPolicy,
          'DBCQLnMlpPolicy': DBCQLnMlpPolicy,
          'DBCQCnnPolicy': DBCQCnnPolicy,
          'DBCQLnCnnPolicy': DBCQLnCnnPolicy,
          'QRDQNMlpPolicy': QRDQNMlpPolicy,
          'QRDQNLnMlpPolicy': QRDQNLnMlpPolicy,
          'QRDQNCnnPolicy': QRDQNCnnPolicy,
          'QRDQNLnCnnPolicy': QRDQNLnCnnPolicy,
          }
#endregion

###############################################
#region models
# implementation of the deployment models. currently using tensorflow

class RLPolicyDeploy:
    def __init__(self,policy, env, verbose=0, _init_setup_model=False,policy_kwargs=None):
        self.policy = policy
        self.env = env
        self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs
        self.observation_space = None
        self.action_space = None
        self.graph = None
        self.sess = None
        self.params = None
        self._param_load_ops = None
        self.n_cpu_tf_sess = None
        self.verbose = verbose
        if env is not None:
            self.observation_space = env.observation_space
            self.action_space = env.action_space
            self.n_envs=1

        self.step_model = None
        self.proba_step = None
        self.params = None

        return

    def set_env(self, env):
        """
        Checks the validity of the environment, and if it is coherent, set it as the current environment.

        :param env: (Gym Environment) The environment for learning a policy
        """
        if env is None and self.env is None:
            if self.verbose >= 1:
                print("Loading a model without an environment, "
                      "this model cannot be trained until it has a valid environment.")
            return
        elif env is None:
            raise ValueError("Error: trying to replace the current environment with None")

        # sanity checking the environment
        assert self.observation_space == env.observation_space, \
            "Error: the environment passed must have at least the same observation space as the model was trained on."
        assert self.action_space == env.action_space, \
            "Error: the environment passed must have at least the same action space as the model was trained on."
        self.env = env


    @staticmethod
    def _load_from_file_cloudpickle(load_path):
        """Legacy code for loading older models stored with cloudpickle

        :param load_path: (str or file-like) where from to load the file
        :return: (dict, OrderedDict) Class parameters and model parameters
        """
        if isinstance(load_path, str):
            if not os.path.exists(load_path):
                if os.path.exists(load_path + ".pkl"):
                    load_path += ".pkl"
                else:
                    raise ValueError("Error: the file {} could not be found".format(load_path))

            with open(load_path, "rb") as file_:
                data, params = cloudpickle.load(file_)
        else:
            # Here load_path is a file-like object, not a path
            data, params = cloudpickle.load(load_path)

        return data, params

    @staticmethod
    def _load_from_file(load_path, load_data=True, custom_objects=None):
        """Load model data from a .zip archive

        :param load_path: (str or file-like) Where to load model from
        :param load_data: (bool) Whether we should load and return data
            (class parameters). Mainly used by `load_parameters` to
            only load model parameters (weights).
        :param custom_objects: (dict) Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            `keras.models.load_model`. Useful when you have an object in
            file that can not be deserialized.
        :return: (dict, OrderedDict) Class parameters and model parameters
        """
        # Check if file exists if load_path is
        # a string
        if isinstance(load_path, str):
            if not os.path.exists(load_path):
                if os.path.exists(load_path + ".zip"):
                    load_path += ".zip"
                else:
                    raise ValueError("Error: the file {} could not be found".format(load_path))

        # Open the zip archive and load data.
        try:
            with zipfile.ZipFile(load_path, "r") as file_:
                namelist = file_.namelist()
                # If data or parameters is not in the
                # zip archive, assume they were stored
                # as None (_save_to_file allows this).
                data = None
                params = None
                if "data" in namelist and load_data:
                    # Load class parameters and convert to string
                    # (Required for json library in Python 3.5)
                    json_data = file_.read("data").decode()
                    data = json_to_data(json_data, custom_objects=custom_objects)

                if "parameters" in namelist:
                    # Load parameter list and and parameters
                    parameter_list_json = file_.read("parameter_list").decode()
                    parameter_list = json.loads(parameter_list_json)
                    serialized_params = file_.read("parameters")
                    params = bytes_to_params(
                        serialized_params, parameter_list
                    )
        except zipfile.BadZipFile:
            # load_path wasn't a zip file. Possibly a cloudpickle
            # file. Show a warning and fall back to loading cloudpickle.
            warnings.warn("It appears you are loading from a file with old format. " +
                          "Older cloudpickle format has been replaced with zip-archived " +
                          "models. Consider saving the model with new format.",
                          DeprecationWarning)
            # Attempt loading with the cloudpickle format.
            # If load_path is file-like, seek back to beginning of file
            if not isinstance(load_path, str):
                load_path.seek(0)
            data, params = RLPolicyDeploy._load_from_file_cloudpickle(load_path)

        return data, params

    def _setup_load_operations(self):
        """
        Create tensorflow operations for loading model parameters
        """
        # Assume tensorflow graphs are static -> check
        # that we only call this function once
        if self._param_load_ops is not None:
            raise RuntimeError("Parameter load operations have already been created")
        # For each loadable parameter, create appropiate
        # placeholder and an assign op, and store them to
        # self.load_param_ops as dict of variable.name -> (placeholder, assign)
        loadable_parameters = self.get_parameter_list()
        # Use OrderedDict to store order for backwards compatibility with
        # list-based params
        self._param_load_ops = OrderedDict()
        with self.graph.as_default():
            for param in loadable_parameters:
                placeholder = tf.placeholder(dtype=param.dtype, shape=param.shape)
                # param.name is unique (tensorflow variables have unique names)
                self._param_load_ops[param.name] = (placeholder, param.assign(placeholder))


    def load_parameters(self, load_path_or_dict, exact_match=True):
        """
        Load model parameters from a file or a dictionary

        Dictionary keys should be tensorflow variable names, which can be obtained
        with ``get_parameters`` function. If ``exact_match`` is True, dictionary
        should contain keys for all model's parameters, otherwise RunTimeError
        is raised. If False, only variables included in the dictionary will be updated.

        This does not load agent's hyper-parameters.

        .. warning::
            This function does not update trainer/optimizer variables (e.g. momentum).
            As such training after using this function may lead to less-than-optimal results.

        :param load_path_or_dict: (str or file-like or dict) Save parameter location
            or dict of parameters as variable.name -> ndarrays to be loaded.
        :param exact_match: (bool) If True, expects load dictionary to contain keys for
            all variables in the model. If False, loads parameters only for variables
            mentioned in the dictionary. Defaults to True.
        """
        # Make sure we have assign ops
        if self._param_load_ops is None:
            self._setup_load_operations()

        if isinstance(load_path_or_dict, dict):
            # Assume `load_path_or_dict` is dict of variable.name -> ndarrays we want to load
            params = load_path_or_dict
        elif isinstance(load_path_or_dict, list):
            warnings.warn("Loading model parameters from a list. This has been replaced " +
                          "with parameter dictionaries with variable names and parameters. " +
                          "If you are loading from a file, consider re-saving the file.",
                          DeprecationWarning)
            # Assume `load_path_or_dict` is list of ndarrays.
            # Create param dictionary assuming the parameters are in same order
            # as `get_parameter_list` returns them.
            params = dict()
            for i, param_name in enumerate(self._param_load_ops.keys()):
                params[param_name] = load_path_or_dict[i]
        else:
            # Assume a filepath or file-like.
            # Use existing deserializer to load the parameters.
            # We only need the parameters part of the file, so
            # only load that part.
            _, params = RLPolicyDeploy._load_from_file(load_path_or_dict, load_data=False)
            params = dict(params)
        feed_dict = {}
        param_update_ops = []
        # Keep track of not-updated variables
        not_updated_variables = set(self._param_load_ops.keys())
        for param_name, param_value in params.items():
            load_op = self._param_load_ops.get(param_name)
            if load_op:
                placeholder, assign_op = load_op
                feed_dict[placeholder] = param_value
                # Create list of tf.assign operations for sess.run
                param_update_ops.append(assign_op)
                # Keep track which variables are updated
                not_updated_variables.remove(param_name)

        # Check that we updated all parameters if exact_match=True
        if exact_match and len(not_updated_variables) > 0:
            raise RuntimeError("Load dictionary did not contain all variables. " +
                               "Missing variables: {}".format(", ".join(not_updated_variables)))

        self.sess.run(param_update_ops, feed_dict=feed_dict)


    @classmethod
    def load(cls, load_path, env=None, custom_objects=None):
        """
        Load the model from file

        :param load_path: (str or file-like) the saved parameter location
        :param env: (Gym Environment) the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model)
        :param custom_objects: (dict) Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            `keras.models.load_model`. Useful when you have an object in
            file that can not be deserialized.
        :param kwargs: extra arguments to change the model when loading
        """
        data, params = cls._load_from_file(load_path, custom_objects=custom_objects)
        # extract the highest level scope so to assign the model identical name (assuming the lower levels are
        # constructed exactly as they were constructed in the agent):
        # example : the loaded parameters can have: 'deepq/model/action_value/fully_connected/weights:0'
        # the setup_model method will create '<scope>/model/action_value/fully_connected/weights:0' we need to provide
        # it the scope
        scope = list(params.keys())[0].split('/')[0]
        policy_name = data.get('policy_name','DQNMlpPolicy')
        policy = POLICIES[policy_name]

        model = cls(policy=policy, env=None, _init_setup_model=False)
        model.__dict__.update(data)
        model.set_env(env)
        model.setup_model(scope)
        model.load_parameters(params)
        return model

    # @abstractmethod
    # def get_parameter_list(self):
    #     """
    #     Get tensorflow Variables of model's parameters
    #
    #     This includes all variables necessary for continuing training (saving / loading).
    #
    #     :return: (list) List of tensorflow Variables
    #     """
    #     pass

    def get_parameter_list(self):
        return self.params


    # @abstractmethod
    # def setup_model(self):
    #     """
    #     Create all the functions and tensorflow graphs necessary to train the model
    #     """
    #     pass


    def setup_model(self,scope):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # self.set_random_seed(self.seed)
            self.sess = tf_make_session(num_cpu=self.n_cpu_tf_sess, graph=self.graph)
            with tf.variable_scope(scope):
                # sess, ob_space, ac_space, n_env, n_steps, n_batch
                self.step_model = self.policy(self.sess, self.observation_space,self.action_space, 1, 1, None,
                                              **self.policy_kwargs)
            # Initialize the parameters and copy them to the target network.
            tf_initialize(self.sess)
            self.params = tf_get_trainable_vars(scope)


    # @abstractmethod
    # def predict(self, observation, state=None, mask=None, deterministic=False):
    #     pass

    def predict(self, observation, state=None, mask=None, deterministic=True):
        observation = np.array(observation)
        observation = observation.reshape((-1,) + self.observation_space.shape)
        with self.sess.as_default():
            actions, _, _ = self.step_model.step(observation, deterministic=deterministic)
        actions = actions[0]
        return actions


# Do we need DQN Deploy ? maybe we can use one class for all the policies ? after all, the difference is in the topology
# of the network. lets try one model.
# class DQNDeploy(OffPolicyDeploy):
#     def __init__(self,policy, env, verbose=0, _init_setup_model=False,policy_kwargs=None):
#         super(DQNDeploy,self).__init__(policy, env, verbose=verbose, _init_setup_model=_init_setup_model,
#                                        policy_kwargs=policy_kwargs)
#
#         self.step_model = None
#         self.proba_step = None
#         self.params = None
#
#
#     def predict(self, observation, state=None, mask=None, deterministic=True):
#         observation = np.array(observation)
#         observation = observation.reshape((-1,) + self.observation_space.shape)
#         with self.sess.as_default():
#             actions, _, _ = self.step_model.step(observation, deterministic=deterministic)
#         actions = actions[0]
#         return actions
#
#     # def setup_model(self):
#     #     assert issubclass(self.policy, DQNFeedForwardPolicy), "Error: the input policy for the DQN model must be " \
#     #                                                "an instance of DQNFeedForwardPolicy."
#     #     self.graph = tf.Graph()
#     #     with self.graph.as_default():
#     #         # self.set_random_seed(self.seed)
#     #         self.sess = tf_make_session(num_cpu=self.n_cpu_tf_sess, graph=self.graph)
#     #         with tf.variable_scope("step_model"):
#     #             # sess, ob_space, ac_space, n_env, n_steps, n_batch
#     #             self.step_model = self.policy(self.sess, self.observation_space,self.action_space, 1, 1, None,
#     #                                           **self.policy_kwargs)
#     #         # Initialize the parameters and copy them to the target network.
#     #         tf_initialize(self.sess)


def load_stbl_model(path_to_model,env):
    model = RLPolicyDeploy.load(path_to_model, env=env)
    def policy_fn(obs):
        action = model.predict(obs,deterministic=True)
        return action
    return policy_fn
#endregion

