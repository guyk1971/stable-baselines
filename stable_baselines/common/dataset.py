import numpy as np


class Dataset(object):
    def __init__(self, data_map, shuffle=True):
        """
        Data loader that handles batches and shuffling.
        WARNING: this will alter the given data_map ordering, as dicts are mutable

        :param data_map: (dict) the input data, where every column is a key
        :param shuffle: (bool) Whether to shuffle or not the dataset
            Important: this should be disabled for recurrent policies
        """
        self.data_map = data_map
        self.shuffle = shuffle
        self.n_samples = next(iter(data_map.values())).shape[0]
        self._next_id = 0
        if self.shuffle:
            self.shuffle_dataset()

    def shuffle_dataset(self):
        """
        Shuffles the data_map
        """
        perm = np.arange(self.n_samples)
        np.random.shuffle(perm)

        for key in self.data_map:
            self.data_map[key] = self.data_map[key][perm]

    def next_batch(self, batch_size):
        """
        returns a batch of data of a given size

        :param batch_size: (int) the size of the batch
        :return: (dict) a batch of the input data of size 'batch_size'
        """
        if self._next_id >= self.n_samples:
            self._next_id = 0
            if self.shuffle:
                self.shuffle_dataset()

        cur_id = self._next_id
        cur_batch_size = min(batch_size, self.n_samples - self._next_id)
        self._next_id += cur_batch_size

        data_map = dict()
        for key in self.data_map:
            data_map[key] = self.data_map[key][cur_id:cur_id+cur_batch_size]
        return data_map

    def iterate_once(self, batch_size):
        """
        generator that iterates over the dataset

        :param batch_size: (int) the size of the batch
        :return: (dict) a batch of the input data of size 'batch_size'
        """
        if self.shuffle:
            self.shuffle_dataset()

        while self._next_id <= self.n_samples - batch_size:
            yield self.next_batch(batch_size)
        self._next_id = 0

    def subset(self, num_elements, shuffle=True):
        """
        Return a subset of the current dataset

        :param num_elements: (int) the number of element you wish to have in the subset
        :param shuffle: (bool) Whether to shuffle or not the dataset
        :return: (Dataset) a new subset of the current Dataset object
        """
        data_map = dict()
        for key in self.data_map:
            data_map[key] = self.data_map[key][:num_elements]
        return Dataset(data_map, shuffle)


def iterbatches(arrays, *, num_batches=None, batch_size=None, shuffle=True, include_final_partial_batch=True):
    """
    Iterates over arrays in batches, must provide either num_batches or batch_size, the other must be None.

    :param arrays: (tuple) a tuple of arrays
    :param num_batches: (int) the number of batches, must be None is batch_size is defined
    :param batch_size: (int) the size of the batch, must be None is num_batches is defined
    :param shuffle: (bool) enable auto shuffle
    :param include_final_partial_batch: (bool) add the last batch if not the same size as the batch_size
    :return: (tuples) a tuple of a batch of the arrays
    """
    assert (num_batches is None) != (batch_size is None), 'Provide num_batches or batch_size, but not both'
    arrays = tuple(map(np.asarray, arrays))
    n_samples = arrays[0].shape[0]
    assert all(a.shape[0] == n_samples for a in arrays[1:])
    inds = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(inds)
    sections = np.arange(0, n_samples, batch_size)[1:] if num_batches is None else num_batches
    for batch_inds in np.array_split(inds, sections):
        if include_final_partial_batch or len(batch_inds) == batch_size:
            yield tuple(a[batch_inds] for a in arrays)


############################################
# Experience Dataset for Batch RL mode
############################################
# Based on gail.dataset code
# alternative : look at https://www.tensorflow.org/guide/data_performance
import queue
import time
from multiprocessing import Queue, Process
import cv2  # pytype:disable=import-error
import numpy as np
from joblib import Parallel, delayed
from stable_baselines import logger

class ExperienceDataset(object):
    """
    Dataset for using Batch mode RL.
    Currently supported only for Discrete action space
    The structure of the experience dataset is a dict, saved as an ".npz" archive.
    The dictionary contains the keys 'actions', 'episode_returns', 'rewards', 'obs' and 'episode_starts'.
    The corresponding values have data concatenated across episode: the first axis is the timestep,
    the remaining axes index into the data. In case of images, 'obs' contains the relative path to
    the images, to enable space saving from image compression.

    :param traj_path: (str) The path to trajectory data (.npz file). Mutually exclusive with traj_data.
    :param traj_data: (dict) Trajectory data, in format described above. Mutually exclusive with experience_path.
    :param train_fraction: (float) the train validation split (0 to 1)
        for pre-training using behavior cloning (BC)
    :param batch_size: (int) the minibatch size for behavior cloning
    :param traj_limitation: (int) the number of trajectory to use (if -1, load all)
    :param randomize: (bool) if the dataset should be shuffled
    :param verbose: (int) Verbosity
    :param sequential_preprocessing: (bool) Do not use subprocess to preprocess
        the data (slower but use less memory for the CI)
    """

    def __init__(self, traj_path=None, traj_data=None, train_fraction=0.7, batch_size=64,
                 traj_limitation=-1, randomize=True, verbose=1, sequential_preprocessing=False):
        if traj_data is not None and traj_path is not None:
            raise ValueError("Cannot specify both 'traj_data' and 'expert_path'")
        if traj_data is None and traj_path is None:
            raise ValueError("Must specify one of 'traj_data' or 'expert_path'")
        if traj_data is None:
            traj_data = np.load(traj_path, allow_pickle=True)

        if verbose > 0:
            for key, val in traj_data.items():
                print(key, val.shape)

        # Array of bool where episode_starts[i] = True for each new episode
        episode_starts = traj_data['episode_starts']

        traj_limit_idx = len(traj_data['obs'])

        if traj_limitation > 0:
            n_episodes = 0
            # Retrieve the index corresponding
            # to the traj_limitation trajectory
            for idx, episode_start in enumerate(episode_starts):
                n_episodes += int(episode_start)
                if n_episodes == (traj_limitation + 1):
                    traj_limit_idx = idx - 1

        observations = traj_data['obs'][:traj_limit_idx]
        actions = traj_data['actions'][:traj_limit_idx]
        observations_tp1 = traj_data['obs_tp1'][:traj_limit_idx]
        rewards = traj_data['rewards'][:traj_limit_idx]
        dones = traj_data['dones'][:traj_limit_idx]
        infos = traj_data['infos'][:traj_limit_idx]

        # obs, actions: shape (N * L, ) + S
        # where N = # episodes, L = episode length
        # and S is the environment observation/action space.
        # S = (1, ) for discrete space
        # Flatten to (N * L, prod(S))
        if len(observations.shape) > 2:
            observations = np.reshape(observations, [-1, np.prod(observations.shape[1:])])
        if len(actions.shape) > 2:
            actions = np.reshape(actions, [-1, np.prod(actions.shape[1:])])

        indices = np.random.permutation(len(observations)).astype(np.int64)

        # Train/Validation split when using behavior cloning
        train_indices = indices[:int(train_fraction * len(indices))]
        val_indices = indices[int(train_fraction * len(indices)):]

        assert len(train_indices) > 0, "No sample for the training set"
        # assert len(val_indices) > 0, "No sample for the validation set"

        self.observations = observations
        self.actions = actions
        self.observations_tp1 = observations_tp1
        self.rewards = rewards
        self.dones = dones
        self.infos = infos


        self.returns = traj_data['episode_returns'][:traj_limit_idx]
        self.avg_ret = sum(self.returns) / len(self.returns)
        self.std_ret = np.std(np.array(self.returns))
        self.verbose = verbose

        assert len(self.observations) == len(self.actions), "The number of actions and observations differ " \
                                                            "please check your expert dataset"
        self.num_traj = min(traj_limitation, np.sum(episode_starts))
        self.num_transition = len(self.observations)
        self.randomize = randomize
        self.sequential_preprocessing = sequential_preprocessing

        self.dataloader = None
        self.train_loader = ExperienceDataLoader(train_indices, self.observations, self.actions, self.rewards,
                                                 self.observations_tp1, self.dones,self.infos,
                                                 batch_size,shuffle=self.randomize, start_process=False,
                                                 sequential=sequential_preprocessing)
        self.val_loader = None
        if len(val_indices)>0:
            self.val_loader = ExperienceDataLoader(val_indices, self.observations, self.actions, self.rewards,
                                                   self.observations_tp1, self.dones,self.infos,
                                                   batch_size,shuffle=self.randomize, start_process=False,
                                                   sequential=sequential_preprocessing)

        if self.verbose >= 1:
            self.log_info()

    def init_dataloader(self, batch_size):
        """
        Initialize the dataloader

        :param batch_size: (int)
        """
        indices = np.random.permutation(len(self.observations)).astype(np.int64)
        self.dataloader = ExperienceDataLoader(indices, self.observations, self.actions, self.rewards,
                                               self.observations_tp1, self.dones, self.infos,
                                               batch_size, shuffle=self.randomize, start_process=False,
                                               sequential=self.sequential_preprocessing)

    def __del__(self):
        del self.dataloader, self.train_loader, self.val_loader

    def prepare_pickling(self):
        """
        Exit processes in order to pickle the dataset.
        """
        self.dataloader, self.train_loader, self.val_loader = None, None, None

    def log_info(self):
        """
        Log the information of the dataset.
        """
        logger.log("Total trajectories: {}".format(self.num_traj))
        logger.log("Total transitions: {}".format(self.num_transition))
        logger.log("Average returns: {}".format(self.avg_ret))
        logger.log("Std for returns: {}".format(self.std_ret))

    def get_next_batch(self, split=None):
        """
        Get the batch from the dataset.

        :param split: (str) the type of data split (can be None, 'train', 'val')
        :return: (np.ndarray, np.ndarray) inputs and labels
        """
        dataloader = {
            None: self.dataloader,
            'train': self.train_loader,
            'val': self.val_loader
        }[split]

        if dataloader.process is None:
            dataloader.start_process()
        try:
            return next(dataloader)
        except StopIteration:
            dataloader = iter(dataloader)
            return next(dataloader)

    def plot(self):
        """
        Show histogram plotting of the episode returns
        """
        # Isolate dependency since it is only used for plotting and also since
        # different matplotlib backends have further dependencies themselves.
        import matplotlib.pyplot as plt
        plt.hist(self.returns)
        plt.show()


class ExperienceDataLoader(object):
    """
    A custom dataloader to preprocessing observations (including images)
    and feed them to the network.

    Original code for the dataloader from https://github.com/araffin/robotics-rl-srl
    (MIT licence)
    Authors: Antonin Raffin, René Traoré, Ashley Hill

    :param indices: ([int]) list of transitions indices (the transitions are (obs,act,rew,obs_tp1,done))
    The following defines the transitions:
    :param observations: (np.ndarray) observations or images path
    :param actions: (np.ndarray) actions
    :param obervations_tp1: (np.ndarray) observations of time t+1
    :param rewards: (np.ndarray) rewards at time t
    :param dones: (np.ndarray) dones

    :param batch_size: (int) Number of samples per minibatch
    :param n_workers: (int) number of preprocessing worker (for loading the images)
    :param infinite_loop: (bool) whether to have an iterator that can be reset
    :param max_queue_len: (int) Max number of minibatches that can be preprocessed at the same time
    :param shuffle: (bool) Shuffle the minibatch after each epoch
    :param start_process: (bool) Start the preprocessing process (default: True)
    :param backend: (str) joblib backend (one of 'multiprocessing', 'sequential', 'threading'
        or 'loky' in newest versions)
    :param sequential: (bool) Do not use subprocess to preprocess the data
        (slower but use less memory for the CI)
    :param partial_minibatch: (bool) Allow partial minibatches (minibatches with a number of element
        lesser than the batch_size)
    """

    def __init__(self, indices, observations, actions, rewards, observations_tp1, dones, infos,
                 batch_size, n_workers=1, infinite_loop=True, max_queue_len=1, shuffle=False,
                 start_process=True, backend='threading', sequential=False, partial_minibatch=True):
        super(ExperienceDataLoader, self).__init__()
        self.n_workers = n_workers
        self.infinite_loop = infinite_loop
        self.indices = indices
        self.original_indices = indices.copy()
        self.n_minibatches = len(indices) // batch_size
        # Add a partial minibatch, for instance
        # when there is not enough samples
        if partial_minibatch and len(indices) / batch_size > 0:
            self.n_minibatches += 1
        self.batch_size = batch_size
        # the following defines the transition
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.observations_tp1 = observations_tp1
        self.dones = dones
        self.infos = infos

        self.shuffle = shuffle
        self.queue = Queue(max_queue_len)
        self.process = None
        # self.load_images = isinstance(observations[0], str)
        self.load_images = False        # support in handling images is blocked
        self.backend = backend
        self.sequential = sequential
        self.start_idx = 0
        if start_process:
            self.start_process()

    def start_process(self):
        """Start preprocessing process"""
        # Skip if in sequential mode
        if self.sequential:
            return
        self.process = Process(target=self._run)
        # Make it a deamon, so it will be deleted at the same time
        # of the main process
        self.process.daemon = True
        self.process.start()

    @property
    def _minibatch_indices(self):
        """
        Current minibatch indices given the current pointer
        (start_idx) and the minibatch size
        :return: (np.ndarray) 1D array of indices
        """
        return self.indices[self.start_idx:self.start_idx + self.batch_size]

    def sequential_next(self):
        """
        Sequential version of the pre-processing.
        """
        if self.start_idx > len(self.indices):
            raise StopIteration

        if self.start_idx == 0:
            if self.shuffle:
                # Shuffle indices
                np.random.shuffle(self.indices)

        obs = self.observations[self._minibatch_indices]
        if self.load_images:
            obs = np.concatenate([self._make_batch_element(image_path) for image_path in obs],
                                 axis=0)

        actions = self.actions[self._minibatch_indices]
        rewards = self.rewards[self._minibatch_indices]
        obs_tp1 = self.observations_tp1[self._minibatch_indices]
        dones = self.dones[self._minibatch_indices]
        infos = self.infos[self._minibatch_indices]
        self.start_idx += self.batch_size
        return obs, actions, rewards, obs_tp1, dones, infos


    def _run(self):
        start = True
        with Parallel(n_jobs=self.n_workers, batch_size="auto", backend=self.backend) as parallel:
            while start or self.infinite_loop:
                start = False

                if self.shuffle:
                    np.random.shuffle(self.indices)

                for minibatch_idx in range(self.n_minibatches):

                    self.start_idx = minibatch_idx * self.batch_size

                    obs = self.observations[self._minibatch_indices]
                    if self.load_images:
                        if self.n_workers <= 1:
                            obs = [self._make_batch_element(image_path)
                                   for image_path in obs]

                        else:
                            obs = parallel(delayed(self._make_batch_element)(image_path)
                                           for image_path in obs)

                        obs = np.concatenate(obs, axis=0)

                    actions = self.actions[self._minibatch_indices]
                    rewards = self.rewards[self._minibatch_indices]
                    obses_tp1 = self.observations_tp1[self._minibatch_indices]
                    dones = self.dones[self._minibatch_indices]
                    infos = self.infos[self._minibatch_indices]

                    self.queue.put((obs, actions, rewards, obses_tp1, dones,infos))

                    # Free memory
                    del obs

                self.queue.put(None)

    @classmethod
    def _make_batch_element(cls, image_path):
        """
        Process one element.

        :param image_path: (str) path to an image
        :return: (np.ndarray)
        """
        # cv2.IMREAD_UNCHANGED is needed to load
        # grey and RGBa images
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        # Grey image
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]

        if image is None:
            raise ValueError("Tried to load {}, but it was not found".format(image_path))
        # Convert from BGR to RGB
        if image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.reshape((1,) + image.shape)
        return image

    def __len__(self):
        return self.n_minibatches

    def __iter__(self):
        self.start_idx = 0
        self.indices = self.original_indices.copy()
        return self

    def __next__(self):
        if self.sequential:
            return self.sequential_next()

        if self.process is None:
            raise ValueError("You must call .start_process() before using the dataloader")
        while True:
            try:
                val = self.queue.get_nowait()
                break
            except queue.Empty:
                time.sleep(0.001)
                continue
        if val is None:
            raise StopIteration
        return val

    def __del__(self):
        if self.process is not None:
            self.process.terminate()
