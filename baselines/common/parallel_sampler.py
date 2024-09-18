from multiprocessing import Process, Queue, Event
import os
import time
import sys
import numpy as np
from mpi4py import MPI
import tensorflow as tf
import random
import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder

def set_global_seeds(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def traj_segment_function(pi, env, n_episodes, horizon, stochastic):
    '''
    Collects trajectories by running the policy `pi` in the environment `env`.
    '''
    # Initialize state variables
    t = 0
    ac = env.action_space.sample()
    new = True
    ob, info = env.reset()
    # Determine the shape of the observation
    ob_shape = ob.shape if isinstance(ob, np.ndarray) else (1,)
    
    cur_ep_ret = 0
    cur_ep_len = 0
    ep_rets = []
    ep_lens = []

    # Initialize history arrays with the correct shape
    obs = np.zeros([horizon * n_episodes] + list(ob_shape), dtype=np.float32)
    rews = np.zeros(horizon * n_episodes, 'float32')
    vpreds = np.zeros(horizon * n_episodes, 'float32')
    news = np.zeros(horizon * n_episodes, 'int32')
    acs = np.array([ac for _ in range(horizon * n_episodes)])
    prevacs = acs.copy()
    mask = np.ones(horizon * n_episodes, 'float32')

    i = 0  # Number of episodes collected so far
    j = 0  # Steps in current episode
    while True:
        prevac = ac
        ac, vpred = pi.act(ob, stochastic=stochastic)
        # Ensure `ac` and `vpred` are numpy arrays
        ac = ac.numpy() if isinstance(ac, tf.Tensor) else ac
        vpred = vpred.numpy() if isinstance(vpred, tf.Tensor) else vpred

        if i == n_episodes:
            return {
                "ob": obs,
                "rew": rews,
                "vpred": vpreds,
                "new": news,
                "ac": acs,
                "prevac": prevacs,
                "nextvpred": vpred * (1 - new),
                "ep_rets": ep_rets,
                "ep_lens": ep_lens,
                "mask": mask
            }
        if isinstance(ob, tuple):
            ob = ob[0]
        obs[t] = ob
        vpreds[t] = vpred
        news[t] = new
        acs[t] = ac
        prevacs[t] = prevac

        ob, rew, terminated, truncated, _ = env.step(ac)
        new = terminated or truncated
        rews[t] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        j += 1
        if new or j == horizon:
            new = True
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)

            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()

            next_t = (i + 1) * horizon

            mask[t+1:next_t] = 0.
            acs[t+1:next_t] = acs[t]
            obs[t+1:next_t] = obs[t]

            t = next_t - 1
            i += 1
            j = 0
        t += 1

class Worker(Process):
    '''
    A worker is an independent process with its own environment and policy instantiated locally
    after being created.
    '''
    def __init__(self, output, input, event, make_env, make_pi, traj_segment_generator, seed):
        super(Worker, self).__init__()
        self.output = output
        self.input = input
        self.make_env = make_env
        self.make_pi = make_pi
        self.traj_segment_generator = traj_segment_generator
        self.event = event
        self.seed = seed

    def run(self):
        # Set seeds for reproducibility
        tf.random.set_seed(self.seed)
        set_global_seeds(self.seed)

        env = self.make_env()
        workerseed = self.seed + 10000 * MPI.COMM_WORLD.Get_rank()
        set_global_seeds(workerseed)
        env.seed(workerseed)

        # Create policy instance
        pi = self.make_pi('pi%s' % os.getpid(), env.observation_space, env.action_space)

        while True:
            self.event.wait()
            self.event.clear()
            command, weights = self.input.get()
            if command == 'collect':
                pi.set_weights_flat(weights)
                samples = self.traj_segment_generator(pi, env)
                self.output.put((os.getpid(), samples))
            elif command == 'exit':
                env.close()
                break

class ParallelSampler(object):

    def __init__(self, make_pi, make_env, n_episodes, horizon, stochastic, n_workers=-1, seed=0):
        try:
            affinity = len(os.sched_getaffinity(0))
            if n_workers == -1:
                self.n_workers = affinity
            else:
                self.n_workers = min(n_workers, affinity)
        except:
            self.n_workers = max(1, n_workers)

        if seed is None:
            seed = int(time.time())

        self.output_queue = Queue()
        self.input_queues = [Queue() for _ in range(self.n_workers)]
        self.events = [Event() for _ in range(self.n_workers)]

        n_episodes_per_process = n_episodes // self.n_workers
        remainder = n_episodes % self.n_workers

        f = lambda pi, env: traj_segment_function(pi, env, n_episodes_per_process, horizon, stochastic)
        f_rem = lambda pi, env: traj_segment_function(pi, env, n_episodes_per_process + 1, horizon, stochastic)
        fun = [f] * (self.n_workers - remainder) + [f_rem] * remainder
        self.workers = [
            Worker(
                self.output_queue,
                self.input_queues[i],
                self.events[i],
                make_env,
                make_pi,
                fun[i],
                seed + i * 100
            ) for i in range(self.n_workers)
        ]

        for w in self.workers:
            w.start()

    def collect(self, actor_weights):
        for i in range(self.n_workers):
            self.input_queues[i].put(('collect', actor_weights))

        for e in self.events:
            e.set()

        sample_batches = [self.output_queue.get() for _ in range(self.n_workers)]
        sample_batches = sorted(sample_batches, key=lambda x: x[0])
        _, sample_batches = zip(*sample_batches)

        return self._merge_sample_batches(sample_batches)

    def _merge_sample_batches(self, sample_batches):
        '''
        Merges sample batches collected from different workers.
        '''
        np_fields = ['ob', 'rew', 'vpred', 'new', 'ac', 'prevac', 'mask']
        list_fields = ['ep_rets', 'ep_lens']

        new_dict = {field: sample_batches[0][field] for field in np_fields}
        new_dict.update({field: list(sample_batches[0][field]) for field in list_fields})
        new_dict['nextvpred'] = sample_batches[-1]['nextvpred']

        for batch in sample_batches[1:]:
            for f in np_fields:
                new_dict[f] = np.concatenate((new_dict[f], batch[f]))
            for f in list_fields:
                new_dict[f].extend(batch[f])

        return new_dict

    def close(self):
        for i in range(self.n_workers):
            self.input_queues[i].put(('exit', None))

        for e in self.events:
            e.set()

        for w in self.workers:
            w.join()
