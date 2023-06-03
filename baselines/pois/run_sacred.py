#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
'''
    This script runs rllab or gym environments. To run RLLAB, use the format
    rllab.<env_name> as env name, otherwise gym will be used.

    export SACRED_RUNS_DIRECTORY to log sacred to a directory
    export SACRED_SLACK_CONFIG to use a slack plugin
'''
# Common imports
import sys, re, os, time, logging
from collections import defaultdict

# Framework imports
import gym
import tensorflow as tf

# Self imports: utils
from baselines.common import set_global_seeds
from baselines import logger
import baselines.common.tf_util as U
from baselines.common.rllab_utils import Rllab2GymWrapper, rllab_env_from_name
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.parallel_sampler import ParallelSampler
from baselines.common.cmd_util import get_env_type
# Self imports: algorithm
from baselines.policy.mlp_policy import MlpPolicy
from baselines.policy.cnn_policy import CnnPolicy
from baselines.pois import pois

# Sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver, SlackObserver

# Create experiment, assign the name if provided in env variables
if os.environ.get('EXPERIMENT_NAME') is not None:
    ex = Experiment(os.environ.get('EXPERIMENT_NAME'))
else:
    ex = Experiment('POIS')

# Set a File Observer
if os.environ.get('SACRED_RUNS_DIRECTORY') is not None:
    print("Sacred logging at:", os.environ.get('SACRED_RUNS_DIRECTORY'))
    ex.observers.append(FileStorageObserver.create(os.environ.get('SACRED_RUNS_DIRECTORY')))
if os.environ.get('SACRED_SLACK_CONFIG') is not None:
    print("Sacred is using slack.")
    ex.observers.append(SlackObserver.from_config(os.environ.get('SACRED_SLACK_CONFIG')))

@ex.config
def custom_config():
    seed = 0
    env = 'rllab.cartpole'
    num_episodes = 100
    max_iters = 500
    horizon = 500
    iw_method = 'is'
    iw_norm = 'none'
    natural = False
    file_name = 'progress'
    logdir = 'logs'
    bound = 'max-d2'
    delta = 0.99
    njobs = -1
    save_weights = 0
    policy = 'nn'
    policy_init = 'xavier'
    max_offline_iters = 10
    gamma = 1.0
    center = False
    clipping = False
    entropy = 'none'
    reward_clustering = 'none'
    positive_return = False
    experiment_name = None
    # ENTROPY can be of 4 schemes:
    #    - 'none': no entropy bonus
    #    - 'step:<height>:<duration>': step function which is <height> tall for <duration> iterations
    #    - 'lin:<max>:<min>': linearly decreasing function from <max> to <min> over all iterations, clipped to 0 for negatives
    #    - 'exp:<height>:<scale>': exponentially decreasing curve <height> tall, use <scale> to make it "spread" more
    # REWARD_CLUSTERING can be of 4 schemes:
    #    - 'none': do nothing
    #    - 'manual:<N>:<min>:<max>': N classes between min and max
    #    - 'global:<N>': N classes over global min and max (as seen so far)
    #    - 'batch:<N>': N classes over batch min and max (as seen so far)
    # TODO: quantiles discretization?
    # Create the filename
    if file_name == 'progress':
        file_name = '%s_iw=%s_bound=%s_delta=%s_gamma=%s_center=%s_entropy=%s_seed=%s_%s' % (env.upper(), iw_method, bound, delta, gamma, center, entropy, seed, time.time())
    else:
        file_name = file_name

def train(env, policy, policy_init, n_episodes, horizon, seed, njobs=1, save_weights=0, **alg_args):

    if env.startswith('rllab.'):
        # Get env name and class
        env_name = re.match('rllab.(\S+)', env).group(1)
        env_rllab_class = rllab_env_from_name(env_name)
        # Define env maker
        def make_env():
            env_rllab = env_rllab_class()
            _env = Rllab2GymWrapper(env_rllab)
            return _env
        # Used later
        env_type = 'rllab'
    else:
        # Normal gym, get if Atari or not.
        env_type = get_env_type(env)
        assert env_type is not None, "Env not recognized."
        # Define the correct env maker
        if env_type == 'atari':
            # Atari, custom env creation
            def make_env():
                _env = make_atari(env)
                return wrap_deepmind(_env)
        else:
            # Not atari, standard env creation
            def make_env():
                env_rllab = gym.make(env)
                return env_rllab

    if policy == 'linear':
        hid_size = num_hid_layers = 0
        use_bias = False
    elif policy == 'simple-nn':
        hid_size = [16]
        num_hid_layers = 1
        use_bias = True
    elif policy == 'nn':
        hid_size = [100, 50, 25]
        num_hid_layers = 3
        use_bias = True

    if policy_init == 'xavier':
        policy_initializer = tf.contrib.layers.xavier_initializer()
    elif policy_init == 'zeros':
        policy_initializer = U.normc_initializer(0.0)
    elif policy_init == 'small-weights':
        policy_initializer = U.normc_initializer(0.1)
    else:
        raise Exception('Unrecognized policy initializer.')

    if policy == 'linear' or policy == 'nn' or policy == 'simple-nn':
        def make_policy(name, ob_space, ac_space):
            return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                             hid_size=hid_size, num_hid_layers=num_hid_layers, gaussian_fixed_var=True, use_bias=use_bias, use_critic=False,
                             hidden_W_init=policy_initializer, output_W_init=policy_initializer)
    elif policy == 'cnn':
        def make_policy(name, ob_space, ac_space):
            return CnnPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                         gaussian_fixed_var=True, use_bias=False, use_critic=False,
                         hidden_W_init=policy_initializer,
                         output_W_init=policy_initializer)
    else:
        raise Exception('Unrecognized policy type.')

    sampler = ParallelSampler(make_policy, make_env, n_episodes, horizon, True, n_workers=njobs, seed=seed)

    try:
        affinity = len(os.sched_getaffinity(0))
    except:
        affinity = njobs
    sess = U.make_session(affinity)
    sess.__enter__()

    set_global_seeds(seed)

    gym.logger.setLevel(logging.WARN)

    pois.learn(make_env, make_policy, n_episodes=n_episodes, horizon=horizon,
                sampler=sampler, save_weights=save_weights, **alg_args)

    sampler.close()

@ex.automain
def main(seed, env, num_episodes, horizon, iw_method, iw_norm, natural,
            file_name, logdir, bound, delta, njobs, save_weights, policy,
            policy_init, max_offline_iters, gamma, center, clipping, entropy,
            max_iters, positive_return, reward_clustering, _run):

    logger.configure(dir=logdir, format_strs=['stdout', 'csv', 'tensorboard', 'sacred'], file_name=file_name, run=_run)
    train(env=env,
          policy=policy,
          policy_init=policy_init,
          n_episodes=num_episodes,
          horizon=horizon,
          seed=seed,
          njobs=njobs,
          save_weights=save_weights,
          max_iters=max_iters,
          iw_method=iw_method,
          iw_norm=iw_norm,
          use_natural_gradient=natural,
          bound=bound,
          delta=delta,
          gamma=gamma,
          max_offline_iters=max_offline_iters,
          center_return=center,
          clipping=clipping,
          entropy=entropy,
          reward_clustering=reward_clustering)
