#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
'''
    This script runs rllab or gym environments. To run RLLAB, use the format
    rllab.<env_name> as env name, otherwise gym will be used.
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
# Self imports: algorithm
from baselines.policy.mlp_policy import MlpPolicy
from baselines.policy.cnn_policy import CnnPolicy

from baselines.trpo_mpi import trpo_mpi

def get_env_type(env_id):
    #First load all envs
    _game_envs = defaultdict(set)
    for env in gym.envs.registry.all():
        # TODO: solve this with regexes
        env_type = env.entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)
    # Get env type
    env_type = None
    for g, e in _game_envs.items():
        if env_id in e:
            env_type = g
            break
    return env_type

def train(env, policy, policy_init, n_episodes, horizon, seed, njobs=1, save_weights=0,
          learnable_variance=True, variance_init=1, **alg_args):

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
    elif policy == 'nn':
        hid_size = [100, 50, 25]
        num_hid_layers = 3

    if policy_init == 'xavier':
        policy_initializer = tf.contrib.layers.xavier_initializer()
    elif policy_init == 'zeros':
        policy_initializer = U.normc_initializer(0.0)
    else:
        raise Exception('Unrecognized policy initializer.')

    if policy == 'linear' or policy == 'nn':
        def make_policy(name, ob_space, ac_space):
            return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                             hid_size=hid_size, num_hid_layers=num_hid_layers, gaussian_fixed_var=True, use_bias=False, use_critic=True,
                             hidden_W_init=policy_initializer, output_W_init=policy_initializer, learnable_variance=learnable_variance,
                             variance_initializer=variance_init)
    elif policy == 'cnn':
        def make_policy(name, ob_space, ac_space):
            return CnnPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                         gaussian_fixed_var=True, use_bias=False, use_critic=True,
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

    gym.logger.setLevel(logging.WARNING)

    env = make_env()

    trpo_mpi.learn(env, make_policy, batch_size=n_episodes, task_horizon=horizon,
                max_kl=alg_args['max_kl'], cg_iters=alg_args['cg_iters'], sampler=sampler,
                gamma=alg_args['gamma'], lam=alg_args['lam'], max_iters=alg_args['max_iters'])

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--env', type=str, default='cartpole')
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--horizon', type=int, default=500)
    parser.add_argument('--njobs', type=int, default=-1)
    parser.add_argument('--policy', type=str, default='nn')
    parser.add_argument('--policy_init', type=str, default='xavier')
    parser.add_argument('--file_name', type=str, default='progress')
    parser.add_argument('--logdir', type=str, default='logs')
    parser.add_argument('--max_iters', type=int, default=500)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--lam', type=float, default=1.0)
    parser.add_argument('--center', type=bool, default=False)
    parser.add_argument('--max_kl', type=float, default=0.0)
    parser.add_argument('--cg_iters', type=int, default=10)
    parser.add_argument('--entropy', type=str, default='none')
    parser.add_argument('--experiment_name', type=str, default='none')
    parser.add_argument('--save_weights', action='store_true', default=False, help='Save policy weights.')
    parser.add_argument('--learnable_variance', type=bool, default=False)
    parser.add_argument('--variance_init', type=float, default=-1)
    args = parser.parse_args()
    if args.file_name == 'progress':
        file_name = '%s_batchsize=%s_maxkl=%s_seed=%s_%s' % (args.env.upper(), args.num_episodes, args.max_kl, args.seed, time.time())
    else:
        file_name = args.file_name
    logger.configure(dir=args.logdir, format_strs=['stdout', 'csv', 'tensorboard'], file_name=file_name)
    train(env=args.env,
          policy=args.policy,
          policy_init=args.policy_init,
          n_episodes=args.num_episodes,
          horizon=args.horizon,
          seed=args.seed,
          njobs=args.njobs,
          save_weights=args.save_weights,
          max_iters=args.max_iters,
          max_kl=args.max_kl,
          cg_iters=args.cg_iters,
          gamma=args.gamma,
          lam=args.lam,
          center_return=args.center,
          entropy=args.entropy,
          variance_init=args.variance_init,
          learnable_variance=args.learnable_variance,)

if __name__ == '__main__':
    main()
