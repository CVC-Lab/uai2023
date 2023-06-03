'''
    Testing a policy weights on a specified environment by performing a fixed
    number of rollouts.
'''
# Common imports
import sys, re, os, time, logging
from collections import defaultdict
import numpy as np

# Framework imports
import gym
import tensorflow as tf

# Self imports: utils
from baselines.common import set_global_seeds
import baselines.common.tf_util as U
from baselines.common.rllab_utils import Rllab2GymWrapper, rllab_env_from_name
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.parallel_sampler import ParallelSampler
from baselines.common.cmd_util import get_env_type
# Self imports: algorithm
from baselines.policy.mlp_policy import MlpPolicy
from baselines.policy.cnn_policy import CnnPolicy

def create_sampler(env=None, policy='linear', n_episodes=100, horizon=500, njobs=1, seed=42):
    # Create the environment
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

    # Select policy architecture
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
    policy_initializer = U.normc_initializer(0.0)
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
    # Create the sampler
    sampler = ParallelSampler(make_policy, make_env, n_episodes, horizon, True, n_workers=njobs, seed=seed)
    try:
        affinity = len(os.sched_getaffinity(0))
    except:
        affinity = njobs
    sess = U.make_session(affinity)
    sess.__enter__()
    # Set random seed
    set_global_seeds(seed)
    return sampler

def evaluate(policy_weights, **sampler_args):
    sampler = create_sampler(**sampler_args)
    # Collect trajectories
    seg = sampler.collect(policy_weights)
    # Close sampler when we are done
    sampler.close()
    return seg

if __name__ == '__main__':
    # TODO: argparse to make this standalone
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='RNG seed', type=int, default=42)
    parser.add_argument('--env', type=str, default='rllab.cartpole')
    parser.add_argument('--n_episodes', type=int, default=100)
    parser.add_argument('--horizon', type=int, default=500)
    parser.add_argument('--njobs', type=int, default=-1)
    parser.add_argument('--policy', type=str, default='linear')
    args = parser.parse_args()
    evaluate(np.zeros(5), **vars(args))
