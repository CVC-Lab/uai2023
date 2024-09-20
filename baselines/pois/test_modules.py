import tensorflow as tf
import numpy as np
from baselines.pois import pois
from baselines.policy.mlp_policy import MlpPolicy
from baselines.common.parallel_sampler import ParallelSampler
from baselines.common.rllab_utils import Rllib2GymWrapper
import gymnasium as gym
import pytest
import pdb
tf.config.run_functions_eagerly(True)


def evaluate_policy(env, policy, n_episodes):
    """
    Evaluate the policy for a given number of episodes and return the rewards.
    """
    rewards = []
    for _ in range(n_episodes):
        ob, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            ob = ob.astype(np.float32)
            ob_input = tf.constant(ob[None, :], dtype=tf.float32)
            ac_dist, _ = policy(ob_input)
            ac = ac_dist.mode().numpy()[0]  # Use deterministic actions for evaluation
            ob, rew, done, _, _ = env.step(ac)
            episode_reward += rew
        rewards.append(episode_reward)
    return rewards

def make_test_env():
    env = Rllib2GymWrapper('CartPole-v1')
    return env

def make_test_policy(observation_space, action_space):
    hid_size = num_hid_layers = 0
    policy = MlpPolicy(name='pi', ob_space=observation_space, ac_space=action_space,
                             hid_size=hid_size, num_hid_layers=num_hid_layers, 
                             gaussian_fixed_var=True,
                             use_bias=False, use_critic=False,
                             learnable_variance=False, variance_initializer=None)
    return policy

# Define functions to set and get policy parameters
def set_policy_parameters(model, new_theta):
    idx = 0
    for var in model.trainable_variables:
        var_shape = var.shape
        var_size_a = tf.size(var).numpy()
        var_size_b = np.prod(var_shape)
        var_size = var_size_b
        assert var_size_a == var_size_b
        new_values = new_theta[idx:idx + var_size].reshape(var_shape)
        var.assign(new_values)
        idx += var_size

def get_policy_parameters(model):
    trainable_variables = model.get_trainable_variables()
    if not trainable_variables:
        raise ValueError("The model has no trainable variables.")
    
    flattened_variables = [tf.reshape(v, [-1]) for v in trainable_variables]
    if not flattened_variables:
        raise ValueError("No variables to concatenate after flattening.")
    
    return tf.concat(flattened_variables, axis=0).numpy()

def test_set_policy_parameters():
    # Create a simple model for testing
    env = make_test_env()
    policy = make_test_policy(env.observation_space, env.action_space)
    # make policy.layers trainable
    policy.build((None,env.observation_space.shape[0]))
    # Get initial parameters
    initial_params = get_policy_parameters(policy)
    # Create new parameters
    new_params = np.random.randn(initial_params.shape[0])
    # Set new parameters
    set_policy_parameters(policy, new_params)
    # Get updated parameters
    updated_params = get_policy_parameters(policy)
    # Check if parameters were updated correctly
    np.testing.assert_allclose(new_params, updated_params, rtol=1e-5, atol=1e-5)
    

def test_optimize_offline():
    env = make_test_env()
    policy = make_test_policy(env.observation_space, env.action_space)
    policy.build((None,env.observation_space.shape[0]))
    # Check if policy has trainable variables
    if not policy.trainable_variables:
        raise ValueError("The policy has no trainable variables. Check the policy initialization.")
    
    # Get initial parameters
    initial_params = get_policy_parameters(policy)
    delta = 0.99
    gamma = 1.0
    max_iters = 10
    n_episodes = 5
    horizon = 100
    njobs = 1
    seed = 0
    sampler = ParallelSampler(make_pi=lambda *args: policy, 
                               make_env=lambda: env, n_episodes=n_episodes, 
                               horizon=horizon, stochastic=True, n_workers=njobs)
    # Run a single iteration of the learning process
    pois.learn(make_env=lambda: env, make_policy=lambda *args: policy, bound='max-d2',
          n_episodes=5, horizon=100, delta=delta, gamma=gamma, max_iters=max_iters,
          sampler=sampler, save_weights=0, learnable_variance=False,
          variance_init=-1)
    # Get updated parameters
    updated_params = get_policy_parameters(policy)
    print(initial_params)
    print(updated_params)
    # Check that parameters have changed
    assert not np.allclose(initial_params, updated_params)
    

def test_learn():
    env = make_test_env()
    policy = make_test_policy(env.observation_space, env.action_space)
    initial_performance = evaluate_policy(env, policy, n_episodes=10)
    delta = 0.99
    gamma = 1.0
    max_iters = 10
    n_episodes = 5
    horizon = 100
    njobs = 1
    seed = 0
    sampler = ParallelSampler(make_pi=lambda *args: policy, 
                               make_env=lambda: env, n_episodes=n_episodes, 
                               horizon=horizon, stochastic=True, n_workers=njobs)
    pois.learn(make_env=lambda: env, make_policy=lambda *args: policy, bound='max-d2',
          n_episodes=5, horizon=100, delta=delta, gamma=gamma, max_iters=max_iters,
          sampler=sampler, save_weights=0, learnable_variance=False,
          variance_init=-1)
    final_performance = evaluate_policy(env, policy, n_episodes=10)
    assert np.mean(final_performance) > np.mean(initial_performance)
    

if __name__ == '__main__':
    test_set_policy_parameters()
    test_optimize_offline()
    test_learn()