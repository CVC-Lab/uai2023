'''
    Utils to use OpenAI Gymnasium environments wherever possible, fallback to Ray RLlib when necessary
'''

import gymnasium as gym
from baselines.common import set_global_seeds as set_all_seeds
import numpy as np
import ray
from ray import rllib
from ray.tune.registry import register_env
import difflib  # Added for string matching

def load_env(env_name):
    available_envs = list(gym.envs.registry.keys())
    closest_matches = difflib.get_close_matches(env_name, available_envs, n=1, cutoff=0.6)
    if closest_matches:
        closest_env = closest_matches[0]
        try:
            env = gym.make(closest_env)
            print(f"Loaded '{closest_env}' instead.")
            return env
        except gym.error.Error:
            print(f"Found closest match '{closest_env}' but failed to load it.")
    print(f"{env_name} not found in OpenAI Gymnasium.")
    raise Exception(f"Environment {env_name} not found in gym.")

# Wrapper for Ray RLlib and Gymnasium environments
class Rllib2GymWrapper(gym.Env):
    def __init__(self, env_name):
        self.env = load_env(env_name)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def seed(self, seed=0):
        set_all_seeds(seed)

    def render(self, mode='human'):
        return self.env.render()

# Example of how to use the environment loader and Ray RLlib for training
if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)

    # Registering a custom Gym environment with Ray RLlib
    def env_creator(config):
        return Rllib2GymWrapper('CartPole-v1')

    # Register the environment with Ray
    register_env("custom_cartpole", env_creator)

    # Configuring the RL algorithm (PPO in this case)
    config = rllib.algorithms.ppo.PPOConfig().environment(env="custom_cartpole")

    # Build the agent/trainer
    agent = config.build()

    # Train for some iterations
    for i in range(5):
        result = agent.train()
        print(f"Iteration {i}: reward = {result['episode_reward_mean']}")

    ray.shutdown()
