"""
    Sampler which uses a single centralized policy over a set of parallel environments.
    This sampler is useful when using GPUs for the policy.

    Used in POIS2.
"""

import time
import numpy as np

def traj_segment_generator(pi, env, n_episodes, horizon, stochastic, gamma):
    """
        Returns a generator of complete rollouts. It need to be fed a vectorized
        environment and a single policy.
    """
    policy_time = 0
    env_time = 0

    # Initialize state variables
    t = 0
    ac = np.array([env.action_space.sample()] * env.num_envs)

    _env_s = time.time()
    ob = env.reset()
    env_time += time.time() - _env_s
    zero_ob = np.zeros(ob.shape)

    current_indexes = np.arange(0, env.num_envs)

    def filter_indexes(idx_vector, t_vector):
        return map(list, zip(*[(i, v, t) for i,(v,t) in enumerate(zip(idx_vector, t_vector)) if v != -1]))

    def has_ended(idx_vector):
        return sum(idx_vector) == -len(idx_vector)

    # Iterate to make yield continuous
    while True:

        _tt = time.time()

        # Initialize history arrays
        obs = np.array([[zero_ob[0] for _t in range(horizon)] for _e in range(n_episodes)])
        rews = np.zeros((n_episodes, horizon), 'float32')
        vpreds = np.zeros((n_episodes, horizon), 'float32')
        news = np.zeros((n_episodes, horizon), 'int32')
        acs = np.array([[ac[0] for _t in range(horizon)] for _e in range(n_episodes)])
        prevacs = acs.copy()
        mask = np.zeros((n_episodes, horizon), 'int32')
        # Initialize indexes and timesteps
        current_indexes = np.arange(0, env.num_envs)
        current_timesteps = np.zeros((env.num_envs), dtype=np.int32)
        # Set to -1 indexes if njobs > num_episodes
        current_indexes[n_episodes:] = -1
        # Indexes log: remember which indexes have been completed
        indexes_log = list(current_indexes)

        while not has_ended(current_indexes):

            # Get the action and save the previous one
            prevac = ac

            _pi_s = time.time()
            ac, vpred = pi.act(stochastic, ob)
            policy_time += time.time() - _pi_s

            # Filter the current indexes
            ci_ob, ci_memory, ct = filter_indexes(current_indexes, current_timesteps)

            # Save the current properties
            obs[ci_memory, ct,:] = ob[ci_ob]
            #vpreds[ci_memory, ct] = np.reshape(np.array(vpred), (-1,))[ci_ob]
            acs[ci_memory, ct] = ac[ci_ob]
            prevacs[ci_memory, ct] = prevac[ci_ob]

            # Take the action
            _env_s = time.time()
            env.step_async(ac)
            ob, rew, done, _ = env.step_wait()
            env_time += time.time() - _env_s

            # Save the reward
            rews[ci_memory, ct] = rew[ci_ob]
            mask[ci_memory, ct] = 1
            news[ci_memory, ct] = np.reshape(np.array(done), (-1, ))[ci_ob]

            # Update the indexes and timesteps
            for i, d in enumerate(done):
                if not d and current_timesteps[i] < (horizon-1):
                    current_timesteps[i] += 1
                elif max(indexes_log) < n_episodes - 1:
                    current_timesteps[i] = 0 # Reset the timestep
                    current_indexes[i] = max(indexes_log) + 1 # Increment the index
                    indexes_log.append(current_indexes[i])
                else:
                    current_indexes[i] = -1 # Disabling

        # Add discounted reward (here is simpler)
        gamma_log = np.log(np.full((horizon), gamma, dtype='float32'))
        gamma_discounter = np.exp(np.cumsum(gamma_log))
        discounted_reward = rews * gamma_discounter

        total_time = time.time() - _tt

        # Reshape to flatten episodes and yield
        yield {'ob': np.reshape(obs, (n_episodes * horizon,)+obs.shape[2:]),
               'rew': np.reshape(rews, (n_episodes * horizon)),
               'vpred': np.reshape(vpreds, (n_episodes * horizon)),
               'ac': np.reshape(acs, (n_episodes * horizon,)+acs.shape[2:]),
               'prevac': np.reshape(prevacs, (n_episodes * horizon,)+prevacs.shape[2:]),
               'nextvpred': [], # FIXME: what is my goal?
               'ep_rets': np.sum(rews * mask, axis=1),
               'ep_lens': np.sum(mask, axis=1),
               'mask': np.reshape(mask, (n_episodes * horizon)),
               'new': np.reshape(news, (n_episodes * horizon)),
               'disc_rew': np.reshape(discounted_reward, (n_episodes * horizon)),
               'ep_disc_ret': np.sum(discounted_reward, axis=1),
               'total_time': total_time,
               'policy_time': policy_time,
               'env_time': env_time}

        # Reset time counters
        policy_time = 0
        env_time = 0
