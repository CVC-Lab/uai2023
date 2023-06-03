"""
    Generic utils used in the POIS algorithm.
    These are shared between all versions of POIS.
"""

import numpy as np

def add_disc_rew(seg, gamma):
    """
        Discount the reward of the generated batch of trajectories.
    """
    new = np.append(seg['new'], 1)
    rew = seg['rew']

    n_ep = len(seg['ep_rets'])
    n_samp = len(rew)

    seg['ep_disc_ret'] = ep_disc_ret = np.empty(n_ep, 'float32')
    seg['disc_rew'] = disc_rew = np.empty(n_samp, 'float32')

    discounter = 0
    ret = 0.
    i = 0
    for t in range(n_samp):
        disc_rew[t] = rew[t] * gamma ** discounter
        ret += disc_rew[t]

        if new[t + 1]:
            discounter = 0
            ep_disc_ret[i] = ret
            i += 1
            ret = 0.
        else:
            discounter += 1

def cluster_rewards(ep_reward, reward_clustering='none'):
    """
        Cluster the episode return with the provided strategy.
    """
    if reward_clustering == 'none':
        pass
    elif reward_clustering == 'floor':
        ep_reward = np.floor(ep_reward)
    elif reward_clustering == 'ceil':
        ep_reward = np.ceil(ep_reward)
    elif reward_clustering == 'floor10':
        ep_reward = np.floor(ep_reward * 10) / 10
    elif reward_clustering == 'ceil10':
        ep_reward = np.ceil(ep_reward * 10) / 10
    elif reward_clustering == 'floor100':
        ep_reward = np.floor(ep_reward * 100) / 100
    elif reward_clustering == 'ceil100':
        ep_reward = np.ceil(ep_reward * 100) / 100
    return ep_reward
