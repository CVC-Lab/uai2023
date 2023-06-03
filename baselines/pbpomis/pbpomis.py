#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 18:13:18 2018

@author: matteo
"""
"""References
    PGPE: Sehnke, Frank, et al. "Policy gradients with parameter-based exploration for
        control." International Conference on Artificial Neural Networks. Springer,
        Berlin, Heidelberg, 2008.
"""

import numpy as np
from baselines import logger
import warnings
from contextlib import contextmanager
import time
from baselines.common import colorize
from baselines.pbpomis.memory import Memory
import math

#just to display computation times
@contextmanager
def timed(msg, verbose=True):
    if verbose: print(colorize(msg, color='magenta'))
    tstart = time.time()
    yield
    if verbose: print(colorize("done in %.3f seconds"%(time.time() - tstart), color='magenta'))

#simple sequential sampler
def eval_trajectory(env, pol, gamma, horizon, feature_fun):
    ret = disc_ret = 0
    t = 0
    ob = env.reset()
    done = False
    while not done and t<horizon:
        s = feature_fun(ob) if feature_fun else ob
        a = pol.act(s)
        ob, r, done, _ = env.step(a)
        ret += r
        disc_ret += gamma**t * r
        t+=1
        
    return ret, disc_ret, t

#binary search for offline step size
def line_search_binary(target, memory, alpha, natgrad, 
                max_search_ite=30, rmax=None, delta=0.2):
    rho_init = target.eval_params()
    low = 0.
    high = None
    bound_init = target.eval_bound_multi(memory, rmax, delta)
    rho_opt = rho_init
    i_opt = 0.
    delta_bound_opt = 0.
    epsilon_opt = 0.
    epsilon = 1.
    
    for i in range(max_search_ite):
        rho = rho_init + epsilon * natgrad * alpha
        target.set_params(rho)
        bound = target.eval_bound_multi(memory, rmax, delta)
        delta_bound = bound - bound_init        
        if np.isnan(bound):
            warnings.warn('Got NaN bound value: rolling back!')
        if np.isnan(bound) or delta_bound <= delta_bound_opt:
            high = epsilon
        else:
            low = epsilon
            rho_opt = rho
            delta_bound_opt = delta_bound
            i_opt = i
            epsilon_opt = epsilon

        old_epsilon = epsilon
        if high is None:
            epsilon *= 2
        else:
            epsilon = (low + high) / 2.
        if abs(old_epsilon - epsilon) < 1e-6:
            break
    
    return rho_opt, epsilon_opt, delta_bound_opt, i_opt+1

#parabolic search for offline step size
def line_search_parabola(target, memory, alpha, natgrad, 
                max_search_ite=30, rmax=None, delta=0.2):
    epsilon = 1.
    epsilon_old = 0.
    max_increase=2. 
    delta_bound_tol=1e-4
    delta_bound_old = -np.inf
    bound_init = target.eval_bound_multi(memory, rmax, delta)
    rho_old = rho_init = target.eval_params()

    for i in range(max_search_ite):

        rho = rho_init + epsilon * alpha * natgrad
        target.set_params(rho)

        bound = target.eval_bound_multi(memory, rmax, delta)

        if np.isnan(bound):
            warnings.warn('Got NaN bound value: rolling back!')
            return rho_old, epsilon_old, delta_bound_old, i + 1

        delta_bound = bound - bound_init

        epsilon_old = epsilon
        
        if delta_bound > (1. - 1. / (2 * max_increase)) * epsilon_old:
            epsilon = epsilon_old * max_increase
        else:
            epsilon = epsilon_old ** 2 / (2 * (epsilon_old - delta_bound))
        
        if delta_bound <= delta_bound_old + delta_bound_tol:
            if delta_bound_old < 0.:
                return rho_init, 0., 0., i+1
            else:
                return rho_old, epsilon_old, delta_bound_old, i+1

        delta_bound_old = delta_bound
        rho_old = rho

    return rho_old, epsilon_old, delta_bound_old, i+1

#offline optimizer
def optimize_offline(target, memory, grad_tol=1e-4, bound_tol=1e-4, max_offline_ite=100, 
                     max_search_ite=30,
                     rmax=None, delta=0.2, use_parabola=False, verbose=True):
    improvement = 0.
    rho = target.eval_params()
    
    
    fmtstr = "%6i %10.3g %10.3g %18i %18.3g %18.3g %18.3g"
    titlestr = "%6s %10s %10s %18s %18s %18s %18s"
    if verbose: print(titlestr % ("iter", "epsilon", "step size", "num line search", 
                      "gradient norm", "delta bound ite", "delta bound tot"))
    
    natgrad = None
    
    for i in range(max_offline_ite):
        #Candidate policy
        target.set_params(rho)
        
        #Bound with gradient
        bound, grad = target.eval_bound_and_grad_multi(memory, rmax, delta)
        
        if np.any(np.isnan(grad)):
            warnings.warn('Got NaN gradient! Stopping!')
            return rho, improvement
        if np.isnan(bound):
            warnings.warn('Got NaN bound! Stopping!')
            return rho, improvement     

            
        #Natural gradient
        if target.diagonal: 
            natgrad = grad/(target.eval_fisher() + 1e-24)
        else:
            raise NotImplementedError

        grad_norm = np.sqrt(np.dot(grad, natgrad))
        if grad_norm < grad_tol:
            if verbose: print("stopping - gradient norm < gradient_tol")
            if verbose>1: print(rho)
            return rho, improvement
        
        #Step size search
        alpha = 1. / grad_norm ** 2
        line_search = line_search_parabola if use_parabola else line_search_binary
        rho, epsilon, delta_bound, num_line_search = line_search(target, 
                                                                 memory, 
                                                                 alpha, 
                                                                 natgrad, 
                                                                 max_search_ite, 
                                                                 rmax, 
                                                                 delta)
        target.set_params(rho)
        improvement+=delta_bound
        if verbose: print(fmtstr % (i+1, epsilon, alpha*epsilon, num_line_search, grad_norm, delta_bound, improvement))
        if delta_bound < bound_tol:
            if verbose: print("stopping - delta bound < bound_tol")
            if verbose>1: print(rho)
            return rho, improvement
    
    return rho, improvement


#the learning algorithm itself
def learn(env_maker, pol_maker, sampler,
          gamma, n_episodes, horizon, max_iters, 
          feature_fun=None, 
          iw_norm='rows',
          max_offline_iters=10, 
          max_search_ite=30,
          verbose=True, 
          save_weights=False,
          delta=0.4,
          center_return=False,
          line_search_type='parabola',
          capacity=10,
          log_freq=1):
    
    #initialization
    with timed('P-POMIS\n\nCompiling', verbose):
        if iw_norm == 'sn': iw_norm = 'rows'
        env = env_maker()
        behavioral = pol_maker('behavioral', env.observation_space, env.action_space)
        target = pol_maker('target', env.observation_space, env.action_space)
        target.set_params(behavioral.eval_params())
        memory = Memory(capacity)
        memory.build_policies(template=behavioral)
        batch_size = n_episodes
        target.build_miw_graph(memory, batch_size, iw_norm)
        episodes_so_far = 0
        timesteps_so_far = 0
        window_sum = 0
        window_eps = 0
        tstart = time.time()
        
    #select step strategy
    if line_search_type == 'parabola':
        use_parabola = True
    elif line_search_type == 'binary':
        use_parabola = False
    else:
        raise ValueError('unknown step strategy %s' % line_search_type)
    
    #learning loop
    for it in range(max_iters):
        logger.log('\n********** Iteration %i ************' % it)
        rho = behavioral.eval_params() #Higher-order-policy parameters of behavioral policy
        if verbose>1:
            logger.log('Higher-order parameters: ', rho)
        if save_weights: 
            w_to_save = rho
            
        #sampling (fixed batchsize)
        with timed('Sampling', verbose):
            if sampler:
                seg = sampler.collect(rho)
                lens, rets, disc_rets, actor_params = seg['lens'], seg['rets'], seg['disc_rets'], seg['actor_params']
            else:   
                frozen_pol = behavioral.freeze()
                actor_params, rets, disc_rets, lens = [], [], [], []
                for ep in range(n_episodes):
                    theta = frozen_pol.resample()
                    actor_params.append(theta)
                    ret, disc_ret, ep_len = eval_trajectory(env, frozen_pol, gamma, horizon, feature_fun)
                    rets.append(ret)
                    disc_rets.append(disc_ret)
                    lens.append(ep_len)
        
        #normalize reward
        norm_disc_rets = np.array(disc_rets)
        if center_return:
            norm_disc_rets = norm_disc_rets - np.mean(norm_disc_rets)
        rmax = np.max(abs(norm_disc_rets))
        
        #update memory (current policy is also most recent proposal)
        memory.add_batch(behavioral, actor_params, norm_disc_rets)
                        
        #log data of the episodes collected in this iteration
        episodes_so_far+=n_episodes
        timesteps_so_far+=sum(lens[-n_episodes:])
        window_sum += np.sum(norm_disc_rets)
        window_eps += n_episodes
        if verbose and it > 0 and episodes_so_far % log_freq != 0:
            print('Return:', np.mean(norm_disc_rets))
        with timed('summaries before'):
            logger.log("Performance (plain, undiscounted): ", np.mean(rets))
            logger.record_tabular("Iteration", it)
            logger.record_tabular("InitialBound", target.eval_bound_multi(memory, rmax, delta))
            logger.record_tabular("EpLenMean", np.mean(lens))
            logger.record_tabular("EpRewMean", np.mean(norm_disc_rets))
            logger.record_tabular("WindowRewMean", window_sum / window_eps)
            logger.record_tabular("UndEpRewMean", np.mean(norm_disc_rets))
            logger.record_tabular("EpThisIter", n_episodes)
            logger.record_tabular("EpisodesSoFar", episodes_so_far)
            logger.record_tabular("TimestepsSoFar", timesteps_so_far)
            logger.record_tabular("BatchSize", batch_size)
            logger.record_tabular("TimeElapsed", time.time() - tstart)
            logger.record_tabular("NBehaviorals", len(memory))

        #Optimization
        iter_type = 1
        with timed('offline optimization', verbose):
            print(memory)
            rho, improvement = optimize_offline(target, memory,
                                                max_offline_ite=max_offline_iters,
                                                max_search_ite=max_search_ite,
                                                rmax=rmax,
                                                delta=delta,
                                                use_parabola=use_parabola,
                                                verbose=verbose)
            target.set_params(rho)
            
        #Save data
        if save_weights:
            logger.record_tabular('Weights', str(w_to_save))
        
        with timed('summaries after'):
            unn_iws = target.eval_miws(memory, normalize=False)
            iws = target.eval_miws(memory, normalize=True)
            ess = np.linalg.norm(unn_iws, 1) ** 2 / np.linalg.norm(unn_iws, 2) ** 2
            J, varJ = target.eval_performance_multi(memory)
            renyi_bound = target.eval_renyi_bound(memory)
            bound = target.eval_bound_multi(memory, rmax, delta)
            
            #Data regarding the whole batch
            logger.record_tabular('BatchSize', batch_size)
            logger.record_tabular('IterType', iter_type)
            logger.record_tabular('Bound', bound)
            #Discounted, [centered]
            logger.record_tabular('InitialReturnMean', np.mean(norm_disc_rets))
            logger.record_tabular('InitialReturnMax', np.max(norm_disc_rets))
            logger.record_tabular('InitialReturnMin', np.min(norm_disc_rets))
            logger.record_tabular('InitialReturnStd', np.std(norm_disc_rets))
            logger.record_tabular('InitialReturnMin', np.min(norm_disc_rets))
            #Discounted, uncentered
            logger.record_tabular('UncReturnMean', np.mean(disc_rets))
            logger.record_tabular('UncReturnMax', np.max(disc_rets))
            logger.record_tabular('UncReturnMin', np.min(disc_rets))
            logger.record_tabular('UncReturnStd', np.std(disc_rets))
            logger.record_tabular('UncReturnMin', np.min(disc_rets))
            #Undiscounted, uncentered
            logger.record_tabular('PlainReturnMean', np.mean(rets))
            logger.record_tabular('PlainReturnMax', np.max(rets))
            logger.record_tabular('PlainReturnMin', np.min(rets))
            logger.record_tabular('PlainReturnStd', np.std(rets))
            logger.record_tabular('PlainReturnMin', np.min(rets))
            #Iws
            logger.record_tabular('D2Bound', renyi_bound)
            logger.record_tabular('D2', math.log(renyi_bound))
            logger.record_tabular('ReturnMeanIw', J)
            logger.record_tabular('MaxIWNorm', np.max(iws))
            logger.record_tabular('MinIWNorm', np.min(iws))
            logger.record_tabular('MeanIWNorm', np.mean(iws))
            logger.record_tabular('StdIWNorm', np.std(iws))
            logger.record_tabular('MaxIW', np.max(unn_iws))
            logger.record_tabular('MinIW', np.min(unn_iws))
            logger.record_tabular('MeanIW', np.mean(unn_iws))
            logger.record_tabular('StdIW', np.std(unn_iws))
            logger.record_tabular('ESSClassic', ess)
                    
        if it==0 or episodes_so_far % log_freq == 0:
            logger.dump_tabular()
        if episodes_so_far % log_freq == 0:
            window_sum = 0
            window_eps = 0
        
        #Update behavioral
        behavioral.set_params(target.eval_params())
