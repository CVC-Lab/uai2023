#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 10:36:50 2019

@author: matteo
"""
import numpy as np

class Memory:
    """
    Storage of behavioral hyperpolicies with their samples.
    Behaving as a FIFO queue by default (implemented by a circular array)
    Currently supports only fixed batch size.
    Builds a graph for each behavioral
    """
    def __init__(self, capacity=10, strategy='fifo'):
        self.capacity = capacity

        self.strategy = strategy
        
        self.hpolicies = [None for _ in range(self.capacity)]
        self.mask = np.zeros(self.capacity)
        self.params = None
        self.rets = None
        self._built = False
        
        self._len = 0
        self._current = -1
    
    def __len__(self):
        return self._len

    def isfull(self):
        return len(self) == self.capacity
    
    def add_batch(self, behavioral, params, rets):
        if not self._built:
            self.build_policies(behavioral)
        if self.strategy == 'fifo':
            self._add_batch_fifo(behavioral, params, rets)
        else:
            raise NotImplementedError('only supports fifo strategy')
        
        assert len(self) <= self.capacity
        
    def build_policies(self, template):
        for k in range(self.capacity):
            name = 'behavioral_' + str(k+1) + '_hyperpolicy'
            self.hpolicies[k] = template.make_another(name)
            
        self._built = True
        
    def _add_batch_fifo(self, behavioral, params, rets):
        #compute next position in the circular array
        succ = (self._current + 1) % self.capacity
        
        #lazy initialization
        if not self:
            #initialize data
            params = np.array(params)
            self.params = np.broadcast_to(params, (self.capacity,) + params.shape).copy()
            rets = np.array(rets)
            self.rets = np.broadcast_to(rets, (self.capacity,) + rets.shape).copy()
        else:
            #expand data
            self.params[succ] = params
            self.rets[succ] = rets  
                
        #update hyperparameters    
        self.hpolicies[succ].set_params(behavioral.eval_params())
        self.mask[succ] = 1
        
        self._len = min(self._len + 1, self.capacity)
        self._current = succ
    
    def __getitem__(self, index):
        if index >= self._len:
            raise IndexError('memory index out of range')
        
        if self.strategy == 'fifo':
            #compute actual index in circular array
            k = index % self.capacity
            return (self.hpolicies[k], self.params[k], self.rets[k])
        else:
            raise NotImplementedError('only supports fifo strategy')
    
    def __repr__(self):
        return repr([pol.scope for pol, _, _ in self])
        
        
if __name__ == '__main__':
    from baselines.policy.weight_hyperpolicy import PeMlpPolicy
    from gym.spaces import Box
    import baselines.common.tf_util as U
    
    sess = U.make_session(1)
    sess.__enter__()

    sspace = Box(low=-1,high=1, shape=(3,))
    aspace = Box(low=-1,high=1, shape=(2,))
    pol = PeMlpPolicy('test', sspace, aspace)
    n = 5
    
    
    mem = Memory(3)
    #mem.build_policies(pol)
    
    for i in range(4):   
        params = np.concatenate([np.expand_dims(pol.resample(), axis=0) 
                                    for _ in range(n)], axis=0)
        rets = i + np.zeros(n)
        mem.add_batch(pol, params, rets)
        for i, (pol, par, ret) in enumerate(mem):
            print('%s: %s %s' % (pol.scope, par[0,0],
                                '*' if i==mem._current else ''))
        print(mem.mask)
        print()