"""
This class implements the memory required for multiple importance sampling.
Practically, it is a buffer of trajectories and policies.
"""

import numpy as np
import baselines.common.tf_util as U
from baselines.common import zipsame
import tensorflow as tf

class Memory():

    def __init__(self, capacity=10, batch_size=100, horizon=500, ob_space=None,
                 ac_space=None, strategy='fifo'):
        self.capacity = capacity
        self.batch_size = batch_size
        self.horizon = horizon
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.ob_shape = list(ob_space.shape)
        self.ac_shape = list(ac_space.shape)
        self.strategy = strategy
        # Init the trajectory buffer
        self.trajectory_buffer = {
            'ob': None,
            'ac': None,
            'disc_rew': None,
            'rew': None,
            'new': None,
            'mask': None
        }
        # Init the behavioral policies
        self.policies = []
        self.assigning_ops = []
        self.get_flats = []

    def unflatten_batch_dict(self, batch):
        return {k: np.reshape(batch[k], [1, self.batch_size, self.horizon] + list(np.array(batch[k]).shape[1:])) for k in self.trajectory_buffer.keys()}

    def flatten_batch_dict(self, batch):
        return {k: np.reshape(v, [-1] + list(v.shape[3:])) for k,v in batch.items()}

    def trim_batch(self):
        if self.strategy == 'fifo':
            # Remove from observations
            for k, v in self.trajectory_buffer.items():
                if v is not None and v.shape[0] == self.capacity:
                    # We remove the first one since they are inserted in order
                    self.trajectory_buffer[k] = np.delete(v, 0, axis=0)
            # Use assigning ops, in reverse order
            for i in range(self.capacity-1, -1, -1):
                self.assigning_ops[i]()
        else:
            raise Exception('Trimming strategy not recognized.')

    def add_trajectory_batch(self, batch):
        # First, trim the batch if the capacity is reached
        self.trim_batch()
        # Update with the new batch
        batch = self.unflatten_batch_dict(batch)
        for k, v in self.trajectory_buffer.items():
            if v is None:
                self.trajectory_buffer[k] = batch[k]
            else:
                self.trajectory_buffer[k] = np.concatenate((self.trajectory_buffer[k], batch[k]), axis=0)

    def get_trajectories(self):
        return self.flatten_batch_dict(self.trajectory_buffer)

    def get_current_load(self):
        if self.trajectory_buffer['ob'] is not None:
            return self.trajectory_buffer['ob'].shape[0]
        else:
            return 0

    def get_active_policies_mask(self):
        load = self.get_current_load()
        mask = np.zeros(self.capacity)
        mask[:load] = 1
        return mask

    def build_policies(self, make_policy, target_pi):
        # Build the policies
        for i in range(self.capacity):
            name = 'behavioral_' + str(i+1) + '_policy'
            self.policies.append(make_policy(name, self.ob_space, self.ac_space))
            all_var_list = self.policies[i].get_trainable_variables()
            var_list = [v for v in all_var_list if v.name.split('/')[1].startswith('pol')]
            self.get_flats.append(U.GetFlat(var_list))
        # Build the swapping actions
        for i in range(self.capacity):
            if i == 0:
                previous_pi = target_pi
            else:
                previous_pi = self.policies[i-1]
            op = U.function([], [], updates=[tf.assign(oldv, newv) for (oldv, newv) in zipsame(self.policies[i].get_variables(), previous_pi.get_variables())])
            self.assigning_ops.append(op)

    def print_parameters(self):
        if self.trajectory_buffer['ob'] is not None:
            print('//', self.trajectory_buffer['ob'].shape[0])
        for i in range(self.capacity):
            theta = self.get_flats[i]()
            print(i, theta)
