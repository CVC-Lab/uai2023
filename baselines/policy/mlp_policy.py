import numpy as np
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype
from baselines.common.mpi_running_mean_std import RunningMeanStd
import pdb


class MlpPolicy(tf.keras.Model):
    """Gaussian policy with critic, based on multi-layer perceptron"""
    recurrent = False

    def __init__(self, name, ob_space, ac_space, hid_size, num_hid_layers, gaussian_fixed_var=True, use_bias=True, use_critic=True,
                 seed=None, learnable_variance=True, variance_initializer=-1):
        super(MlpPolicy, self).__init__(name=name)
        self.pdtype = pdtype = make_pdtype(ac_space)
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.use_critic = use_critic
        self.gaussian_fixed_var = gaussian_fixed_var
        self.learnable_variance = learnable_variance

        # Set seed for reproducibility
        if seed is not None:
            tf.random.set_seed(seed)

        # Running mean and std for observations
        self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        # Define hidden layers
        self.hidden_layers = []
        for i in range(num_hid_layers):
            self.hidden_layers.append(tf.keras.layers.Dense(
                units=hid_size[i],
                activation='tanh',
                use_bias=use_bias,
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                name=f'hidden_{i+1}'
            ))

        # Critic network
        if use_critic:
            self.v_hidden_layers = []
            for i in range(num_hid_layers):
                self.v_hidden_layers.append(tf.keras.layers.Dense(
                    units=hid_size[i],
                    activation='tanh',
                    use_bias=use_bias,
                    kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                    name=f'v_hidden_{i+1}'
                ))
            self.v_pred = tf.keras.layers.Dense(
                units=1,
                activation=None,
                use_bias=True,
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                name='v_pred'
            )

        # Actor network
        self.mean_layer = tf.keras.layers.Dense(
            units=pdtype.param_shape()[0] // 2,
            activation=None,
            use_bias=use_bias,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            name='mean'
        )

        if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
            if learnable_variance:
                # Log standard deviation as a trainable variable
                self.logstd = tf.Variable(
                    initial_value=variance_initializer * np.ones(pdtype.param_shape()[0] // 2, dtype=np.float32),
                    trainable=True,
                    name="logstd"
                )
            else:
                # Log standard deviation as a fixed variable
                self.logstd = tf.constant(
                    value=variance_initializer * np.ones(pdtype.param_shape()[0] // 2, dtype=np.float32),
                    dtype=tf.float32,
                    name="logstd"
                )
        else:
            # If not Gaussian or action space is not continuous, handle accordingly
            self.logstd = None
            self.pdparam_layer = tf.keras.layers.Dense(
                units=pdtype.param_shape()[0],
                activation=None,
                use_bias=use_bias,
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                name='pdparam'
            )

    def call(self, ob, training=False):
        """
        Forward pass for the policy network.

        Params:
            ob: Observation tensor
            training: Boolean indicating if in training mode

        Returns:
            mean: Mean action tensor
            vpred: Value prediction tensor (if critic is used)
        """
        # Normalize observations
        obz = (ob - self.ob_rms.mean) / (self.ob_rms.std + 1e-8)
        obz = tf.clip_by_value(obz, -5.0, 5.0)

        # Critic forward pass
        if self.use_critic:
            v_out = obz
            for layer in self.v_hidden_layers:
                v_out = layer(v_out)
            vpred = self.v_pred(v_out)
            vpred = tf.squeeze(vpred, axis=1)  # Shape: [batch_size]
        else:
            vpred = tf.zeros(tf.shape(ob)[0], dtype=tf.float32)

        # Actor forward pass
        pol_out = ob
        for layer in self.hidden_layers:
            pol_out = layer(pol_out)

        mean = self.mean_layer(pol_out)

        if self.gaussian_fixed_var and isinstance(self.ac_space, gym.spaces.Box):
            if self.learnable_variance:
                # Expand logstd to match batch size
                logstd = tf.expand_dims(self.logstd, 0)  # Shape: [1, action_dim]
                logstd = tf.tile(logstd, [tf.shape(mean)[0], 1])  # Shape: [batch_size, action_dim]
            else:
                logstd = self.logstd  # Shape: [1, action_dim], broadcastable

            pdparam = tf.concat([mean, logstd], axis=1)  # Shape: [batch_size, 2 * action_dim]
        else:
            pdparam = self.pdparam_layer(pol_out)  # Shape: [batch_size, param_shape]

        pd = self.pdtype.pdfromflat(pdparam)

        return mean, pd, vpred

    def act(self, ob, stochastic=True):
        """
        Sample action from the policy.

        Params:
            ob: Observation array
            stochastic: Boolean indicating if action should be stochastic

        Returns:
            ac: Action sampled or mode action
            vpred: Value prediction
        """
        if isinstance(ob, tuple):
            ob = ob[0]  # The first element is the observation array
        ob = ob.astype(np.float32)
        ob_tensor = tf.convert_to_tensor(ob[None, :])  # Shape: [1, obs_dim]
        mean, pd, vpred = self(ob_tensor, training=False)
        if stochastic:
            ac = pd.sample()
        else:
            ac = pd.mode()
        return ac.numpy()[0], vpred.numpy()[0]

    def get_trainable_variables(self):
        """
        Get all trainable variables of the policy and critic.

        Returns:
            List of trainable variables
        """
        return self.trainable_variables

    def set_weights_flat(self, new_weights):
        """
        Set model weights from a flat numpy array.

        Params:
            new_weights: Flattened weights array
        """
        # Split the flat_weights into individual variables
        shapes = [tf.size(var).numpy() for var in self.trainable_variables]
        splits = np.split(new_weights, np.cumsum(shapes)[:-1])
        new_weight_values = [split.reshape(var.shape) for split, var in zip(splits, self.trainable_variables)]
        # Assign the new weights
        for var, new_w in zip(self.trainable_variables, new_weight_values):
            var.assign(new_w)

    def get_weights_flat(self):
        """
        Get model weights as a flat numpy array.

        Returns:
            Flattened weights array
        """
        return np.concatenate([var.numpy().flatten() for var in self.trainable_variables])

    def eval_renyi(self, states, other, order=2):
        """Exponentiated Renyi divergence exp(Renyi(self, other)) for each state

        Params:
            states: Batch of states
            other: Another policy instance
            order: Order α of the divergence

        Returns:
            Numpy array of exponentiated Renyi divergences
        """
        if order < 2:
            raise NotImplementedError('Only order>=2 is currently supported')

        # Get probability distributions for both policies
        _, self_pd, _ = self(states)
        _, other_pd, _ = other(states)

        # Check if the distribution has a renyi method
        if hasattr(self_pd, 'renyi'):
            # Use the renyi method defined in the probability distribution class
            renyi = self_pd.renyi(other_pd, alpha=order)
        else:
            # Fallback to Gaussian distribution calculation
            self_mean, self_logstd = self_pd.mean(), tf.math.log(self_pd.stddev())
            other_mean, other_logstd = other_pd.mean(), tf.math.log(other_pd.stddev())

            # Normalize standard deviations
            to_check = order / tf.exp(self_logstd) + (1 - order) / tf.exp(other_logstd)
            if not tf.reduce_all(to_check > 0):
                raise ValueError('Conditions on standard deviations are not met')

            detSigma = tf.exp(tf.reduce_sum(self_logstd, axis=-1))
            detOtherSigma = tf.exp(tf.reduce_sum(other_logstd, axis=-1))
            mixSigma = order * tf.exp(self_logstd) + (1 - order) * tf.exp(other_logstd)
            detMixSigma = tf.reduce_prod(mixSigma, axis=-1)
            mean_diff = (self_mean - other_mean) / mixSigma
            renyi = (order / 2) * tf.reduce_sum(mean_diff ** 2, axis=-1) - \
                    (1. / (2 * (order - 1))) * (tf.math.log(detMixSigma) - (1 - order) * tf.math.log(detSigma) - order * tf.math.log(detOtherSigma))

        e_renyi = tf.exp(renyi)
        return e_renyi

    def get_param(self):
        """
        Get the flattened parameters (weights).

        Returns:
            Numpy array of flattened parameters
        """
        return self.get_weights_flat()

    def set_param(self, param):
        """
        Set the flattened parameters (weights).

        Params:
            param: Numpy array of flattened parameters
        """
        self.set_weights_flat(param)
