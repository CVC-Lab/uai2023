import numpy as np
import tensorflow as tf
import gymnasium as gym
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
        self.use_critic = use_critic
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
        if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
            self.mean_layer = tf.keras.layers.Dense(
                units=pdtype.param_shape()[0] // 2,
                activation=None,
                use_bias=use_bias,
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                name='mean'
            )
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
        if self.gaussian_fixed_var and isinstance(self.ac_space, gym.spaces.Box):
            mean = self.mean_layer(pol_out)
            if self.learnable_variance:
                    # Expand logstd to match batch size
                    logstd = tf.expand_dims(self.logstd, 0)  # Shape: [1, action_dim]
                    logstd = tf.tile(logstd, [tf.shape(mean)[0], 1])  # Shape: [batch_size, action_dim]
            else:
                logstd = self.logstd  # Shape: [1, action_dim], broadcastable
                logstd = tf.expand_dims(logstd, 0)
                logstd = tf.tile(logstd, [mean.shape[0], 1])  # Shape: [batch_size, action_dim]
            pdparam = tf.concat([mean, logstd], axis=1)
        else:
            pdparam = self.pdparam_layer(pol_out)  # Shape: [batch_size, param_shape]
        
        pd = self.pdtype.pdfromflat(pdparam)

        return pd, vpred

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
        pd, vpred = self(ob_tensor, training=False)
        if stochastic:
            ac = pd.sample()
        else:
            ac = pd.mode()
        return ac.numpy()[0], vpred.numpy()[0]

    def get_trainable_variables(self):
        return self.trainable_weights
       

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
        """Exponentiated Rényi divergence exp(Renyi(self, other)) for each state."""
        if order <= 0 or order == 1:
            raise ValueError('Order must be positive and not equal to 1')
        
        # Extract parameters
        self_logstd = self.logstd  # Shape: [num_actions]
        other_logstd = other.logstd  # Shape: [num_actions]
        self_mean = self.mean_layer(states)  # Shape: [batch_size, num_actions]
        other_mean = other.mean_layer(states)  # Shape: [batch_size, num_actions]
        
        # Convert log standard deviations to variances
        self_var = tf.exp(2 * self_logstd)  # Variance = (std)^2
        other_var = tf.exp(2 * other_logstd)
        
        # Compute mixture covariance matrix (diagonal assumed)
        mixture_var = order * other_var + (1 - order) * self_var  # Shape: [num_actions]
        
        # Ensure positive definiteness
        valid_mask = tf.reduce_all(mixture_var > 0)
        if not valid_mask:
            raise ValueError('Mixture variance must be positive definite.')
        
        # Compute determinant terms (log determinants for numerical stability)
        log_det_self_var = tf.reduce_sum(tf.math.log(self_var))  # ln|Σ_P|
        log_det_other_var = tf.reduce_sum(tf.math.log(other_var))  # ln|Σ_Q|
        log_det_mixture_var = tf.reduce_sum(tf.math.log(mixture_var))  # ln|Σ_M|
        
        # Compute the mean difference term
        mean_diff = self_mean - other_mean  # Shape: [batch_size, num_actions]
        
        # Compute the quadratic form term
        quadratic_term = tf.reduce_sum(
            (order - 1) * mean_diff ** 2 / other_var, axis=-1
        )
        
        # Compute the determinant term
        determinant_term = log_det_mixture_var - (order - 1) * log_det_other_var - (1 - order) * log_det_self_var
        
        # Compute Rényi divergence
        renyi = 0.5 * (quadratic_term - determinant_term)
        
        # Adjust for order
        renyi /= (order - 1)
        
        # Exponentiate to get exp(Rényi divergence)
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
        
