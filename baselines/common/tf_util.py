import numpy as np
import tensorflow as tf
import copy
import os
import functools
import collections
import multiprocessing

def switch(condition, then_expression, else_expression):
    """Switches between two operations depending on a scalar value (int or bool).
    Note that both `then_expression` and `else_expression`
    should be tensors of the *same shape*.

    # Arguments
        condition: scalar tensor.
        then_expression: TensorFlow operation.
        else_expression: TensorFlow operation.
    """
    condition = tf.cast(condition, tf.bool)
    x = tf.cond(condition, lambda: then_expression, lambda: else_expression)
    x.set_shape(then_expression.shape)
    return x

# ================================================================
# Extras
# ================================================================

def lrelu(x, leak=0.2):
    return tf.nn.leaky_relu(x, alpha=leak)

# ================================================================
# Mathematical utils
# ================================================================

def huber_loss(x, delta=1.0):
    """Reference: https://en.wikipedia.org/wiki/Huber_loss"""
    return tf.where(
        tf.abs(x) < delta,
        0.5 * tf.square(x),
        delta * (tf.abs(x) - 0.5 * delta)
    )

# ================================================================
# Global session
# ================================================================

# TensorFlow 2.x uses eager execution by default, so session management is typically unnecessary.
# However, if you need to configure thread settings, you can use the following approach.

def configure_threading(num_cpu=None):
    """Configures threading for TensorFlow operations."""
    if num_cpu is None:
        num_cpu = int(os.getenv('RCALL_NUM_CPU', multiprocessing.cpu_count()))
    tf.config.threading.set_intra_op_parallelism_threads(num_cpu)
    tf.config.threading.set_inter_op_parallelism_threads(num_cpu)
    # Allow GPU memory growth if GPUs are available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

def single_threaded():
    """Configures TensorFlow to use a single thread."""
    configure_threading(num_cpu=1)

def in_session(f):
    @functools.wraps(f)
    def newfunc(*args, **kwargs):
        configure_threading()
        return f(*args, **kwargs)
    return newfunc

# Initialize is generally handled automatically in TensorFlow 2.x.
# If you need to initialize variables manually, you can do so within a tf.function or class constructor.

# ================================================================
# Model components
# ================================================================

def normc_initializer(std=1.0):
    def _initializer(shape, dtype=None, **kwargs):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), padding="SAME", dtype=tf.float32, summary_tag=None):
    initializer = normc_initializer()(shape=filter_size + [int(x.shape[-1]), num_filters], dtype=dtype)
    conv = tf.keras.layers.Conv2D(
        filters=num_filters,
        kernel_size=filter_size,
        strides=stride,
        padding=padding,
        dtype=dtype,
        kernel_initializer=initializer,
        bias_initializer='zeros',
        name=name
    )
    z = conv(x)
    if summary_tag is not None:
        tf.summary.image(
            summary_tag,
            tf.transpose(tf.reshape(conv.kernel, [filter_size[0], filter_size[1], -1, 1]), [2, 0, 1, 3]),
            max_outputs=10
        )
    return z

# ================================================================
# Theano-like Function
# ================================================================

def function(inputs, outputs, updates=None, givens=None):
    """Creates a TensorFlow function similar to Theano's function.

    Parameters
    ----------
    inputs: list of input tensors
    outputs: list of output tensors or a single tensor
    updates: list of update operations
    givens: dictionary mapping tensors to replace

    Returns
    -------
    A callable function that executes the computation graph.
    """
    if givens is None:
        givens = {}
    @tf.function
    def wrapped_function(*args, **kwargs):
        feed_dict = {}
        for inp, arg in zip(inputs, args):
            feed_dict[inp] = arg
        for key, value in kwargs.items():
            feed_dict[key] = value
        return tf.compat.v1.get_default_graph().gradient_override_map(givens)(*args, **kwargs)
    return wrapped_function

# ================================================================
# Flat vectors
# ================================================================

def var_shape(x):
    return x.shape.as_list()

def numel(x):
    return int(np.prod(var_shape(x)))

def intprod(x):
    return int(np.prod(x))

def flatgrad(loss, var_list, clip_norm=None):
    grads = tf.gradients(loss, var_list)
    if clip_norm is not None:
        grads, _ = tf.clip_by_global_norm(grads, clip_norm)
    flat_g = tf.concat([tf.reshape(g, [-1]) for g in grads if g is not None], axis=0)
    return flat_g

def assignFromFlat(var_list, values):
    assigns = []
    shapes = [var_shape(v) for v in var_list]
    starts = np.cumsum([0] + [intprod(s) for s in shapes])
    for v, start, shape in zip(var_list, starts[:-1], shapes):
        size = intprod(shape)
        assigns.append(tf.assign(v, tf.reshape(values[start:start + size], shape)))
    return tf.group(*assigns)

class SetFromFlat(tf.Module):
    def __init__(self, var_list, dtype=tf.float32):
        super(SetFromFlat, self).__init__()
        self.theta = tf.Variable(tf.zeros([sum(intprod(var_shape(v)) for v in var_list)], dtype=dtype))
        self.assign_ops = []
        start = 0
        for v in var_list:
            size = intprod(var_shape(v))
            self.assign_ops.append(v.assign(tf.reshape(self.theta[start:start + size], var_shape(v))))
            start += size
        self.assign_group = tf.group(*self.assign_ops)

    @tf.function
    def __call__(self, theta):
        self.theta.assign(theta)
        self.assign_group()

class GetFlat(tf.Module):
    def __init__(self, var_list):
        super(GetFlat, self).__init__()
        self.var_list = var_list

    @tf.function
    def __call__(self):
        return tf.concat([tf.reshape(v, [-1]) for v in self.var_list], axis=0)

_PLACEHOLDER_CACHE = {}  # name -> (placeholder, dtype, shape)

def get_placeholder(name, dtype, shape):
    if name in _PLACEHOLDER_CACHE:
        out, dtype1, shape1 = _PLACEHOLDER_CACHE[name]
        assert dtype1 == dtype and shape1 == shape, "Placeholder cache mismatch."
        return out
    else:
        # In TF2.x, placeholders are replaced by function arguments or tf.function inputs
        raise NotImplementedError("Placeholders are not used in TensorFlow 2.x")

def get_placeholder_cached(name):
    if name in _PLACEHOLDER_CACHE:
        return _PLACEHOLDER_CACHE[name][0]
    else:
        raise KeyError(f"Placeholder '{name}' not found in cache.")

def flattenallbut0(x):
    return tf.reshape(x, [tf.shape(x)[0], -1])

def reduce_var(x, axis=None, keepdims=False):
    """Variance of a tensor along specified axis."""
    return tf.math.reduce_variance(x, axis=axis, keepdims=keepdims)

def reduce_std(x, axis=None, keepdims=False):
    """Standard deviation of a tensor along specified axis."""
    return tf.math.reduce_std(x, axis=axis, keepdims=keepdims)
