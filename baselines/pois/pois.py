import numpy as np
import warnings
import tensorflow as tf
import time
from contextlib import contextmanager
from collections import deque
from baselines import logger
from baselines.common.cg import cg
from baselines.pois.utils import add_disc_rew, cluster_rewards
import pdb
np.set_printoptions(precision=16, suppress=False)
# Enable eager execution
tf.config.run_functions_eagerly(True)

@contextmanager
def timed(msg):
    print(msg)
    tstart = time.time()
    yield
    print('done in %.3f seconds' % (time.time() - tstart))
    
def is_in_graph(var):
    return hasattr(var, 'graph')

def update_epsilon(delta_bound, epsilon_old, max_increase=2.):
    if delta_bound > (1. - 1. / (2 * max_increase)) * epsilon_old:
        return epsilon_old * max_increase
    else:
        return epsilon_old ** 2 / (2 * (epsilon_old - delta_bound))

def line_search_parabola(theta_init, alpha, natural_gradient, set_parameter, policy, evaluate_bound,
                         delta_bound_tol=1e-4, max_line_search_ite=30):
    epsilon = 1.
    epsilon_old = 0.
    delta_bound_old = -np.inf
    bound_init = evaluate_bound()
    theta_old = theta_init
    # delta_bound_tol = 1e3
    # print("before line search iter")
    # print(theta_init)
    # print("alpha: ", alpha)
    # print("bound_init: ", bound_init)
    for i in range(max_line_search_ite):
        theta = theta_init + epsilon * alpha * natural_gradient
        # print(policy.trainable_variables)
        # print(epsilon * alpha * natural_gradient)
        # print("natural_gradient: ", natural_gradient)
        # print("theta_init: ", theta_init)
        # print("additive term: ", epsilon * alpha * natural_gradient)
        # print("updated theta: ", theta)
        set_parameter(policy, theta)
        # print("after line search iter %d" % i)
        # print(policy.trainable_variables)
        bound = evaluate_bound()

        if np.isnan(bound):
            warnings.warn('Got NaN bound value: rolling back!')
            return theta_old, epsilon_old, delta_bound_old, i + 1

        delta_bound = bound - bound_init
        # print("bound: ", bound)
        # print("delta_bound: ", delta_bound)
        epsilon_old = epsilon
        epsilon = update_epsilon(delta_bound, epsilon_old)
        if delta_bound <= delta_bound_old + delta_bound_tol:
            if delta_bound_old < 0.:
                return theta_init, 0., 0., i + 1
            else:
                return theta_old, epsilon_old, delta_bound_old, i + 1

        delta_bound_old = delta_bound
        theta_old = theta

    return theta_old, epsilon_old, delta_bound_old, i + 1

def line_search_binary(theta_init, alpha, natural_gradient, set_parameter, evaluate_loss,
                       delta_bound_tol=1e-4, max_line_search_ite=30):
    low = 0.
    high = None
    bound_init = evaluate_loss()
    delta_bound_old = 0.
    theta_opt = theta_init
    i_opt = 0
    delta_bound_opt = 0.
    epsilon_opt = 0.

    epsilon = 1.

    for i in range(max_line_search_ite):

        theta = theta_init + epsilon * natural_gradient * alpha
        set_parameter(theta)

        bound = evaluate_loss()
        delta_bound = bound - bound_init

        if np.isnan(bound):
            warnings.warn('Got NaN bound value: rolling back!')

        if np.isnan(bound) or delta_bound <= delta_bound_opt:
            high = epsilon
        else:
            low = epsilon
            theta_opt = theta
            delta_bound_opt = delta_bound
            i_opt = i
            epsilon_opt = epsilon

        epsilon_old = epsilon

        if high is None:
            epsilon *= 2
        else:
            epsilon = (low + high) / 2.

        if abs(epsilon_old - epsilon) < 1e-12:
            break

    return theta_opt, epsilon_opt, delta_bound_opt, i_opt + 1

def line_search_constant(theta_init, alpha, natural_gradient, set_parameter, evaluate_bound,
                         delta_bound_tol=1e-4, max_line_search_ite=1):
    epsilon = 1
    bound_init = evaluate_bound()
    exit_loop = False

    for _ in range(max_line_search_ite):

        theta = theta_init + epsilon * natural_gradient * alpha
        set_parameter(theta)

        bound = evaluate_bound()

        if np.isnan(bound):
            epsilon /= 2
            continue

        delta_bound = bound - bound_init

        if delta_bound <= -np.inf + delta_bound_tol:
            epsilon /= 2
        else:
            exit_loop = True
            break

    return theta, epsilon, delta_bound, 1

def optimize_offline(theta_init, set_parameter, policy, line_search, evaluate_loss, evaluate_grads,
                     evaluate_natural_gradient=None, gradient_tol=1e-4, bound_tol=1e-4,
                     max_offline_ite=100, constant_step_size=1):
    theta = theta_old = theta_init
    improvement = improvement_old = 0.
    set_parameter(policy, theta)

    fmtstr = '%6i %10.3g %10.3g %18i %18.3g %18.3g %18.3g'
    titlestr = '%6s %10s %10s %18s %18s %18s %18s'
    print(titlestr % ('iter', 'epsilon', 'step size', 'num line search', 'gradient norm', 'delta bound ite', 'delta bound tot'))
    print("starting offline optimization")
    for i in range(max_offline_ite):
        bound = evaluate_loss()
        gradient = evaluate_grads()
        if np.any(np.isnan(bound)):
            warnings.warn('Got NaN bound! Stopping!')
            set_parameter(policy, theta_old)
            return theta_old, improvement

        if np.isnan(bound):
            warnings.warn('Got NaN bound! Stopping!')
            set_parameter(policy, theta_old)
            return theta_old, improvement_old

        if evaluate_natural_gradient is not None:
            natural_gradient = evaluate_natural_gradient(gradient)
        else:
            natural_gradient = gradient

        if np.dot(gradient, natural_gradient) < 0:
            warnings.warn('NatGradient dot Gradient < 0! Using vanilla gradient')
            natural_gradient = gradient

        gradient_norm = np.sqrt(np.dot(gradient, natural_gradient))
        if gradient_norm < gradient_tol:
            print('stopping - gradient norm < gradient_tol')
            return theta, improvement
        
        if constant_step_size != 1:
            alpha = constant_step_size
        else:
            alpha = 1. / gradient_norm ** 2
        if alpha > 2: # limit the step size to avoid instability
            alpha = 2
        theta_old = theta
        improvement_old = improvement
        theta, epsilon, delta_bound, num_line_search = line_search(theta, alpha, natural_gradient, set_parameter, policy, evaluate_loss)
        set_parameter(policy, theta)

        improvement += delta_bound
        print(fmtstr % (i + 1, epsilon, alpha * epsilon, num_line_search, gradient_norm, delta_bound, improvement))

        if delta_bound < bound_tol:
            print('stopping - delta bound < bound_tol')
            return theta, improvement

    return theta, improvement

def learn(make_env, make_policy, *,
          n_episodes,
          horizon,
          delta,
          gamma,
          max_iters,
          sampler=None,
          use_natural_gradient=False,  # can be 'exact', 'approximate'
          fisher_reg=1e-2,
          iw_method='is',
          iw_norm='none',
          bound='J',
          line_search_type='parabola',
          save_weights=0,
          improvement_tol=0.,
          center_return=False,
          render_after=None,
          max_offline_iters=100,
          callback=None,
          clipping=False,
          entropy='none',
          positive_return=False,
          reward_clustering='none',
          learnable_variance=True,
          constant_step_size=1,
          shift_return=False,
          variance_init=-1):

    np.set_printoptions(precision=3)
    max_samples = horizon * n_episodes

    if line_search_type == 'binary':
        line_search = line_search_binary
    elif line_search_type == 'parabola':
        line_search = line_search_parabola
    elif line_search_type == 'constant':
        line_search = line_search_constant
    else:
        raise ValueError("Unsupported line search type.")

    if constant_step_size != 1 and line_search_type != 'constant':
        line_search = line_search_constant

    # Building the environment
    env = make_env()
    ob_space = env.observation_space
    ac_space = env.action_space

    # Building the policy
    pi = make_policy('pi', ob_space, ac_space)
    oldpi = make_policy('oldpi', ob_space, ac_space)

    # Ensure that 'pi' and 'oldpi' are instances of tf.keras.Model
    # Initialize models by running them once
    observation, info = env.reset()
    dummy_ob = tf.constant(observation[None, :], dtype=tf.float32)
    pd, _ = pi(dummy_ob)
    old_pd, _ = oldpi(dummy_ob)

    # Initialize old policy with pi's weights
    oldpi.set_weights(pi.get_weights())

    # Collect the list of policy variables
    var_list = pi.trainable_variables
    shapes = [tf.size(var).numpy() for var in var_list]
    n_parameters = sum(shapes)
    # n_parameters = 12

    # Define the loss and gradient computations
    def compute_loss_and_grad(ob, ac, rew, disc_rew, ep_return, ep_return_opt, mask, iter_number):
        with tf.GradientTape() as tape:
            # Compute log probabilities
            oldpi.trainable = False
            pi.trainable = True
            pd, _ = pi(ob)
            pd_old, _ = oldpi(ob)
            target_log_pdf = pd.logp(ac)
            behavioral_log_pdf = pd_old.logp(ac)
            log_ratio = target_log_pdf - behavioral_log_pdf
            # Reshape operations
            log_ratio_split = tf.reshape(log_ratio * mask, (n_episodes, horizon))
            return_abs_max = tf.reduce_max(tf.abs(ep_return))
            optimization_return_abs_max = tf.reduce_max(tf.abs(ep_return_opt))
            # Compute importance weights
            if iw_method == 'is':
                iw = tf.exp(tf.reduce_sum(log_ratio_split, axis=1))
                if iw_norm == 'none':
                    iwn = iw / tf.cast(n_episodes, tf.float32)
                    if shift_return:
                        w_return_mean = tf.reduce_sum(iwn * ep_return_opt)
                    else:
                        w_return_mean = tf.reduce_sum(iwn * ep_return)
                    J_sample_variance = (1 / (n_episodes - 1)) * tf.reduce_sum(
                        tf.square(iw * ep_return_opt - w_return_mean))
                elif iw_norm == 'sn':
                    iwn = iw / tf.reduce_sum(iw)
                    w_return_mean = tf.reduce_sum(iwn * ep_return)
                elif iw_norm == 'regression':
                    # Implement regression-based importance weighting
                    pass  # To be implemented
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError("Only 'is' importance weighting is implemented.")

            # Compute empirical Renyi divergence
            renyi_d2 = pd.renyi(old_pd, 2)
            renyi_d2 = tf.reshape(renyi_d2, tf.shape(mask))
            emp_d2 = tf.reduce_mean(tf.exp(tf.reduce_sum(renyi_d2 * mask, axis=-1)))
            ess_renyi = tf.cast(n_episodes, tf.float32) / emp_d2
            ESS_RENYI_MIN = 1e-6  # Define a minimum threshold for ess_renyi to avoid numerical instability
            ess_renyi = tf.maximum(ess_renyi, ESS_RENYI_MIN)
            # Compute the bound
            if bound == 'J':
                bound_ = w_return_mean
            elif bound == 'std-d2':
                return_std = tf.math.reduce_std(ep_return)
                bound_ = w_return_mean - tf.sqrt((1 - delta) / (delta * ess_renyi)) * return_std
            elif bound == 'max-d2':
                CLIP_BOUND = 10  # Define a reasonable upper limit for the square root term
                scaling_term = tf.sqrt((1 - delta) / (delta * ess_renyi))
                scaling_term = tf.clip_by_value(scaling_term, 0, CLIP_BOUND)
                if shift_return:
                    bound_ = w_return_mean - scaling_term * optimization_return_abs_max
                else:
                    bound_ = w_return_mean - scaling_term * return_abs_max
            else:
                raise NotImplementedError("Only 'J', 'std-d2', and 'max-d2' bounds are implemented.")

            # Policy entropy for exploration
            ent = pd.entropy()
            meanent = tf.reduce_mean(ent)

            # Add policy entropy bonus if specified
            if entropy != 'none':
                scheme, v1, v2 = entropy.split(':')
                if scheme == 'step':
                    entcoeff = tf.cond(iter_number < int(v2), lambda: float(v1), lambda: float(0.0))
                    entbonus = entcoeff * meanent
                    bound_ = bound_ + entbonus
                elif scheme == 'lin':
                    ip = tf.cast(iter_number / max_iters, tf.float32)
                    entcoeff_decay = tf.maximum(0.0, float(v2) + (float(v1) - float(v2)) * (1.0 - ip))
                    entbonus = entcoeff_decay * meanent
                    bound_ = bound_ + entbonus
                elif scheme == 'exp':
                    iw_mean = tf.reduce_mean(iw)
                    ent_f = tf.exp(-tf.abs(iw_mean - 1) * float(v2)) * float(v1)
                    bound_ = bound_ + ent_f * meanent
                else:
                    raise Exception('Unrecognized entropy scheme.')

            # Assuming the goal is to maximize the bound, we minimize the negative bound
            total_loss = -bound_
            # if np.abs(bound_) > 1e3:
            #     print("bound_: ", bound_)
            #     print("total_loss: ", total_loss)
            #     print("ess_renyi: ", ess_renyi)
            #     print("return_abs_max: ", return_abs_max)
            #     print("optimization_return_abs_max: ", optimization_return_abs_max)
            #     print("w_return_mean: ", w_return_mean)
            #     print("delta", delta)
            #     print("scaling_term",  scaling_term)
            #     # leads to instability
            #     pdb.set_trace()
    
        grads = tape.gradient(total_loss, var_list)
        # print size of grads
        flat_grad = tf.concat([tf.reshape(g, [-1]) for g in grads if g is not None], axis=0)

        # Collect losses for logging
        losses = {
            'MeanEntropy': meanent,
            'Bound': bound_,
            'ReturnMeanIW': w_return_mean,
            'MeanEntropy': meanent,
            'J_sample_variance': J_sample_variance,
            # Add other losses as needed
        }

        return losses, flat_grad

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
        trainable_variables = model.trainable_variables
        if not trainable_variables:
            raise ValueError("The model has no trainable variables.")
        
        flattened_variables = [tf.reshape(v, [-1]) for v in trainable_variables]
        if not flattened_variables:
            raise ValueError("No variables to concatenate after flattening.")
    
        return tf.concat(flattened_variables, axis=0).numpy()
    # Starting optimization
    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=n_episodes)
    rewbuffer = deque(maxlen=n_episodes)

    while True:
        iters_so_far += 1
        if callback:
            callback(locals(), globals())

        if iters_so_far >= max_iters:
            print('Finished...')
            break
        logger.log('********** Iteration %i ************' % iters_so_far)

        theta = get_policy_parameters(pi)
        # print("before sampling")
        # print(theta)

        with timed('sampling'):
            seg = sampler.collect(theta)

        add_disc_rew(seg, gamma)

        lens, rets = seg['ep_lens'], seg['ep_rets']
        lenbuffer.extend(lens)
        rewbuffer.extend(rets)
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)

        # Get clustered reward
        reward_matrix = np.reshape(seg['disc_rew'] * seg['mask'], (n_episodes, horizon))
        ep_reward = np.sum(reward_matrix, axis=1)
        ep_reward = cluster_rewards(ep_reward, reward_clustering)
        ep_reward_opt = ep_reward - np.min(ep_reward)
        args = (seg['ob'], seg['ac'], seg['rew'], seg['disc_rew'], ep_reward, ep_reward_opt, seg['mask'], iters_so_far)

        # Update old policy
        oldpi.set_weights(pi.get_weights())

        def evaluate_grads():
            _, grads = compute_loss_and_grad(*args)
            return grads.numpy()
        
        def evaluate_loss():
            losses, _ = compute_loss_and_grad(*args)
            return losses['Bound'].numpy()

        if use_natural_gradient:
            # Implement natural gradient computation if required
            def evaluate_fisher_vector_prod(x):
                # Implement the Fisher vector product
                return compute_fisher_vector_product(x, args) + fisher_reg * x

            def evaluate_natural_gradient(g):
                return cg(evaluate_fisher_vector_prod, g, cg_iters=10, verbose=0)
        else:
            evaluate_natural_gradient = None

        with timed('summaries before'):
            current_bound = evaluate_loss()
            logger.record_tabular("Iteration", iters_so_far)
            logger.record_tabular("InitialBound", current_bound)
            logger.record_tabular("EpLenMean", np.mean(lenbuffer))
            logger.record_tabular("EpRewMean", np.mean(rewbuffer))
            logger.record_tabular("EpThisIter", len(lens))
            logger.record_tabular("EpisodesSoFar", episodes_so_far)
            logger.record_tabular("TimestepsSoFar", timesteps_so_far)
            logger.record_tabular("TimeElapsed", time.time() - tstart)
            logger.record_tabular("LearnableVariance", learnable_variance)
            logger.record_tabular("VarianceInit", variance_init)

        if save_weights > 0 and iters_so_far % save_weights == 0:
            logger.record_tabular('Weights', str(theta))
            # Optionally, save weights using tf.keras.Model's save methods
            # pi.save_weights(f'checkpoint_iter_{iters_so_far}.h5')

        with timed("offline optimization"):
            theta_opt, improvement = optimize_offline(
                theta_init=theta,
                set_parameter=set_policy_parameters,
                policy=pi,
                line_search=line_search,
                evaluate_loss=evaluate_loss,
                evaluate_grads=evaluate_grads,
                evaluate_natural_gradient=evaluate_natural_gradient,
                max_offline_ite=max_offline_iters,
                constant_step_size=constant_step_size
            )
        # pdb.set_trace()
        # print("after offline optimization")
        # print(theta_opt)
        set_policy_parameters(pi, theta_opt)
        # print(get_policy_parameters(pi))

        with timed('summaries after'):
            losses, _ = compute_loss_and_grad(*args)
            for name, value in losses.items():
                logger.record_tabular(name, value.numpy())

        logger.dump_tabular()

    env.close()

        
        

def compute_fisher_vector_product(x, args):
    # Implement the computation of the Fisher vector product
    # This is a placeholder; actual implementation depends on the policy model
    pass

