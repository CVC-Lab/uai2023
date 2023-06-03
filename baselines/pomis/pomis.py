import numpy as np
import warnings
import baselines.common.tf_util as U
import tensorflow as tf
import time
from baselines.common import zipsame, colorize
from contextlib import contextmanager
from collections import deque
from baselines import logger
from baselines.common.cg import cg
from baselines.pomis.memory import Memory
from baselines.pois.utils import add_disc_rew, cluster_rewards

@contextmanager
def timed(msg):
    print(colorize(msg, color='magenta'))
    tstart = time.time()
    yield
    print(colorize('done in %.3f seconds'%(time.time() - tstart), color='magenta'))

def update_epsilon(delta_bound, epsilon_old, max_increase=2.):
    if delta_bound > (1. - 1. / (2 * max_increase)) * epsilon_old:
        return epsilon_old * max_increase
    else:
        return epsilon_old ** 2 / (2 * (epsilon_old - delta_bound))

def line_search_parabola(theta_init, alpha, natural_gradient, set_parameter, evaluate_bound, delta_bound_tol=1e-4, max_line_search_ite=30):
    epsilon = 1.
    epsilon_old = 0.
    delta_bound_old = -np.inf
    bound_init = evaluate_bound()
    theta_old = theta_init

    for i in range(max_line_search_ite):

        theta = theta_init + epsilon * alpha * natural_gradient
        set_parameter(theta)

        bound = evaluate_bound()

        if np.isnan(bound):
            warnings.warn('Got NaN bound value: rolling back!')
            return theta_old, epsilon_old, delta_bound_old, i + 1

        delta_bound = bound - bound_init

        epsilon_old = epsilon
        epsilon = update_epsilon(delta_bound, epsilon_old)
        if delta_bound <= delta_bound_old + delta_bound_tol:
            if delta_bound_old < 0.:
                return theta_init, 0., 0., i+1
            else:
                return theta_old, epsilon_old, delta_bound_old, i+1

        delta_bound_old = delta_bound
        theta_old = theta

    return theta_old, epsilon_old, delta_bound_old, i+1

def line_search_binary(theta_init, alpha, natural_gradient, set_parameter, evaluate_loss, delta_bound_tol=1e-4, max_line_search_ite=30):
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

    return theta_opt, epsilon_opt, delta_bound_opt, i_opt+1

def optimize_offline(theta_init, set_parameter, line_search, evaluate_loss, evaluate_gradient, evaluate_natural_gradient=None, gradient_tol=1e-4, bound_tol=1e-4, max_offline_ite=100):
    theta = theta_old = theta_init
    improvement = improvement_old = 0.
    set_parameter(theta)


    '''
    bound_init = evaluate_loss()
    import scipy.optimize as opt

    def func(x):
        set_parameter(x)
        return -evaluate_loss()

    def grad(x):
        set_parameter(x)
        return -evaluate_gradient().astype(np.float64)

    theta, bound, d = opt.fmin_l_bfgs_b(func=func,
                                        fprime=grad,
                                x0=theta_init.astype(np.float64),
                                maxiter=100,
                                    )
    print(bound_init, bound)

    print(d)

    set_parameter(theta)
    improvement = bound_init + bound
    return theta, improvement

    '''

    fmtstr = '%6i %10.3g %10.3g %18i %18.3g %18.3g %18.3g'
    titlestr = '%6s %10s %10s %18s %18s %18s %18s'
    print(titlestr % ('iter', 'epsilon', 'step size', 'num line search', 'gradient norm', 'delta bound ite', 'delta bound tot'))

    for i in range(max_offline_ite):
        bound = evaluate_loss()
        gradient = evaluate_gradient()

        if np.any(np.isnan(gradient)):
            warnings.warn('Got NaN gradient! Stopping!')
            set_parameter(theta_old)
            return theta_old, improvement

        if np.isnan(bound):
            warnings.warn('Got NaN bound! Stopping!')
            set_parameter(theta_old)
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

        alpha = 1. / gradient_norm ** 2

        theta_old = theta
        improvement_old = improvement
        theta, epsilon, delta_bound, num_line_search = line_search(theta, alpha, natural_gradient, set_parameter, evaluate_loss)
        set_parameter(theta)

        improvement += delta_bound
        print(fmtstr % (i+1, epsilon, alpha*epsilon, num_line_search, gradient_norm, delta_bound, improvement))

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
          use_natural_gradient=False, #can be 'exact', 'approximate'
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
          capacity=10,
          warm_start=True):

    np.set_printoptions(precision=3)
    max_samples = horizon * n_episodes

    if line_search_type == 'binary':
        line_search = line_search_binary
    elif line_search_type == 'parabola':
        line_search = line_search_parabola
    else:
        raise ValueError()

    # Building the environment
    env = make_env()
    ob_space = env.observation_space
    ac_space = env.action_space

    # Creating the memory buffer
    memory = Memory(capacity=capacity, batch_size=n_episodes, horizon=horizon,
                    ob_space=ob_space, ac_space=ac_space)

    # Building the target policy and saving its parameters
    pi = make_policy('pi', ob_space, ac_space)
    all_var_list = pi.get_trainable_variables()
    var_list = [v for v in all_var_list if v.name.split('/')[1].startswith('pol')]
    shapes = [U.intprod(var.get_shape().as_list()) for var in var_list]
    n_parameters = sum(shapes)

    # Building a set of behavioral policies
    memory.build_policies(make_policy, pi)

    # Placeholders
    ob_ = ob = U.get_placeholder_cached(name='ob')
    ac_ = pi.pdtype.sample_placeholder([None], name='ac')
    mask_ = tf.placeholder(dtype=tf.float32, shape=(None), name='mask')
    rew_ = tf.placeholder(dtype=tf.float32, shape=(None), name='rew')
    disc_rew_ = tf.placeholder(dtype=tf.float32, shape=(None), name='disc_rew')
    clustered_rew_ = tf.placeholder(dtype=tf.float32, shape=(None))
    gradient_ = tf.placeholder(dtype=tf.float32, shape=(n_parameters, 1), name='gradient')
    iter_number_ = tf.placeholder(dtype=tf.int32, name='iter_number')
    active_policies = tf.placeholder(dtype=tf.float32, shape=(capacity), name='active_policies')
    losses_with_name = []

    # Total number of trajectories
    N_total = tf.reduce_sum(active_policies) * n_episodes

    # Split operations
    disc_rew_split = tf.reshape(disc_rew_ * mask_, [-1, horizon])
    rew_split = tf.reshape(rew_ * mask_, [-1, horizon])
    mask_split = tf.reshape(mask_, [-1, horizon])

    # Policy densities
    target_log_pdf = pi.pd.logp(ac_) * mask_
    target_log_pdf_split = tf.reshape(target_log_pdf, [-1, horizon])
    behavioral_log_pdfs = tf.stack([bpi.pd.logp(ac_) * mask_ for bpi in memory.policies]) # Shape is (capacity, ntraj*horizon)
    behavioral_log_pdfs_split = tf.reshape(behavioral_log_pdfs, [memory.capacity, -1, horizon])

    # Compute renyi divergencies and sum over time, then exponentiate
    emp_d2_split = tf.reshape(tf.stack([pi.pd.renyi(bpi.pd, 2) * mask_ for bpi in memory.policies]), [memory.capacity, -1, horizon])
    emp_d2_split_cum = tf.exp(tf.reduce_sum(emp_d2_split, axis=2))
    # Compute arithmetic and harmonic mean of emp_d2
    emp_d2_mean = tf.reduce_mean(emp_d2_split_cum, axis=1)
    emp_d2_arithmetic = tf.reduce_sum(emp_d2_mean * active_policies) / tf.reduce_sum(active_policies)
    emp_d2_harmonic = tf.reduce_sum(active_policies) / tf.reduce_sum(1 / emp_d2_mean)

    # Return processing: clipping, centering, discounting
    ep_return = clustered_rew_ #tf.reduce_sum(mask_split * disc_rew_split, axis=1)
    if clipping:
        rew_split = tf.clip_by_value(rew_split, -1, 1)
    if center_return:
        ep_return = ep_return - tf.reduce_mean(ep_return)
        rew_split = rew_split - (tf.reduce_sum(rew_split) / (tf.reduce_sum(mask_split) + 1e-24))
    discounter = [pow(gamma, i) for i in range(0, horizon)] # Decreasing gamma
    discounter_tf = tf.constant(discounter)
    disc_rew_split = rew_split * discounter_tf

    # Reward statistics
    return_mean = tf.reduce_mean(ep_return)
    return_std = U.reduce_std(ep_return)
    return_max = tf.reduce_max(ep_return)
    return_min = tf.reduce_min(ep_return)
    return_abs_max = tf.reduce_max(tf.abs(ep_return))
    return_step_max = tf.reduce_max(tf.abs(rew_split)) # Max step reward
    return_step_mean = tf.abs(tf.reduce_mean(rew_split))
    positive_step_return_max = tf.maximum(0.0, tf.reduce_max(rew_split))
    negative_step_return_max = tf.maximum(0.0, tf.reduce_max(-rew_split))
    return_step_maxmin = tf.abs(positive_step_return_max - negative_step_return_max)
    losses_with_name.extend([(return_mean, 'InitialReturnMean'),
                             (return_max, 'InitialReturnMax'),
                             (return_min, 'InitialReturnMin'),
                             (return_std, 'InitialReturnStd'),
                             (emp_d2_arithmetic, 'EmpiricalD2Arithmetic'),
                             (emp_d2_harmonic, 'EmpiricalD2Harmonic'),
                             (return_step_max, 'ReturnStepMax'),
                             (return_step_maxmin, 'ReturnStepMaxmin')])

    # Add D2 statistics for each memory cell
    for i in range(capacity):
        losses_with_name.extend([(tf.reduce_mean(emp_d2_split_cum, axis=1)[i], 'MeanD2-'+str(i))])

    if iw_method == 'is':
        # Sum the log prob over time. Shapes: target(Nep, H), behav (Cap, Nep, H)
        target_log_pdf_episode = tf.reduce_sum(target_log_pdf_split, axis=1)
        behavioral_log_pdf_episode = tf.reduce_sum(behavioral_log_pdfs_split, axis=2)
        # To avoid numerical instability, compute the inversed ratio
        log_inverse_ratio = behavioral_log_pdf_episode - target_log_pdf_episode
        abc = tf.exp(log_inverse_ratio) * tf.expand_dims(active_policies, -1)
        iw = 1 / tf.reduce_sum(tf.exp(log_inverse_ratio) * tf.expand_dims(active_policies, -1), axis=0)
        iwn = iw / n_episodes

        # Compute the J
        w_return_mean = tf.reduce_sum(ep_return * iwn)
        # Empirical D2 of the mixture and relative ESS
        ess_renyi_arithmetic = N_total / emp_d2_arithmetic
        ess_renyi_harmonic = N_total / emp_d2_harmonic
        # Log quantities
        losses_with_name.extend([(tf.reduce_max(iw), 'MaxIW'),
                                 (tf.reduce_min(iw), 'MinIW'),
                                 (tf.reduce_mean(iw), 'MeanIW'),
                                 (U.reduce_std(iw), 'StdIW'),
                                 (tf.reduce_min(target_log_pdf_episode), 'MinTargetPdf'),
                                 (tf.reduce_min(behavioral_log_pdf_episode), 'MinBehavPdf'),
                                 (ess_renyi_arithmetic, 'ESSRenyiArithmetic'),
                                 (ess_renyi_harmonic, 'ESSRenyiHarmonic')])
    else:
        raise NotImplementedError()

    if bound == 'J':
        bound_ = w_return_mean
    elif bound == 'max-d2-harmonic':
        bound_ = w_return_mean - tf.sqrt((1 - delta) / (delta * ess_renyi_harmonic)) * return_abs_max
    elif bound == 'max-d2-arithmetic':
        bound_ = w_return_mean - tf.sqrt((1 - delta) / (delta * ess_renyi_arithmetic)) * return_abs_max
    else:
        raise NotImplementedError()

    # Policy entropy for exploration
    ent = pi.pd.entropy()
    meanent = tf.reduce_mean(ent)
    losses_with_name.append((meanent, 'MeanEntropy'))
    # Add policy entropy bonus
    if entropy != 'none':
        scheme, v1, v2 = entropy.split(':')
        if scheme == 'step':
            entcoeff = tf.cond(iter_number_ < int(v2), lambda: float(v1), lambda: float(0.0))
            losses_with_name.append((entcoeff, 'EntropyCoefficient'))
            entbonus = entcoeff * meanent
            bound_ = bound_ + entbonus
        elif scheme == 'lin':
            ip = tf.cast(iter_number_ / max_iters, tf.float32)
            entcoeff_decay = tf.maximum(0.0, float(v2) + (float(v1) - float(v2)) * (1.0 - ip))
            losses_with_name.append((entcoeff_decay, 'EntropyCoefficient'))
            entbonus = entcoeff_decay * meanent
            bound_ = bound_ + entbonus
        elif scheme == 'exp':
            ent_f = tf.exp(-tf.abs(tf.reduce_mean(iw) - 1) * float(v2)) * float(v1)
            losses_with_name.append((ent_f, 'EntropyCoefficient'))
            bound_ = bound_ + ent_f * meanent
        else:
            raise Exception('Unrecognized entropy scheme.')

    losses_with_name.append((w_return_mean, 'ReturnMeanIW'))
    losses_with_name.append((bound_, 'Bound'))
    losses, loss_names = map(list, zip(*losses_with_name))

    '''
    if use_natural_gradient:
        p = tf.placeholder(dtype=tf.float32, shape=[None])
        target_logpdf_episode = tf.reduce_sum(target_log_pdf_split * mask_split, axis=1)
        grad_logprob = U.flatgrad(tf.stop_gradient(iwn) * target_logpdf_episode, var_list)
        dot_product = tf.reduce_sum(grad_logprob * p)
        hess_logprob = U.flatgrad(dot_product, var_list)
        compute_linear_operator = U.function([p, ob_, ac_, disc_rew_, mask_], [-hess_logprob])
    '''

    assert_ops = tf.group(*tf.get_collection('asserts'))
    print_ops = tf.group(*tf.get_collection('prints'))

    compute_lossandgrad = U.function([ob_, ac_, rew_, disc_rew_, clustered_rew_, mask_, iter_number_, active_policies], losses + [U.flatgrad(bound_, var_list), assert_ops, print_ops])
    compute_grad = U.function([ob_, ac_, rew_, disc_rew_, clustered_rew_, mask_, iter_number_, active_policies], [U.flatgrad(bound_, var_list), assert_ops, print_ops])
    compute_bound = U.function([ob_, ac_, rew_, disc_rew_, clustered_rew_, mask_, iter_number_, active_policies], [bound_, assert_ops, print_ops])
    compute_losses = U.function([ob_, ac_, rew_, disc_rew_, clustered_rew_, mask_, iter_number_, active_policies], losses)
    #compute_temp = U.function([ob_, ac_, rew_, disc_rew_, clustered_rew_, mask_, iter_number_, active_policies], [log_inverse_ratio, abc, iw])

    set_parameter = U.SetFromFlat(var_list)
    get_parameter = U.GetFlat(var_list)
    policy_reinit = tf.variables_initializer(var_list)

    if sampler is None:
        seg_gen = traj_segment_generator(pi, env, n_episodes, horizon, stochastic=True)
        sampler = type("SequentialSampler", (object,), {"collect": lambda self, _: seg_gen.__next__()})()

    U.initialize()

    # Starting optimizing
    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=n_episodes)
    rewbuffer = deque(maxlen=n_episodes)

    while True:

        iters_so_far += 1

        if render_after is not None and iters_so_far % render_after == 0:
            if hasattr(env, 'render'):
                render(env, pi, horizon)

        if callback:
            callback(locals(), globals())

        if iters_so_far >= max_iters:
            print('Finished...')
            break

        logger.log('********** Iteration %i ************' % iters_so_far)

        theta = get_parameter()

        with timed('sampling'):
            seg = sampler.collect(theta)

        add_disc_rew(seg, gamma)

        lens, rets = seg['ep_lens'], seg['ep_rets']
        lenbuffer.extend(lens)
        rewbuffer.extend(rets)
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)

        # Adding batch of trajectories to memory
        memory.add_trajectory_batch(seg)

        # Get multiple batches from memory
        seg_with_memory = memory.get_trajectories()

        # Get clustered reward
        reward_matrix = np.reshape(seg_with_memory['disc_rew'] * seg_with_memory['mask'], (-1, horizon))
        ep_reward = np.sum(reward_matrix, axis=1)
        ep_reward = cluster_rewards(ep_reward, reward_clustering)

        args = ob, ac, rew, disc_rew, clustered_rew, mask, iter_number, active_policies = (seg_with_memory['ob'],
                                                                                           seg_with_memory['ac'],
                                                                                           seg_with_memory['rew'],
                                                                                           seg_with_memory['disc_rew'],
                                                                                           ep_reward,
                                                                                           seg_with_memory['mask'],
                                                                                           iters_so_far,
                                                                                           memory.get_active_policies_mask())

        def evaluate_loss():
            loss = compute_bound(*args)
            return loss[0]

        def evaluate_gradient():
            gradient = compute_grad(*args)
            return gradient[0]

        if use_natural_gradient:
            def evaluate_fisher_vector_prod(x):
                return compute_linear_operator(x, *args)[0] + fisher_reg * x

            def evaluate_natural_gradient(g):
                return cg(evaluate_fisher_vector_prod, g, cg_iters=10, verbose=0)
        else:
            evaluate_natural_gradient = None

        with timed('summaries before'):
            logger.record_tabular("Iteration", iters_so_far)
            logger.record_tabular("InitialBound", evaluate_loss())
            logger.record_tabular("EpLenMean", np.mean(lenbuffer))
            logger.record_tabular("EpRewMean", np.mean(rewbuffer))
            logger.record_tabular("EpThisIter", len(lens))
            logger.record_tabular("EpisodesSoFar", episodes_so_far)
            logger.record_tabular("TimestepsSoFar", timesteps_so_far)
            logger.record_tabular("TimeElapsed", time.time() - tstart)

        if save_weights > 0 and iters_so_far % save_weights == 0:
            logger.record_tabular('Weights', str(get_parameter()))
            import pickle
            file = open('checkpoint' + str(iters_so_far) + '.pkl', 'wb')
            pickle.dump(theta, file)

        if not warm_start or memory.get_current_load() == capacity:
            # Optimize
            with timed("offline optimization"):
                theta, improvement = optimize_offline(theta,
                                                  set_parameter,
                                                  line_search,
                                                  evaluate_loss,
                                                  evaluate_gradient,
                                                  evaluate_natural_gradient,
                                                  max_offline_ite=max_offline_iters)

            set_parameter(theta)
            print(theta)

            with timed('summaries after'):
                meanlosses = np.array(compute_losses(*args))
                for (lossname, lossval) in zip(loss_names, meanlosses):
                    logger.record_tabular(lossname, lossval)
        else:
            # Reinitialize the policy
            tf.get_default_session().run(policy_reinit)

        logger.dump_tabular()

    env.close()
