import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
from jax.nn import softplus
import optax
import numpy as np
from typing import NamedTuple
from tensorflow_probability.substrates.jax.distributions import Poisson
from dynamax.generalized_gaussian_ssm import ParamsGGSSM, EKFIntegrals
from dynamax.generalized_gaussian_ssm import conditional_moments_gaussian_smoother as cmgs
from .preprocessing import stack_trial_pairs_by_time_length, stack_trials_by_time_length

RATE_FLOOR = 1e-4
COV_FLOOR = 1e-4


class PoissonLDSParams(NamedTuple):
    """Trainable Poisson LDS parameters stored in unconstrained optimizer space."""
    A: jnp.ndarray
    C: jnp.ndarray
    d: jnp.ndarray
    log_q: jnp.ndarray
    m0: jnp.ndarray
    log_s0: jnp.ndarray


class PoissonLDSInputParams(NamedTuple):
    """Trainable controlled Poisson LDS parameters stored in unconstrained optimizer space.

    This input-driven variant adds B shaped (state_dim, input_dim). The covariance
    diagonal fields log_q and log_s0 remain unconstrained optimizer parameters.
    """
    A: jnp.ndarray
    B: jnp.ndarray
    C: jnp.ndarray
    d: jnp.ndarray
    log_q: jnp.ndarray
    m0: jnp.ndarray
    log_s0: jnp.ndarray


def inverse_softplus(x):
    """Map positive values to unconstrained softplus inputs.

    Args:
        x: Positive scalar or array of positive values.

    Returns:
        Array with the same shape as x in unconstrained space.
    """
    return jnp.log(jnp.expm1(jnp.maximum(jnp.asarray(x), 1e-6)))


def positive_diag(log_diag):
    """Convert unconstrained covariance diagonal parameters.

    Args:
        log_diag: Vector of unconstrained diagonal covariance parameters.

    Returns:
        Positive covariance diagonal vector with the same shape.
    """
    return softplus(log_diag) + COV_FLOOR


def poisson_rate(trainable_params, z):
    """Compute emission rates for one latent state.

    Args:
        trainable_params: PoissonLDSParams with C shaped (n_neurons, state_dim)
            and d shaped (n_neurons,).
        z: Latent state vector shaped (state_dim,).

    Returns:
        Positive Poisson rates shaped (n_neurons,).
    """
    return softplus(trainable_params.C @ z + trainable_params.d) + RATE_FLOOR


def make_poisson_lds_params(trainable_params):
    """Build Dynamax GGSSM parameters from unconstrained trainable arrays.

    Args:
        trainable_params: PoissonLDSParams containing A (state_dim, state_dim),
            C (n_neurons, state_dim), d (n_neurons,), log_q (state_dim,),
            m0 (state_dim,), and log_s0 (state_dim,).

    Returns:
        ParamsGGSSM with diagonal initial/dynamics covariances and Poisson emissions.
    """
    rate = lambda z: poisson_rate(trainable_params, z)
    return ParamsGGSSM(
        initial_mean=trainable_params.m0,
        initial_covariance=jnp.diag(positive_diag(trainable_params.log_s0)),
        dynamics_function=lambda z: trainable_params.A @ z,
        dynamics_covariance=jnp.diag(positive_diag(trainable_params.log_q)),
        emission_mean_function=rate,
        emission_cov_function=lambda z: jnp.diag(rate(z)),
        emission_dist=lambda mu, Sigma: Poisson(rate=mu),
    )


def make_poisson_lds_input_params(trainable_params):
    """Build Dynamax GGSSM parameters for a controlled Poisson LDS.

    Args:
        trainable_params: PoissonLDSInputParams containing A (state_dim, state_dim),
            B (state_dim, input_dim), C (n_neurons, state_dim), d (n_neurons,),
            log_q (state_dim,), m0 (state_dim,), and log_s0 (state_dim,).

    Returns:
        ParamsGGSSM whose dynamics and emission moment functions accept inputs.
    """
    rate = lambda z: poisson_rate(trainable_params, z)
    return ParamsGGSSM(
        initial_mean=trainable_params.m0,
        initial_covariance=jnp.diag(positive_diag(trainable_params.log_s0)),
        dynamics_function=lambda z, u: trainable_params.A @ z + trainable_params.B @ u,
        dynamics_covariance=jnp.diag(positive_diag(trainable_params.log_q)),
        emission_mean_function=lambda z, u: rate(z),
        emission_cov_function=lambda z, u: jnp.diag(rate(z)),
        emission_dist=lambda mu, Sigma: Poisson(rate=mu),
    )


def initialize_trainable_params(y_train, state_dim, key):
    """Initialize trainable parameters from equal-length spike-count trials.

    Args:
        y_train: Count array shaped (n_trials, n_bins, n_neurons).
        state_dim: Number of latent dimensions.
        key: JAX PRNG key used to initialize C.

    Returns:
        PoissonLDSParams in unconstrained optimizer space.
    """
    emission_dim = y_train.shape[-1]
    mean_counts = jnp.mean(y_train, axis=(0, 1))
    baseline_rate = jnp.maximum(mean_counts - RATE_FLOOR, 1e-3)
    return PoissonLDSParams(
        A=0.95 * jnp.eye(state_dim),
        C=0.01 * jr.normal(key, (emission_dim, state_dim)),
        d=inverse_softplus(baseline_rate),
        log_q=inverse_softplus(0.1 * jnp.ones(state_dim)),
        m0=jnp.zeros(state_dim),
        log_s0=inverse_softplus(jnp.ones(state_dim)),
    )


def initialize_trainable_params_from_trials(y_trials, state_dim, key):
    """Initialize trainable parameters from variable-length spike-count trials.

    Args:
        y_trials: Nonempty list of arrays, each shaped (n_bins_i, n_neurons);
            all trials must have the same neuron count.
        state_dim: Number of latent dimensions.
        key: JAX PRNG key used to initialize C.

    Returns:
        PoissonLDSParams in unconstrained optimizer space.
    """
    emission_dim = y_trials[0].shape[-1]
    if any(y.shape[-1] != emission_dim for y in y_trials):
        raise ValueError("all trials must have the same number of neurons")
    mean_counts = jnp.asarray(np.concatenate(y_trials, axis=0).mean(axis=0), dtype=jnp.float64)
    baseline_rate = jnp.maximum(mean_counts - RATE_FLOOR, 1e-3)
    return PoissonLDSParams(
        A=0.95 * jnp.eye(state_dim),
        C=0.01 * jr.normal(key, (emission_dim, state_dim)),
        d=inverse_softplus(baseline_rate),
        log_q=inverse_softplus(0.1 * jnp.ones(state_dim)),
        m0=jnp.zeros(state_dim),
        log_s0=inverse_softplus(jnp.ones(state_dim)),
    )


def initialize_trainable_input_params(y_train, u_train, state_dim, key):
    """Initialize controlled Poisson LDS parameters from equal-length trials.

    Args:
        y_train: Count array shaped (n_trials, n_bins, n_neurons).
        u_train: Input array shaped (n_trials, n_bins, input_dim).
        state_dim: Number of latent dimensions.
        key: JAX PRNG key used to initialize C.

    Returns:
        PoissonLDSInputParams in unconstrained optimizer space.
    """
    if y_train.shape[:2] != u_train.shape[:2]:
        raise ValueError("y_train and u_train must have matching trial and time dimensions")
    emission_dim = y_train.shape[-1]
    input_dim = u_train.shape[-1]
    mean_counts = jnp.mean(y_train, axis=(0, 1))
    baseline_rate = jnp.maximum(mean_counts - RATE_FLOOR, 1e-3)
    return PoissonLDSInputParams(
        A=0.95 * jnp.eye(state_dim),
        B=jnp.zeros((state_dim, input_dim)),
        C=0.01 * jr.normal(key, (emission_dim, state_dim)),
        d=inverse_softplus(baseline_rate),
        log_q=inverse_softplus(0.1 * jnp.ones(state_dim)),
        m0=jnp.zeros(state_dim),
        log_s0=inverse_softplus(jnp.ones(state_dim)),
    )


def initialize_trainable_input_params_from_trials(y_trials, u_trials, state_dim, key):
    """Initialize controlled Poisson LDS parameters from variable-length trials.

    Args:
        y_trials: Nonempty list of arrays, each shaped (n_bins_i, n_neurons).
        u_trials: Nonempty list of arrays, each shaped (n_bins_i, input_dim).
        state_dim: Number of latent dimensions.
        key: JAX PRNG key used to initialize C.

    Returns:
        PoissonLDSInputParams in unconstrained optimizer space.
    """
    stack_trial_pairs_by_time_length(y_trials, u_trials)
    emission_dim = y_trials[0].shape[-1]
    input_dim = u_trials[0].shape[-1]
    mean_counts = jnp.asarray(np.concatenate(y_trials, axis=0).mean(axis=0), dtype=jnp.float64)
    baseline_rate = jnp.maximum(mean_counts - RATE_FLOOR, 1e-3)
    return PoissonLDSInputParams(
        A=0.95 * jnp.eye(state_dim),
        B=jnp.zeros((state_dim, input_dim)),
        C=0.01 * jr.normal(key, (emission_dim, state_dim)),
        d=inverse_softplus(baseline_rate),
        log_q=inverse_softplus(0.1 * jnp.ones(state_dim)),
        m0=jnp.zeros(state_dim),
        log_s0=inverse_softplus(jnp.ones(state_dim)),
    )


def poisson_lds_loss(trainable_params, y_train):
    """Compute negative marginal log likelihood per bin for stacked trials.

    Args:
        trainable_params: PoissonLDSParams to score.
        y_train: Count array shaped (n_trials, n_bins, n_neurons).

    Returns:
        Scalar JAX loss averaged over trials and time bins.
    """
    params = make_poisson_lds_params(trainable_params)
    posts = vmap(lambda y: cmgs(params, EKFIntegrals(), y))(y_train)
    return -jnp.mean(posts.marginal_loglik) / y_train.shape[1]


def poisson_lds_grouped_loss(trainable_params, grouped_y_train):
    """Compute negative marginal log likelihood per bin for grouped trials.

    Args:
        trainable_params: PoissonLDSParams to score.
        grouped_y_train: Tuple of arrays shaped (n_trials_g, n_bins_g, n_neurons),
            one group per shared trial length.

    Returns:
        Scalar JAX loss averaged over all trials and time bins.
    """
    params = make_poisson_lds_params(trainable_params)
    total_loglik = 0.0
    total_bins = 0
    for y_group in grouped_y_train:
        posts = vmap(lambda y: cmgs(params, EKFIntegrals(), y))(y_group)
        total_loglik = total_loglik + jnp.sum(posts.marginal_loglik)
        total_bins += y_group.shape[0] * y_group.shape[1]
    return -total_loglik / total_bins


def poisson_lds_input_loss(trainable_params, y_train, u_train):
    """Compute negative marginal log likelihood per bin for stacked input trials.

    Args:
        trainable_params: PoissonLDSInputParams to score.
        y_train: Count array shaped (n_trials, n_bins, n_neurons).
        u_train: Input array shaped (n_trials, n_bins, input_dim).

    Returns:
        Scalar JAX loss averaged over trials and time bins.
    """
    params = make_poisson_lds_input_params(trainable_params)
    posts = vmap(lambda y, u: cmgs(params, EKFIntegrals(), y, inputs=u))(y_train, u_train)
    return -jnp.mean(posts.marginal_loglik) / y_train.shape[1]


def poisson_lds_input_grouped_loss(trainable_params, grouped_yu_train):
    """Compute negative marginal log likelihood per bin for grouped input trials.

    Args:
        trainable_params: PoissonLDSInputParams to score.
        grouped_yu_train: Tuple of (y_group, u_group) pairs, one pair per
            shared trial length.

    Returns:
        Scalar JAX loss averaged over all trials and time bins.
    """
    params = make_poisson_lds_input_params(trainable_params)
    total_loglik = 0.0
    total_bins = 0
    for y_group, u_group in grouped_yu_train:
        posts = vmap(lambda y, u: cmgs(params, EKFIntegrals(), y, inputs=u))(y_group, u_group)
        total_loglik = total_loglik + jnp.sum(posts.marginal_loglik)
        total_bins += y_group.shape[0] * y_group.shape[1]
    return -total_loglik / total_bins


def learn_poisson_lds(y_train, state_dim, key, learning_steps=100, learning_rate=1e-2, verbose=False):
    """Fit a Poisson LDS to equal-length spike-count trials.

    Args:
        y_train: Count array shaped (n_trials, n_bins, n_neurons).
        state_dim: Number of latent dimensions.
        key: JAX PRNG key used for parameter initialization.
        learning_steps: Number of Adam updates.
        learning_rate: Adam step size.
        verbose: If True, print periodic loss values.

    Returns:
        Tuple of trainable params, Dynamax params, and marginal log-likelihood trace.
    """
    optimizer = optax.adam(learning_rate)
    trainable_params = initialize_trainable_params(y_train, state_dim, key)
    opt_state = optimizer.init(trainable_params)

    @jax.jit
    def step(trainable_params, opt_state):
        """Run one Adam update for current params and optimizer state."""
        loss, grads = jax.value_and_grad(poisson_lds_loss)(trainable_params, y_train)
        updates, opt_state = optimizer.update(grads, opt_state, trainable_params)
        trainable_params = optax.apply_updates(trainable_params, updates)
        return trainable_params, opt_state, loss

    losses = []
    for i in range(learning_steps):
        trainable_params, opt_state, loss = step(trainable_params, opt_state)
        loss_value = float(loss)
        if not np.isfinite(loss_value):
            raise FloatingPointError(f"non-finite Poisson LDS loss at step {i + 1}: {loss_value}")
        losses.append(loss_value)
        if verbose and (i == 0 or (i + 1) % 10 == 0 or i + 1 == learning_steps):
            print(f"step {i + 1:03d}: loss={loss_value:.3f}")

    training_log_likelihoods = -np.asarray(losses) * y_train.shape[1]
    return trainable_params, make_poisson_lds_params(trainable_params), training_log_likelihoods


def learn_poisson_lds_with_inputs(
    y_train,
    u_train,
    state_dim,
    key,
    learning_steps=100,
    learning_rate=1e-2,
    verbose=False,
):
    """Fit a controlled Poisson LDS to equal-length spike-count trials.

    Args:
        y_train: Count array shaped (n_trials, n_bins, n_neurons).
        u_train: Input array shaped (n_trials, n_bins, input_dim).
        state_dim: Number of latent dimensions.
        key: JAX PRNG key used for parameter initialization.
        learning_steps: Number of Adam updates.
        learning_rate: Adam step size.
        verbose: If True, print periodic loss values.

    Returns:
        Tuple of trainable params, Dynamax params, and marginal log-likelihood trace.
    """
    optimizer = optax.adam(learning_rate)
    trainable_params = initialize_trainable_input_params(y_train, u_train, state_dim, key)
    opt_state = optimizer.init(trainable_params)

    @jax.jit
    def step(trainable_params, opt_state):
        """Run one Adam update for current controlled params and optimizer state."""
        loss, grads = jax.value_and_grad(poisson_lds_input_loss)(trainable_params, y_train, u_train)
        updates, opt_state = optimizer.update(grads, opt_state, trainable_params)
        trainable_params = optax.apply_updates(trainable_params, updates)
        return trainable_params, opt_state, loss

    losses = []
    for i in range(learning_steps):
        trainable_params, opt_state, loss = step(trainable_params, opt_state)
        loss_value = float(loss)
        if not np.isfinite(loss_value):
            raise FloatingPointError(f"non-finite controlled Poisson LDS loss at step {i + 1}: {loss_value}")
        losses.append(loss_value)
        if verbose and (i == 0 or (i + 1) % 10 == 0 or i + 1 == learning_steps):
            print(f"controlled step {i + 1:03d}: loss={loss_value:.3f}")

    training_log_likelihoods = -np.asarray(losses) * y_train.shape[1]
    return trainable_params, make_poisson_lds_input_params(trainable_params), training_log_likelihoods


def learn_poisson_lds_grouped(y_train, state_dim, key, learning_steps=100, learning_rate=1e-2, verbose=False):
    """Fit a Poisson LDS to variable-length spike-count trials.

    Args:
        y_train: List of arrays shaped (n_bins_i, n_neurons).
        state_dim: Number of latent dimensions.
        key: JAX PRNG key used for parameter initialization.
        learning_steps: Number of Adam updates.
        learning_rate: Adam step size.
        verbose: If True, print periodic loss values.

    Returns:
        Tuple of trainable params, Dynamax params, and per-bin log-likelihood trace.
    """
    grouped_y_train = stack_trials_by_time_length(y_train)
    optimizer = optax.adam(learning_rate)
    trainable_params = initialize_trainable_params_from_trials(y_train, state_dim, key)
    opt_state = optimizer.init(trainable_params)

    @jax.jit
    def step(trainable_params, opt_state):
        """Run one Adam update for current params and optimizer state."""
        loss, grads = jax.value_and_grad(poisson_lds_grouped_loss)(trainable_params, grouped_y_train)
        updates, opt_state = optimizer.update(grads, opt_state, trainable_params)
        trainable_params = optax.apply_updates(trainable_params, updates)
        return trainable_params, opt_state, loss

    losses = []
    for i in range(learning_steps):
        trainable_params, opt_state, loss = step(trainable_params, opt_state)
        loss_value = float(loss)
        if not np.isfinite(loss_value):
            raise FloatingPointError(f"non-finite pooled Poisson LDS loss at step {i + 1}: {loss_value}")
        losses.append(loss_value)
        if verbose and (i == 0 or (i + 1) % 10 == 0 or i + 1 == learning_steps):
            print(f"pooled step {i + 1:03d}: neg loglik/bin={loss_value:.3f}")

    training_log_likelihoods_per_bin = -np.asarray(losses)
    return trainable_params, make_poisson_lds_params(trainable_params), training_log_likelihoods_per_bin


def learn_poisson_lds_with_inputs_grouped(
    y_train,
    u_train,
    state_dim,
    key,
    learning_steps=100,
    learning_rate=1e-2,
    verbose=False,
):
    """Fit a controlled Poisson LDS to variable-length spike-count trials.

    Args:
        y_train: List of arrays shaped (n_bins_i, n_neurons).
        u_train: List of arrays shaped (n_bins_i, input_dim).
        state_dim: Number of latent dimensions.
        key: JAX PRNG key used for parameter initialization.
        learning_steps: Number of Adam updates.
        learning_rate: Adam step size.
        verbose: If True, print periodic loss values.

    Returns:
        Tuple of trainable params, Dynamax params, and per-bin log-likelihood trace.
    """
    grouped_yu_train = stack_trial_pairs_by_time_length(y_train, u_train)
    optimizer = optax.adam(learning_rate)
    trainable_params = initialize_trainable_input_params_from_trials(y_train, u_train, state_dim, key)
    opt_state = optimizer.init(trainable_params)

    @jax.jit
    def step(trainable_params, opt_state):
        """Run one Adam update for current controlled params and optimizer state."""
        loss, grads = jax.value_and_grad(poisson_lds_input_grouped_loss)(trainable_params, grouped_yu_train)
        updates, opt_state = optimizer.update(grads, opt_state, trainable_params)
        trainable_params = optax.apply_updates(trainable_params, updates)
        return trainable_params, opt_state, loss

    losses = []
    for i in range(learning_steps):
        trainable_params, opt_state, loss = step(trainable_params, opt_state)
        loss_value = float(loss)
        if not np.isfinite(loss_value):
            raise FloatingPointError(f"non-finite pooled controlled Poisson LDS loss at step {i + 1}: {loss_value}")
        losses.append(loss_value)
        if verbose and (i == 0 or (i + 1) % 10 == 0 or i + 1 == learning_steps):
            print(f"pooled controlled step {i + 1:03d}: neg loglik/bin={loss_value:.3f}")

    training_log_likelihoods_per_bin = -np.asarray(losses)
    return trainable_params, make_poisson_lds_input_params(trainable_params), training_log_likelihoods_per_bin


def infer_latents(params, trials):
    """Infer smoothed latent means for each trial.

    Args:
        params: ParamsGGSSM returned by make_poisson_lds_params.
        trials: List of count arrays shaped (n_bins_i, n_neurons).

    Returns:
        List of arrays shaped (n_bins_i, state_dim), one per input trial.
    """
    return [np.asarray(cmgs(params, EKFIntegrals(), jnp.asarray(y)).smoothed_means) for y in trials]


def infer_latents_grouped(params, trials):
    """Infer latents for variable-length trials while preserving input order.

    Args:
        params: ParamsGGSSM returned by make_poisson_lds_params.
        trials: List of count arrays shaped (n_bins_i, n_neurons).

    Returns:
        List of arrays shaped (n_bins_i, state_dim), one per input trial.
    """
    groups = {}
    for i, y in enumerate(trials):
        groups.setdefault(y.shape[0], []).append((i, y))
    latents = [None] * len(trials)
    for n_bins in sorted(groups):
        indices, ys = zip(*groups[n_bins])
        y_stack = jnp.asarray(np.stack(ys), dtype=jnp.float64)
        posts = vmap(lambda y: cmgs(params, EKFIntegrals(), y))(y_stack)
        z_stack = np.asarray(posts.smoothed_means)
        for local_i, trial_i in enumerate(indices):
            latents[trial_i] = z_stack[local_i]
    return latents


def infer_latents_with_inputs(params, y_trials, u_trials):
    """Infer latents for paired count/input trials while preserving input order.

    Args:
        params: ParamsGGSSM returned by make_poisson_lds_input_params.
        y_trials: List of count arrays shaped (n_bins_i, n_neurons).
        u_trials: List of input arrays shaped (n_bins_i, input_dim).

    Returns:
        List of arrays shaped (n_bins_i, state_dim), one per input trial.
    """
    stack_trial_pairs_by_time_length(y_trials, u_trials)
    groups = {}
    for i, (y, u) in enumerate(zip(y_trials, u_trials)):
        groups.setdefault(y.shape[0], []).append((i, y, u))

    latents = [None] * len(y_trials)
    for n_bins in sorted(groups):
        entries = groups[n_bins]
        indices = [entry[0] for entry in entries]
        y_stack = jnp.asarray(np.stack([entry[1] for entry in entries]), dtype=jnp.float64)
        u_stack = jnp.asarray(np.stack([entry[2] for entry in entries]), dtype=jnp.float64)
        posts = vmap(lambda y, u: cmgs(params, EKFIntegrals(), y, inputs=u))(y_stack, u_stack)
        z_stack = np.asarray(posts.smoothed_means)
        for local_i, trial_i in enumerate(indices):
            latents[trial_i] = z_stack[local_i]
    return latents


def predict_expected_counts(params, trainable_params, trials):
    """Predict expected counts for each trial from smoothed latent states.

    Args:
        params: ParamsGGSSM used for latent inference.
        trainable_params: PoissonLDSParams used to map latents to rates.
        trials: List of count arrays shaped (n_bins_i, n_neurons).

    Returns:
        List of nonnegative expected-count arrays matching each trial's shape.
    """
    latents = infer_latents(params, trials)
    expected_counts = [np.asarray(vmap(lambda z: poisson_rate(trainable_params, z))(jnp.asarray(z))) for z in latents]
    for expected, y in zip(expected_counts, trials):
        if expected.shape != y.shape or not np.all(np.isfinite(expected)) or np.any(expected < 0):
            raise FloatingPointError(f"invalid Poisson expected count with shape {expected.shape}; expected {y.shape}")
    return expected_counts


def predict_expected_counts_with_inputs(params, trainable_params, y_trials, u_trials):
    """Predict expected counts for paired count/input trials.

    Args:
        params: ParamsGGSSM used for latent inference.
        trainable_params: PoissonLDSInputParams used to map latents to rates.
        y_trials: List of count arrays shaped (n_bins_i, n_neurons).
        u_trials: List of input arrays shaped (n_bins_i, input_dim).

    Returns:
        List of nonnegative expected-count arrays matching each y_trial shape.
    """
    latents = infer_latents_with_inputs(params, y_trials, u_trials)
    expected_counts = [np.asarray(vmap(lambda z: poisson_rate(trainable_params, z))(jnp.asarray(z))) for z in latents]
    for expected, y in zip(expected_counts, y_trials):
        if expected.shape != y.shape or not np.all(np.isfinite(expected)) or np.any(expected < 0):
            raise FloatingPointError(f"invalid Poisson expected count with shape {expected.shape}; expected {y.shape}")
    return expected_counts


def marginal_log_prob_per_bin(params, trials):
    """Average marginal log probability per time bin across trials.

    Args:
        params: ParamsGGSSM used by the EKF smoother.
        trials: List of count arrays shaped (n_bins_i, n_neurons).

    Returns:
        Python float: mean over trials of marginal log likelihood divided by bins.
    """
    vals = [float(cmgs(params, EKFIntegrals(), jnp.asarray(y)).marginal_loglik) / y.shape[0] for y in trials]
    return float(np.mean(vals))


def marginal_log_prob_per_bin_with_inputs(params, y_trials, u_trials):
    """Average marginal log probability per time bin for paired count/input trials.

    Args:
        params: ParamsGGSSM used by the EKF smoother.
        y_trials: List of count arrays shaped (n_bins_i, n_neurons).
        u_trials: List of input arrays shaped (n_bins_i, input_dim).

    Returns:
        Python float: mean over trials of marginal log likelihood divided by bins.
    """
    stack_trial_pairs_by_time_length(y_trials, u_trials)
    vals = [
        float(cmgs(params, EKFIntegrals(), jnp.asarray(y), inputs=jnp.asarray(u)).marginal_loglik) / y.shape[0]
        for y, u in zip(y_trials, u_trials)
    ]
    return float(np.mean(vals))
