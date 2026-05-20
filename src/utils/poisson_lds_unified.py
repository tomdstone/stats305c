"""Unified Poisson LDS utilities with optional control inputs.

This module is a clean draft of what poisson_lds.py could look like if the
no-input and input-aware paths shared one implementation. The key convention is
that a model with no controls uses zero-width inputs and B shaped
(state_dim, 0), so the same Dynamax calls work for both p(y) and p(y | u).
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
from jax.nn import softplus
import numpy as np
import optax
from tensorflow_probability.substrates.jax.distributions import Poisson

from dynamax.generalized_gaussian_ssm import EKFIntegrals, ParamsGGSSM
from dynamax.generalized_gaussian_ssm import conditional_moments_gaussian_smoother as cmgs

from .preprocessing import stack_trial_pairs_by_time_length

RATE_FLOOR = 1e-4
COV_FLOOR = 1e-4


class PoissonLDSParams(NamedTuple):
    """Trainable Poisson LDS parameters in unconstrained optimizer space.

    B is the control-input matrix. For no-input models, B has shape
    (state_dim, 0), and inputs have shape (..., 0).
    """

    A: jnp.ndarray
    B: jnp.ndarray
    C: jnp.ndarray
    d: jnp.ndarray
    log_q: jnp.ndarray
    m0: jnp.ndarray
    log_s0: jnp.ndarray


def inverse_softplus(x):
    """Map positive values to unconstrained softplus inputs."""
    return jnp.log(jnp.expm1(jnp.maximum(jnp.asarray(x), 1e-6)))


def positive_diag(log_diag):
    """Convert unconstrained covariance diagonal parameters to positive values."""
    return softplus(log_diag) + COV_FLOOR


def _as_stacked_counts(y_train):
    """Convert stacked spike-count trials to a float64 JAX array."""
    y_train = jnp.asarray(y_train, dtype=jnp.float64)
    if y_train.ndim != 3:
        raise ValueError("y_train must have shape (n_trials, n_bins, n_neurons)")
    return y_train


def _as_stacked_inputs(y_train, u_train=None):
    """Return inputs shaped (n_trials, n_bins, input_dim), using input_dim 0 by default."""
    if u_train is None:
        return jnp.zeros(y_train.shape[:2] + (0,), dtype=y_train.dtype)
    u_train = jnp.asarray(u_train, dtype=jnp.float64)
    if u_train.ndim != 3:
        raise ValueError("u_train must have shape (n_trials, n_bins, input_dim)")
    if y_train.shape[:2] != u_train.shape[:2]:
        raise ValueError("y_train and u_train must have matching trial and time dimensions")
    return u_train


def _as_trial_list(trials, name):
    """Convert a trial sequence to a nonempty list of float64 NumPy arrays."""
    trials = [np.asarray(trial, dtype=np.float64) for trial in trials]
    if len(trials) == 0:
        raise ValueError(f"{name} must contain at least one trial")
    if any(trial.ndim != 2 for trial in trials):
        raise ValueError(f"each {name} trial must be two-dimensional")
    return trials


def _as_input_trial_list(y_trials, u_trials=None):
    """Return input trials aligned to y_trials, using input_dim 0 by default."""
    if u_trials is None:
        return [np.zeros((y.shape[0], 0), dtype=np.float64) for y in y_trials]
    return _as_trial_list(u_trials, "u_trials")


def poisson_rate(trainable_params, z):
    """Compute positive Poisson rates for one latent state."""
    return softplus(trainable_params.C @ z + trainable_params.d) + RATE_FLOOR


def make_poisson_lds_params(trainable_params):
    """Build Dynamax GGSSM parameters for a Poisson LDS with unified inputs.

    The returned functions expect inputs. For no-input models, pass zero-width
    inputs shaped (n_bins, 0).
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


def initialize_trainable_params(y_train, state_dim, key, u_train=None):
    """Initialize trainable parameters from equal-length spike-count trials."""
    y_train = _as_stacked_counts(y_train)
    u_train = _as_stacked_inputs(y_train, u_train)
    emission_dim = y_train.shape[-1]
    input_dim = u_train.shape[-1]
    mean_counts = jnp.mean(y_train, axis=(0, 1))
    baseline_rate = jnp.maximum(mean_counts - RATE_FLOOR, 1e-3)
    return PoissonLDSParams(
        A=0.95 * jnp.eye(state_dim),
        B=jnp.zeros((state_dim, input_dim)),
        C=0.01 * jr.normal(key, (emission_dim, state_dim)),
        d=inverse_softplus(baseline_rate),
        log_q=inverse_softplus(0.1 * jnp.ones(state_dim)),
        m0=jnp.zeros(state_dim),
        log_s0=inverse_softplus(jnp.ones(state_dim)),
    )


def initialize_trainable_params_from_trials(y_trials, state_dim, key, u_trials=None):
    """Initialize trainable parameters from variable-length spike-count trials."""
    y_trials = _as_trial_list(y_trials, "y_trials")
    u_trials = _as_input_trial_list(y_trials, u_trials)
    stack_trial_pairs_by_time_length(y_trials, u_trials)
    emission_dim = y_trials[0].shape[-1]
    input_dim = u_trials[0].shape[-1]
    mean_counts = jnp.asarray(np.concatenate(y_trials, axis=0).mean(axis=0), dtype=jnp.float64)
    baseline_rate = jnp.maximum(mean_counts - RATE_FLOOR, 1e-3)
    return PoissonLDSParams(
        A=0.95 * jnp.eye(state_dim),
        B=jnp.zeros((state_dim, input_dim)),
        C=0.01 * jr.normal(key, (emission_dim, state_dim)),
        d=inverse_softplus(baseline_rate),
        log_q=inverse_softplus(0.1 * jnp.ones(state_dim)),
        m0=jnp.zeros(state_dim),
        log_s0=inverse_softplus(jnp.ones(state_dim)),
    )


def poisson_lds_loss(trainable_params, y_train, u_train):
    """Compute negative marginal log likelihood per bin for stacked trials."""
    params = make_poisson_lds_params(trainable_params)
    posts = vmap(lambda y, u: cmgs(params, EKFIntegrals(), y, inputs=u))(y_train, u_train)
    return -jnp.mean(posts.marginal_loglik) / y_train.shape[1]


def poisson_lds_grouped_loss(trainable_params, grouped_yu_train):
    """Compute negative marginal log likelihood per bin for grouped trials."""
    params = make_poisson_lds_params(trainable_params)
    total_loglik = 0.0
    total_bins = 0
    for y_group, u_group in grouped_yu_train:
        posts = vmap(lambda y, u: cmgs(params, EKFIntegrals(), y, inputs=u))(y_group, u_group)
        total_loglik = total_loglik + jnp.sum(posts.marginal_loglik)
        total_bins += y_group.shape[0] * y_group.shape[1]
    return -total_loglik / total_bins


def _run_adam_learning(
    trainable_params,
    loss_fn,
    loss_args,
    make_params_fn,
    learning_steps,
    learning_rate,
    trace_scale,
    verbose=False,
    log_prefix="step",
):
    """Run the shared Adam loop for Poisson LDS learning."""
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(trainable_params)

    @jax.jit
    def step(trainable_params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(trainable_params, *loss_args)
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
            print(f"{log_prefix} {i + 1:03d}: neg loglik/bin={loss_value:.3f}")

    likelihood_trace = -np.asarray(losses) * trace_scale
    return trainable_params, make_params_fn(trainable_params), likelihood_trace


def learn_poisson_lds(
    y_train,
    state_dim,
    key,
    u_train=None,
    learning_steps=100,
    learning_rate=1e-2,
    verbose=False,
):
    """Fit a Poisson LDS to equal-length trials, optionally conditioned on inputs.

    Args:
        y_train: Count array shaped (n_trials, n_bins, n_neurons).
        state_dim: Number of latent dimensions.
        key: JAX PRNG key used for parameter initialization.
        u_train: Optional input array shaped (n_trials, n_bins, input_dim). If
            omitted, a no-input model is represented with input_dim 0.
        learning_steps: Number of Adam updates.
        learning_rate: Adam step size.
        verbose: If True, print periodic loss values.

    Returns:
        Tuple of trainable params, Dynamax params, and marginal log-likelihood trace.
    """
    y_train = _as_stacked_counts(y_train)
    u_train = _as_stacked_inputs(y_train, u_train)
    trainable_params = initialize_trainable_params(y_train, state_dim, key, u_train=u_train)
    return _run_adam_learning(
        trainable_params,
        poisson_lds_loss,
        (y_train, u_train),
        make_poisson_lds_params,
        learning_steps,
        learning_rate,
        trace_scale=y_train.shape[1],
        verbose=verbose,
        log_prefix="poisson",
    )


def learn_poisson_lds_grouped(
    y_train,
    state_dim,
    key,
    u_train=None,
    learning_steps=100,
    learning_rate=1e-2,
    verbose=False,
):
    """Fit a Poisson LDS to variable-length trials, optionally conditioned on inputs."""
    y_train = _as_trial_list(y_train, "y_train")
    u_train = _as_input_trial_list(y_train, u_train)
    grouped_yu_train = stack_trial_pairs_by_time_length(y_train, u_train)
    trainable_params = initialize_trainable_params_from_trials(y_train, state_dim, key, u_trials=u_train)
    return _run_adam_learning(
        trainable_params,
        poisson_lds_grouped_loss,
        (grouped_yu_train,),
        make_poisson_lds_params,
        learning_steps,
        learning_rate,
        trace_scale=1.0,
        verbose=verbose,
        log_prefix="pooled poisson",
    )


def infer_latents(params, trials, u_trials=None):
    """Infer smoothed latent means for each trial, optionally conditioned on inputs."""
    trials = _as_trial_list(trials, "trials")
    u_trials = _as_input_trial_list(trials, u_trials)
    stack_trial_pairs_by_time_length(trials, u_trials)
    return [
        np.asarray(cmgs(params, EKFIntegrals(), jnp.asarray(y), inputs=jnp.asarray(u)).smoothed_means)
        for y, u in zip(trials, u_trials)
    ]


def infer_latents_grouped(params, trials, u_trials=None):
    """Infer latents for variable-length trials while preserving input order."""
    trials = _as_trial_list(trials, "trials")
    u_trials = _as_input_trial_list(trials, u_trials)
    groups = {}
    for i, (y, u) in enumerate(zip(trials, u_trials)):
        groups.setdefault(y.shape[0], []).append((i, y, u))

    latents = [None] * len(trials)
    for n_bins in sorted(groups):
        indices, ys, us = zip(*groups[n_bins])
        y_stack = jnp.asarray(np.stack(ys), dtype=jnp.float64)
        u_stack = jnp.asarray(np.stack(us), dtype=jnp.float64)
        posts = vmap(lambda y, u: cmgs(params, EKFIntegrals(), y, inputs=u))(y_stack, u_stack)
        z_stack = np.asarray(posts.smoothed_means)
        for local_i, trial_i in enumerate(indices):
            latents[trial_i] = z_stack[local_i]
    return latents


def predict_expected_counts(params, trainable_params, trials, u_trials=None):
    """Predict expected counts from smoothed latent states."""
    latents = infer_latents(params, trials, u_trials=u_trials)
    expected_counts = [np.asarray(vmap(lambda z: poisson_rate(trainable_params, z))(jnp.asarray(z))) for z in latents]
    for expected, y in zip(expected_counts, trials):
        if expected.shape != y.shape or not np.all(np.isfinite(expected)) or np.any(expected < 0):
            raise FloatingPointError(f"invalid Poisson expected count with shape {expected.shape}; expected {y.shape}")
    return expected_counts


def marginal_log_prob_per_bin(params, trials, u_trials=None):
    """Average marginal log probability per time bin across trials."""
    trials = _as_trial_list(trials, "trials")
    u_trials = _as_input_trial_list(trials, u_trials)
    vals = [
        float(cmgs(params, EKFIntegrals(), jnp.asarray(y), inputs=jnp.asarray(u)).marginal_loglik) / y.shape[0]
        for y, u in zip(trials, u_trials)
    ]
    return float(np.mean(vals))
