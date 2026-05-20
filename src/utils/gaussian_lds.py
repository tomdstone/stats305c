"""Gaussian LDS utilities with optional Dynamax input support."""

from functools import partial

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import vmap
from jax.tree_util import tree_map

from dynamax.linear_gaussian_ssm import LinearGaussianConjugateSSM

from .preprocessing import stack_trial_pairs_by_time_length, stack_trials, stack_trials_by_time_length


def _as_stacked_observations(y_train):
    """Convert stacked observation trials to a float64 JAX array."""
    y_train = jnp.asarray(y_train, dtype=jnp.float64)
    if y_train.ndim != 3:
        raise ValueError("y_train must have shape (n_trials, n_bins, n_features)")
    return y_train


def _as_stacked_inputs(y_train, u_train=None):
    """Convert optional stacked input trials to a float64 JAX array."""
    if u_train is None:
        return None
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
    feature_dim = trials[0].shape[-1]
    if any(trial.shape[-1] != feature_dim for trial in trials):
        raise ValueError(f"all {name} trials must have the same feature dimension")
    return trials


def _as_input_trial_list(y_trials, u_trials=None):
    """Convert optional input trials and validate alignment with observation trials."""
    if u_trials is None:
        return None
    u_trials = _as_trial_list(u_trials, "u_trials")
    if len(y_trials) != len(u_trials):
        raise ValueError("y_trials and u_trials must have the same number of trials")
    for y, u in zip(y_trials, u_trials):
        if y.shape[0] != u.shape[0]:
            raise ValueError("paired y and u trials must have the same number of time bins")
    return u_trials


def initialize_gaussian_lds(y_train, state_dim, key, u_train=None):
    """Create and initialize a LinearGaussianConjugateSSM.

    Args:
        y_train: Observation array shaped (n_trials, n_bins, n_features).
        state_dim: Number of latent dimensions.
        key: JAX PRNG key used for Dynamax initialization.
        u_train: Optional input array shaped (n_trials, n_bins, input_dim).

    Returns:
        Tuple of model, params, and parameter properties.
    """
    y_train = _as_stacked_observations(y_train)
    u_train = _as_stacked_inputs(y_train, u_train)
    emission_dim = y_train.shape[-1]
    input_dim = 0 if u_train is None else u_train.shape[-1]
    model = LinearGaussianConjugateSSM(state_dim=state_dim, emission_dim=emission_dim, input_dim=input_dim)
    params, props = model.initialize(
        key=key,
        initial_covariance=jnp.eye(state_dim),
        dynamics_covariance=0.1 * jnp.eye(state_dim),
        emission_covariance=jnp.eye(emission_dim),
    )
    return model, params, props


def _fit_em_grouped_by_time_length(model, params, props, grouped_emissions, grouped_inputs=None, num_iters=50, verbose=True):
    """Fit EM over groups of equal-length trials, optionally with grouped inputs."""
    log_probs = []
    m_step_state = model.initialize_m_step_state(params, props)

    for i in range(num_iters):
        batch_stats_parts = []
        total_ll = 0.0
        if grouped_inputs is None:
            for emissions in grouped_emissions:
                batch_stats, lls = vmap(partial(model.e_step, params))(emissions)
                batch_stats_parts.append(batch_stats)
                total_ll = total_ll + jnp.sum(lls)
        else:
            for emissions, inputs in zip(grouped_emissions, grouped_inputs):
                batch_stats, lls = vmap(lambda y, u: model.e_step(params, y, inputs=u))(emissions, inputs)
                batch_stats_parts.append(batch_stats)
                total_ll = total_ll + jnp.sum(lls)

        batch_stats = tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *batch_stats_parts)
        marginal_logprob = model.log_prior(params) + total_ll
        params, m_step_state = model.m_step(params, props, batch_stats, m_step_state)
        log_probs.append(marginal_logprob)
        if verbose:
            print(f"EM iter {i + 1}/{num_iters}: {float(marginal_logprob):.3f}", flush=True)

    return params, jnp.asarray(log_probs)


def learn_gaussian_lds(y_train, state_dim, key, u_train=None, em_iters=100, verbose=False):
    """Fit a Gaussian LDS to equal-length trials, optionally conditioned on inputs.

    Args:
        y_train: Observation array shaped (n_trials, n_bins, n_features).
        state_dim: Number of latent dimensions.
        key: JAX PRNG key used for Dynamax initialization.
        u_train: Optional input array shaped (n_trials, n_bins, input_dim).
        em_iters: Number of EM iterations.
        verbose: If True, show Dynamax EM progress.

    Returns:
        Tuple of model, fitted params, and EM log-likelihood trace.
    """
    y_train = _as_stacked_observations(y_train)
    u_train = _as_stacked_inputs(y_train, u_train)
    model, params, props = initialize_gaussian_lds(y_train, state_dim, key, u_train=u_train)
    params, lls = model.fit_em(params, props, y_train, inputs=u_train, num_iters=em_iters, verbose=verbose)
    lls = np.asarray(lls, dtype=float)
    if not np.all(np.isfinite(lls)):
        raise FloatingPointError("Gaussian LDS produced non-finite EM log likelihoods")
    return model, params, lls


def learn_gaussian_lds_grouped(y_train, state_dim, key, u_train=None, em_iters=100, verbose=False):
    """Fit a Gaussian LDS to variable-length trials, optionally conditioned on inputs.

    Args:
        y_train: List of arrays shaped (n_bins_i, n_features).
        state_dim: Number of latent dimensions.
        key: JAX PRNG key used for Dynamax initialization.
        u_train: Optional list of arrays shaped (n_bins_i, input_dim).
        em_iters: Number of EM iterations.
        verbose: If True, print per-iteration log probabilities.

    Returns:
        Tuple of model, fitted params, and EM log-likelihood trace.
    """
    y_train = _as_trial_list(y_train, "y_train")
    u_train = _as_input_trial_list(y_train, u_train)
    init_y = stack_trials([y_train[0]])
    init_u = None if u_train is None else stack_trials([u_train[0]])
    model, params, props = initialize_gaussian_lds(init_y, state_dim, key, u_train=init_u)

    if u_train is None:
        grouped_y = stack_trials_by_time_length(y_train)
        grouped_u = None
    else:
        grouped_pairs = stack_trial_pairs_by_time_length(y_train, u_train)
        grouped_y = tuple(y_group for y_group, _ in grouped_pairs)
        grouped_u = tuple(u_group for _, u_group in grouped_pairs)

    params, lls = _fit_em_grouped_by_time_length(
        model,
        params,
        props,
        grouped_y,
        grouped_inputs=grouped_u,
        num_iters=em_iters,
        verbose=verbose,
    )
    lls = np.asarray(lls, dtype=float)
    if not np.all(np.isfinite(lls)):
        raise FloatingPointError("grouped Gaussian LDS produced non-finite EM log likelihoods")
    return model, params, lls


def infer_latents(model, params, trials, u_trials=None):
    """Infer smoothed latent means for each trial."""
    trials = _as_trial_list(trials, "trials")
    u_trials = _as_input_trial_list(trials, u_trials)
    if u_trials is None:
        return [np.asarray(model.smoother(params, jnp.asarray(y)).smoothed_means) for y in trials]
    return [
        np.asarray(model.smoother(params, jnp.asarray(y), inputs=jnp.asarray(u)).smoothed_means)
        for y, u in zip(trials, u_trials)
    ]


def infer_latents_grouped(model, params, trials, u_trials=None):
    """Infer latents for variable-length trials while preserving input order."""
    trials = _as_trial_list(trials, "trials")
    u_trials = _as_input_trial_list(trials, u_trials)
    groups = {}
    if u_trials is None:
        for i, y in enumerate(trials):
            groups.setdefault(y.shape[0], []).append((i, y, None))
    else:
        for i, (y, u) in enumerate(zip(trials, u_trials)):
            groups.setdefault(y.shape[0], []).append((i, y, u))

    latents = [None] * len(trials)
    for n_bins in sorted(groups):
        entries = groups[n_bins]
        indices = [entry[0] for entry in entries]
        y_stack = jnp.asarray(np.stack([entry[1] for entry in entries]), dtype=jnp.float64)
        if u_trials is None:
            posts = vmap(lambda y: model.smoother(params, y))(y_stack)
        else:
            u_stack = jnp.asarray(np.stack([entry[2] for entry in entries]), dtype=jnp.float64)
            posts = vmap(lambda y, u: model.smoother(params, y, inputs=u))(y_stack, u_stack)
        z_stack = np.asarray(posts.smoothed_means)
        for local_i, trial_i in enumerate(indices):
            latents[trial_i] = z_stack[local_i]
    return latents


def _reconstruct_from_latents(params, latents, inputs=None):
    """Map smoothed latent means and optional inputs to Gaussian emission means."""
    reconstructed = latents @ params.emissions.weights.T
    if params.emissions.input_weights is not None and params.emissions.input_weights.shape[-1] > 0:
        if inputs is None:
            raise ValueError("inputs are required for reconstruction with nonzero emission input weights")
        reconstructed = reconstructed + inputs @ params.emissions.input_weights.T
    if params.emissions.bias is not None:
        reconstructed = reconstructed + params.emissions.bias
    return reconstructed


def reconstruct_observations(model, params, trials, u_trials=None):
    """Predict posterior mean observations for each trial."""
    trials = _as_trial_list(trials, "trials")
    u_trials = _as_input_trial_list(trials, u_trials)
    latents = infer_latents(model, params, trials, u_trials=u_trials)
    if u_trials is None:
        u_trials = [None] * len(trials)
    reconstructions = [
        np.asarray(_reconstruct_from_latents(params, jnp.asarray(z), None if u is None else jnp.asarray(u)))
        for z, u in zip(latents, u_trials)
    ]
    for reconstructed, y in zip(reconstructions, trials):
        if reconstructed.shape != y.shape or not np.all(np.isfinite(reconstructed)):
            raise FloatingPointError(f"invalid Gaussian reconstruction with shape {reconstructed.shape}; expected {y.shape}")
    return reconstructions


def marginal_log_prob_per_bin(model, params, trials, u_trials=None):
    """Average marginal log probability per time bin across trials."""
    trials = _as_trial_list(trials, "trials")
    u_trials = _as_input_trial_list(trials, u_trials)
    if u_trials is None:
        vals = [float(model.marginal_log_prob(params, jnp.asarray(y))) / y.shape[0] for y in trials]
    else:
        vals = [
            float(model.marginal_log_prob(params, jnp.asarray(y), inputs=jnp.asarray(u))) / y.shape[0]
            for y, u in zip(trials, u_trials)
        ]
    return float(np.mean(vals))
