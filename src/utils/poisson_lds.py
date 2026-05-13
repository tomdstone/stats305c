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
from .preprocessing import stack_trials_by_time_length

RATE_FLOOR = 1e-4
COV_FLOOR = 1e-4

class PoissonLDSParams(NamedTuple):
    A: jnp.ndarray
    C: jnp.ndarray
    d: jnp.ndarray
    log_q: jnp.ndarray
    m0: jnp.ndarray
    log_s0: jnp.ndarray


def inverse_softplus(x):
    return jnp.log(jnp.expm1(jnp.maximum(jnp.asarray(x), 1e-6)))


def positive_diag(log_diag):
    return softplus(log_diag) + COV_FLOOR


def poisson_rate(trainable_params, z):
    return softplus(trainable_params.C @ z + trainable_params.d) + RATE_FLOOR


def make_poisson_lds_params(trainable_params):
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


def initialize_trainable_params(y_train, state_dim, key):
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


def poisson_lds_loss(trainable_params, y_train):
    params = make_poisson_lds_params(trainable_params)
    posts = vmap(lambda y: cmgs(params, EKFIntegrals(), y))(y_train)
    return -jnp.mean(posts.marginal_loglik) / y_train.shape[1]

def poisson_lds_grouped_loss(trainable_params, grouped_y_train):
    params = make_poisson_lds_params(trainable_params)
    total_loglik = 0.0
    total_bins = 0
    for y_group in grouped_y_train:
        posts = vmap(lambda y: cmgs(params, EKFIntegrals(), y))(y_group)
        total_loglik = total_loglik + jnp.sum(posts.marginal_loglik)
        total_bins += y_group.shape[0] * y_group.shape[1]
    return -total_loglik / total_bins

def learn_poisson_lds(y_train, state_dim, key, learning_steps=100, learning_rate=1e-2, verbose=False):
    optimizer = optax.adam(learning_rate)
    trainable_params = initialize_trainable_params(y_train, state_dim, key)
    opt_state = optimizer.init(trainable_params)

    @jax.jit
    def step(trainable_params, opt_state):
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

def learn_poisson_lds_grouped(y_train, state_dim, key, learning_steps=100, learning_rate=1e-2, verbose=False):
    grouped_y_train = stack_trials_by_time_length(y_train)
    optimizer = optax.adam(learning_rate)
    trainable_params = initialize_trainable_params_from_trials(y_train, state_dim, key)
    opt_state = optimizer.init(trainable_params)

    @jax.jit
    def step(trainable_params, opt_state):
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

def infer_latents(params, trials):
    return [np.asarray(cmgs(params, EKFIntegrals(), jnp.asarray(y)).smoothed_means) for y in trials]

def infer_latents_grouped(params, trials):
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

def predict_expected_counts(params, trainable_params, trials):
    latents = infer_latents(params, trials)
    expected_counts = [np.asarray(vmap(lambda z: poisson_rate(trainable_params, z))(jnp.asarray(z))) for z in latents]
    for expected, y in zip(expected_counts, trials):
        if expected.shape != y.shape or not np.all(np.isfinite(expected)) or np.any(expected < 0):
            raise FloatingPointError(f"invalid Poisson expected count with shape {expected.shape}; expected {y.shape}")
    return expected_counts

def marginal_log_prob_per_bin(params, trials):
    vals = [float(cmgs(params, EKFIntegrals(), jnp.asarray(y)).marginal_loglik) / y.shape[0] for y in trials]
    return float(np.mean(vals))
