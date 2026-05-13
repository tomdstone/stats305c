import numpy as np
import jax.numpy as jnp
from scipy.ndimage import gaussian_filter1d

def _trim_to_bins(arr, bin_ms):
    n_bins = arr.shape[-1] // bin_ms
    if n_bins < 2:
        raise ValueError(f"trial is too short for {bin_ms} ms bins")
    return arr[..., : n_bins * bin_ms], n_bins


def bin_and_smooth_trials(trials, bin_ms=50, smooth_sigma_bins=1.0, kind="spikes"):
    processed = []
    for trial in trials:
        arr = np.asarray(trial, dtype=float)
        trimmed, n_bins = _trim_to_bins(arr, bin_ms)
        if kind == "spikes":
            binned = trimmed.reshape(arr.shape[0], n_bins, bin_ms).sum(axis=2).T
        elif kind == "force":
            binned = trimmed.reshape(arr.shape[0], n_bins, bin_ms).mean(axis=2).T
        else:
            raise ValueError("kind must be 'spikes' or 'force'")
        if smooth_sigma_bins > 0:
            binned = gaussian_filter1d(binned, sigma=smooth_sigma_bins, axis=0, mode="nearest")
        processed.append(binned.astype(np.float64))
    return processed


def split_condition_trials(condition, cond, test_fraction=0.2, seed=7):
    idx = np.flatnonzero(condition == cond)
    rng = np.random.default_rng(seed + int(cond))
    idx = rng.permutation(idx)
    n_test = max(1, int(round(test_fraction * len(idx))))
    return np.sort(idx[n_test:]), np.sort(idx[:n_test])


def pooled_split_indices(condition, test_fraction=0.2, seed=7):
    train_parts = []
    test_parts = []
    for cond in sorted(np.unique(condition)):
        train_idx, test_idx = split_condition_trials(condition, cond, test_fraction, seed)
        train_parts.append(np.asarray(train_idx, dtype=int))
        test_parts.append(np.asarray(test_idx, dtype=int))
    train_idx = np.sort(np.concatenate(train_parts))
    test_idx = np.sort(np.concatenate(test_parts))
    if np.intersect1d(train_idx, test_idx).size:
        raise ValueError("pooled train/test indices overlap")
    return train_idx, test_idx


def standardize_train_test(train_trials, test_trials, eps=1e-6):
    stacked = np.concatenate(train_trials, axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std = np.where(std < eps, 1.0, std)
    train_z = [(x - mean) / std for x in train_trials]
    test_z = [(x - mean) / std for x in test_trials]
    return train_z, test_z, mean, std


def stack_trials(trials):
    return jnp.asarray(np.stack(trials), dtype=jnp.float64)

def stack_trials_by_time_length(trials):
    grouped = {}
    for trial in trials:
        arr = np.asarray(trial, dtype=np.float64)
        grouped.setdefault(arr.shape[0], []).append(arr)
    return tuple(jnp.asarray(np.stack(grouped[n_bins]), dtype=jnp.float64) for n_bins in sorted(grouped))
