import numpy as np
import jax.numpy as jnp
from scipy.ndimage import gaussian_filter1d


def _trim_to_bins(arr, bin_ms):
    """Trim the time axis to an integer number of bins.

    Args:
        arr: Array whose last axis is raw time samples.
        bin_ms: Positive integer samples per bin.

    Returns:
        Tuple of trimmed array and number of complete bins.
    """
    n_bins = arr.shape[-1] // bin_ms
    if n_bins < 2:
        raise ValueError(f"trial is too short for {bin_ms} ms bins")
    return arr[..., : n_bins * bin_ms], n_bins


def bin_and_smooth_trials(trials, bin_ms=50, smooth_sigma_bins=1.0, kind="spikes"):
    """Bin and optionally smooth raw trial time series.

    Args:
        trials: Iterable of arrays shaped (n_features, n_time_samples).
        bin_ms: Positive integer samples per output bin.
        smooth_sigma_bins: Gaussian smoothing sigma in output-bin units; 0 disables it.
        kind: "spikes" sums samples within bins; "force" averages samples within bins.

    Returns:
        List of float64 arrays shaped (n_bins, n_features).
    """
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
    """Split indices for one condition label.

    Args:
        condition: One-dimensional array of condition labels, one per trial.
        cond: Label value to select; must be int-convertible for seeding.
        test_fraction: Fraction of selected trials assigned to test.
        seed: Base RNG seed; cond is added for per-condition reproducibility.

    Returns:
        Sorted train and test index arrays.
    """
    idx = np.flatnonzero(condition == cond)
    rng = np.random.default_rng(seed + int(cond))
    idx = rng.permutation(idx)
    n_test = max(1, int(round(test_fraction * len(idx))))
    return np.sort(idx[n_test:]), np.sort(idx[:n_test])


def pooled_split_indices(condition, test_fraction=0.2, seed=7):
    """Create train/test indices with each condition represented.

    Args:
        condition: One-dimensional array of condition labels, one per trial.
        test_fraction: Fraction of each condition's trials assigned to test.
        seed: Base RNG seed passed to split_condition_trials.

    Returns:
        Sorted train and test index arrays with no overlap.
    """
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
    """Z-score train and test trials using train-set statistics.

    Args:
        train_trials: List of arrays shaped (n_bins_i, n_features).
        test_trials: List of arrays shaped (n_bins_i, n_features).
        eps: Features with train std below eps are scaled by 1.0.

    Returns:
        Standardized train trials, standardized test trials, feature means, and scales.
    """
    stacked = np.concatenate(train_trials, axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std = np.where(std < eps, 1.0, std)
    train_z = [(x - mean) / std for x in train_trials]
    test_z = [(x - mean) / std for x in test_trials]
    return train_z, test_z, mean, std


def stack_trials(trials):
    """Stack equal-length trials for batched JAX code.

    Args:
        trials: List of arrays with identical shape (n_bins, n_features).

    Returns:
        float64 JAX array shaped (n_trials, n_bins, n_features).
    """
    return jnp.asarray(np.stack(trials), dtype=jnp.float64)


def stack_trials_by_time_length(trials):
    """Group variable-length trials by number of time bins.

    Args:
        trials: List of arrays shaped (n_bins_i, n_features).

    Returns:
        Tuple of float64 JAX arrays shaped (n_trials_g, n_bins_g, n_features),
        ordered by increasing n_bins_g.
    """
    grouped = {}
    for trial in trials:
        arr = np.asarray(trial, dtype=np.float64)
        grouped.setdefault(arr.shape[0], []).append(arr)
    return tuple(jnp.asarray(np.stack(grouped[n_bins]), dtype=jnp.float64) for n_bins in sorted(grouped))


def stack_trial_pairs_by_time_length(y_trials, u_trials):
    """Group paired observation and input trials by shared time length.

    Args:
        y_trials: List of arrays shaped (n_bins_i, n_observation_features).
        u_trials: List of arrays shaped (n_bins_i, n_input_features).

    Returns:
        Tuple of (y_group, u_group) pairs ordered by increasing n_bins. Each
        y_group is shaped (n_trials_g, n_bins_g, n_observation_features), and
        each u_group is shaped (n_trials_g, n_bins_g, n_input_features).
    """
    if len(y_trials) != len(u_trials):
        raise ValueError("y_trials and u_trials must have the same number of trials")
    if len(y_trials) == 0:
        raise ValueError("at least one trial is required")

    observation_dim = y_trials[0].shape[-1]
    input_dim = u_trials[0].shape[-1]
    grouped = {}
    for y, u in zip(y_trials, u_trials):
        y = np.asarray(y, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)
        if y.ndim != 2 or u.ndim != 2:
            raise ValueError("each y and u trial must be two-dimensional")
        if y.shape[0] != u.shape[0]:
            raise ValueError("paired y and u trials must have the same number of time bins")
        if y.shape[-1] != observation_dim:
            raise ValueError("all y trials must have the same observation dimension")
        if u.shape[-1] != input_dim:
            raise ValueError("all u trials must have the same input dimension")
        grouped.setdefault(y.shape[0], []).append((y, u))

    grouped_pairs = []
    for n_bins in sorted(grouped):
        ys, us = zip(*grouped[n_bins])
        y_group = jnp.asarray(np.stack(ys), dtype=jnp.float64)
        u_group = jnp.asarray(np.stack(us), dtype=jnp.float64)
        grouped_pairs.append((y_group, u_group))
    return tuple(grouped_pairs)
