import numpy as np


def variance_explained(y_true, y_hat):
    """Compute pooled variance explained.

    Args:
        y_true: Ground-truth observations shaped (n_samples, n_features).
        y_hat: Predicted observations with the same shape as y_true.

    Returns:
        Scalar R^2-like variance explained, or nan if y_true is constant.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    sse = np.sum((y_true - y_hat) ** 2)
    sst = np.sum((y_true - y_true.mean(axis=0)) ** 2)
    return np.nan if sst <= 1e-9 else 1.0 - sse / sst


def mean_feature_variance_explained(y_true, y_hat):
    """Compute mean variance explained across nonconstant features.

    Args:
        y_true: Ground-truth observations shaped (n_samples, n_features).
        y_hat: Predicted observations with the same shape as y_true.

    Returns:
        Tuple of mean per-feature variance explained and count of valid features.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    per_feature_sse = np.sum((y_true - y_hat) ** 2, axis=0)
    per_feature_sst = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
    valid = per_feature_sst > 1e-9
    per_feature = np.full(y_true.shape[1], np.nan, dtype=float)
    per_feature[valid] = 1.0 - per_feature_sse[valid] / per_feature_sst[valid]
    return float(np.nanmean(per_feature)), int(valid.sum())


def evaluate_spike_reconstruction(fit, *, predict_observations_fn):
    """Evaluate spike/count reconstruction for a fitted model.

    Args:
        fit: Dict containing y_train and y_test lists of arrays shaped
            (n_bins_i, n_neurons).
        predict_observations_fn: Callable taking (fit, trials) and returning
            predicted arrays with shapes matching trials.

    Returns:
        Dict with train/test predictions and variance-explained metrics.
    """
    train_pred = predict_observations_fn(fit, fit["y_train"])
    test_pred = predict_observations_fn(fit, fit["y_test"])

    y_true = np.concatenate(fit["y_test"], axis=0)
    y_hat = np.concatenate(test_pred, axis=0)
    mean_feature_ve, n_valid = mean_feature_variance_explained(y_true, y_hat)

    return {
        "train_recon": train_pred,
        "test_recon": test_pred,
        "spike_var_explained": float(variance_explained(y_true, y_hat)),
        "spike_var_explained_mean_neuron": mean_feature_ve,
        "spike_var_explained_n_neurons": n_valid,
    }


def evaluate_force_decoding(fit, *, infer_latents_fn, ridge_alphas):
    """Fit and score a ridge decoder from latents to force.

    Args:
        fit: Dict containing y_train, y_test, force_train, and force_test trial lists.
        infer_latents_fn: Callable taking (fit, trials) and returning latent arrays
            shaped (n_bins_i, state_dim).
        ridge_alphas: Candidate RidgeCV regularization strengths.

    Returns:
        Dict with the fitted decoder, test R^2, and selected alpha.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import r2_score

    z_train = np.concatenate(infer_latents_fn(fit, fit["y_train"]), axis=0)
    z_test = np.concatenate(infer_latents_fn(fit, fit["y_test"]), axis=0)
    f_train = np.concatenate(fit["force_train"], axis=0).ravel()
    f_test = np.concatenate(fit["force_test"], axis=0).ravel()

    decoder = RidgeCV(alphas=ridge_alphas).fit(z_train, f_train)
    pred = decoder.predict(z_test)
    return {
        "force_decoder": decoder,
        "force_r2": float(r2_score(f_test, pred)),
        "force_alpha": float(decoder.alpha_),
    }


def summarize_fit(
    fit,
    *,
    evaluate_spike_reconstruction_fn,
    evaluate_force_decoding_fn,
    marginal_log_prob_per_bin_fn,
    final_log_likelihood_key,
    final_log_likelihood_name,
):
    """Summarize fit quality for one condition.

    Args:
        fit: Dict with condition, train/test indices, y_train, y_test, and training trace.
        evaluate_spike_reconstruction_fn: Callable taking fit and returning spike metrics.
        evaluate_force_decoding_fn: Callable taking fit and returning force metrics.
        marginal_log_prob_per_bin_fn: Callable taking (fit, trials) and returning a float.
        final_log_likelihood_key: Key in fit containing the training log-likelihood trace.
        final_log_likelihood_name: Output key name for the final trace value.

    Returns:
        Dict of counts, likelihoods, spike metrics, and force metrics.
    """
    spike_eval = evaluate_spike_reconstruction_fn(fit)
    force_eval = evaluate_force_decoding_fn(fit)
    log_likelihoods = np.asarray(fit[final_log_likelihood_key], dtype=float)
    return {
        "condition": fit["condition"],
        "n_train": len(fit["train_idx"]),
        "n_test": len(fit["test_idx"]),
        "n_bins": fit["y_train"][0].shape[0],
        final_log_likelihood_name: float(log_likelihoods[-1]),
        "train_ll_per_bin": marginal_log_prob_per_bin_fn(fit, fit["y_train"]),
        "test_ll_per_bin": marginal_log_prob_per_bin_fn(fit, fit["y_test"]),
        **{k: v for k, v in spike_eval.items() if not k.endswith("recon")},
        **{k: v for k, v in force_eval.items() if k != "force_decoder"},
    }
