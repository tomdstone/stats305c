from pathlib import Path
import pickle

import numpy as np


def _metadata_value(value):
    """Normalize one metadata value for stable pickle comparison.

    Args:
        value: NumPy scalar, Path, or already pickle-stable Python value.

    Returns:
        int, float, str, or the original value.
    """
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def lds_metadata(cache_version, kind, emission_distribution, data_path, condition=None, **fields):
    """Build metadata that fingerprints an LDS fit and its source data.

    Args:
        cache_version: Version string/number used to invalidate old caches.
        kind: Model family or fitting mode, e.g. "pooled" or "condition".
        emission_distribution: Emission name; stored lowercase.
        data_path: Source data path whose mtime and size enter the fingerprint.
        condition: Optional condition label for condition-specific fits.
        **fields: Extra scalar settings such as bin_ms, state_dim, seed, or steps.

    Returns:
        Dict suitable for exact equality checks before cache reuse.
    """
    data_path = Path(data_path)
    metadata = {
        "cache_version": str(cache_version),
        "kind": str(kind),
        "emission_distribution": str(emission_distribution).lower(),
    }
    for key, value in fields.items():
        metadata[key] = _metadata_value(value)
    metadata.update(
        {
            "data_path": str(data_path),
            "data_mtime_ns": data_path.stat().st_mtime_ns if data_path.exists() else None,
            "data_size": data_path.stat().st_size if data_path.exists() else None,
        }
    )
    if condition is not None:
        metadata["condition"] = int(condition)
    return metadata


def lds_path(model_dir, kind, metadata):
    """Create a deterministic cache path for an LDS fit.

    Args:
        model_dir: Directory where model cache files are stored.
        kind: Model family or fitting mode used in the filename.
        metadata: Dict from lds_metadata containing emission_distribution, bin_ms,
            state_dim, seed, and optionally condition/learning_steps/em_iters.

    Returns:
        Path ending in a descriptive .pkl filename.
    """
    fields = [kind, metadata["emission_distribution"]]
    if "condition" in metadata:
        fields.append(f"cond{metadata['condition']}")
    fields += [
        f"bin{metadata['bin_ms']}",
        f"state{metadata['state_dim']}",
    ]
    if "learning_steps" in metadata:
        fields.append(f"steps{metadata['learning_steps']}")
    elif "em_iters" in metadata:
        fields.append(f"iters{metadata['em_iters']}")
    fields.append(f"seed{metadata['seed']}")
    return Path(model_dir) / ("_".join(fields) + ".pkl")


def save_lds_fit(path, metadata, fit, payload_fn):
    """Serialize a fitted LDS and metadata.

    Args:
        path: Destination pickle path; parent directories are created.
        metadata: Dict that will be checked on load.
        fit: Model-specific fit object or dict.
        payload_fn: Callable converting fit into a pickle-safe payload.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata, "fit": payload_fn(fit)}
    with path.open("wb") as f:
        pickle.dump(payload, f)
    print(f"saved model parameters {path}", flush=True)


def load_lds_fit(path, expected_metadata, restore_fn):
    """Load a cached LDS fit if its metadata exactly matches.

    Args:
        path: Pickle path written by save_lds_fit.
        expected_metadata: Metadata dict required for cache reuse.
        restore_fn: Callable converting the stored payload back to a fit object.

    Returns:
        Restored fit, or None when metadata does not match.
    """
    path = Path(path)
    with path.open("rb") as f:
        payload = pickle.load(f)
    if payload.get("metadata") != expected_metadata:
        print(f"model metadata mismatch for {path.name}; refitting", flush=True)
        return None
    print(f"loading model parameters {path}", flush=True)
    return restore_fn(payload["fit"])


def load_or_fit_lds(path, metadata, fit_fn, payload_fn, restore_fn):
    """Load a valid cache or fit and save a new one.

    Args:
        path: Cache pickle path.
        metadata: Expected metadata dict for this run.
        fit_fn: Zero-argument callable that fits the model on cache miss.
        payload_fn: Callable converting a fit into a pickle-safe payload.
        restore_fn: Callable converting a stored payload back to a fit.

    Returns:
        Cached or newly fitted model object.
    """
    path = Path(path)
    if path.exists():
        fit = load_lds_fit(path, metadata, restore_fn)
        if fit is not None:
            return fit
    fit = fit_fn()
    save_lds_fit(path, metadata, fit, payload_fn)
    return fit


def gaussian_fit_payload_for_pickle(fit):
    """Convert a Gaussian LDS fit dict into a pickle payload.

    Args:
        fit: Dict containing Dynamax params and optionally a non-pickled model object.

    Returns:
        Shallow copy of fit without the model key.
    """
    payload = dict(fit)
    payload.pop("model", None)
    return payload


def restore_gaussian_fit_from_pickle(payload):
    """Restore a Gaussian LDS fit dict from a pickle payload.

    Args:
        payload: Dict produced by gaussian_fit_payload_for_pickle, including params.

    Returns:
        Fit dict with a recreated LinearGaussianConjugateSSM model.
    """
    from dynamax.linear_gaussian_ssm import LinearGaussianConjugateSSM

    fit = dict(payload)
    params = fit["params"]
    state_dim = int(params.dynamics.weights.shape[0])
    emission_dim = int(params.emissions.weights.shape[0])
    fit["model"] = LinearGaussianConjugateSSM(state_dim=state_dim, emission_dim=emission_dim)
    return fit


def poisson_trainable_params_to_arrays(trainable_params):
    """Convert Poisson trainable parameters to NumPy arrays.

    Args:
        trainable_params: PoissonLDSParams with JAX/NumPy array fields.

    Returns:
        Dict mapping each parameter name to a NumPy array.
    """
    return {name: np.asarray(getattr(trainable_params, name)) for name in trainable_params._fields}


def poisson_fit_payload_for_pickle(fit):
    """Convert a Poisson LDS fit dict into a pickle payload.

    Args:
        fit: Dict containing trainable_params plus derived params/model/latents/predictions.

    Returns:
        Shallow copy with derived values removed and trainable_params stored as arrays.
    """
    payload = dict(fit)
    payload.pop("params", None)
    payload.pop("model", None)
    payload.pop("latent_train", None)
    payload.pop("latent_test", None)
    payload.pop("expected_train", None)
    payload.pop("expected_test", None)
    payload["trainable_params"] = poisson_trainable_params_to_arrays(payload["trainable_params"])
    return payload


def restore_poisson_fit_from_pickle(payload):
    """Restore a Poisson LDS fit dict from a pickle payload.

    Args:
        payload: Dict produced by poisson_fit_payload_for_pickle.

    Returns:
        Fit dict with PoissonLDSParams, ParamsGGSSM, and GeneralizedGaussianSSM model.
    """
    import jax.numpy as jnp
    from dynamax.generalized_gaussian_ssm import GeneralizedGaussianSSM
    from utils.poisson_lds import PoissonLDSParams, make_poisson_lds_params

    fit = dict(payload)
    if isinstance(fit["trainable_params"], dict):
        fit["trainable_params"] = PoissonLDSParams(
            **{name: jnp.asarray(fit["trainable_params"][name]) for name in PoissonLDSParams._fields}
        )
    trainable_params = fit["trainable_params"]
    state_dim = int(trainable_params.A.shape[0])
    emission_dim = int(trainable_params.C.shape[0])
    fit["params"] = make_poisson_lds_params(trainable_params)
    fit["model"] = GeneralizedGaussianSSM(state_dim=state_dim, emission_dim=emission_dim)
    return fit
