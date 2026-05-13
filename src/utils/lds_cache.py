from pathlib import Path
import pickle

import numpy as np


def _metadata_value(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def lds_metadata(cache_version, kind, emission_distribution, data_path, condition=None, **fields):
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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata, "fit": payload_fn(fit)}
    with path.open("wb") as f:
        pickle.dump(payload, f)
    print(f"saved model parameters {path}", flush=True)


def load_lds_fit(path, expected_metadata, restore_fn):
    path = Path(path)
    with path.open("rb") as f:
        payload = pickle.load(f)
    if payload.get("metadata") != expected_metadata:
        print(f"model metadata mismatch for {path.name}; refitting", flush=True)
        return None
    print(f"loading model parameters {path}", flush=True)
    return restore_fn(payload["fit"])


def load_or_fit_lds(path, metadata, fit_fn, payload_fn, restore_fn):
    path = Path(path)
    if path.exists():
        fit = load_lds_fit(path, metadata, restore_fn)
        if fit is not None:
            return fit
    fit = fit_fn()
    save_lds_fit(path, metadata, fit, payload_fn)
    return fit


def gaussian_fit_payload_for_pickle(fit):
    payload = dict(fit)
    payload.pop("model", None)
    return payload


def restore_gaussian_fit_from_pickle(payload):
    from dynamax.linear_gaussian_ssm import LinearGaussianConjugateSSM

    fit = dict(payload)
    params = fit["params"]
    state_dim = int(params.dynamics.weights.shape[0])
    emission_dim = int(params.emissions.weights.shape[0])
    fit["model"] = LinearGaussianConjugateSSM(state_dim=state_dim, emission_dim=emission_dim)
    return fit


def poisson_trainable_params_to_arrays(trainable_params):
    return {name: np.asarray(getattr(trainable_params, name)) for name in trainable_params._fields}


def poisson_fit_payload_for_pickle(fit):
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
