"""Compare held-out NLL for state-8 Poisson LDS models with and without input."""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

os.environ.setdefault("JAX_PLATFORMS", "cpu")

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dynamax.generalized_gaussian_ssm import EKFIntegrals  # noqa: E402
from dynamax.generalized_gaussian_ssm import conditional_moments_gaussian_smoother as cmgs  # noqa: E402
from utils.poisson_lds import (  # noqa: E402
    PoissonLDSInputParams,
    PoissonLDSParams,
    make_poisson_lds_input_params,
    make_poisson_lds_params,
)


DEFAULT_INPUT_MODEL_DIR = SRC_DIR / "models" / "models"
DEFAULT_NO_INPUT_MODEL_DIR = SRC_DIR / "eda" / "saved_models"
DEFAULT_OUTPUT_CSV = (
    SRC_DIR / "models" / "plots" / "poisson_lds_force_input_state8" / "state8_poisson_nll_comparison.csv"
)
INPUT_MODEL_GLOB = "condition_input_lds_poisson_cond*_bin50_state8_steps100_seed7.pkl"
NO_INPUT_MODEL_GLOB = "condition_lds_poisson_cond*_bin50_state8_steps100_seed7.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a CSV comparing held-out NLL for state-8 Poisson LDS condition models."
    )
    parser.add_argument("--input-model-dir", type=Path, default=DEFAULT_INPUT_MODEL_DIR)
    parser.add_argument("--no-input-model-dir", type=Path, default=DEFAULT_NO_INPUT_MODEL_DIR)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--conditions", type=int, nargs="*", default=None)
    return parser.parse_args()


def condition_from_payload(payload: dict) -> int:
    metadata = payload.get("metadata", {})
    if "condition" in metadata:
        return int(metadata["condition"])
    fit = payload.get("fit", {})
    if "condition" in fit:
        return int(fit["condition"])
    raise KeyError("could not determine condition from pickle payload")


def load_payloads(model_dir: Path, pattern: str, conditions: list[int] | None) -> dict[int, dict]:
    paths = sorted(model_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no model files found in {model_dir} matching {pattern}")

    payloads = {}
    for path in paths:
        with path.open("rb") as f:
            payload = pickle.load(f)
        condition = condition_from_payload(payload)
        if conditions is not None and condition not in conditions:
            continue
        payloads[condition] = payload

    if not payloads:
        raise ValueError(f"no models matched requested conditions: {conditions}")
    return dict(sorted(payloads.items()))


def restore_input_fit(payload: dict) -> dict:
    fit = dict(payload["fit"])
    params = fit["trainable_params"]
    if isinstance(params, dict):
        params = PoissonLDSInputParams(**{name: jnp.asarray(params[name]) for name in PoissonLDSInputParams._fields})
    fit["trainable_params"] = params
    fit["params"] = make_poisson_lds_input_params(params)
    return fit


def restore_no_input_fit(payload: dict) -> dict:
    fit = dict(payload["fit"])
    params = fit["trainable_params"]
    if isinstance(params, dict):
        params = PoissonLDSParams(**{name: jnp.asarray(params[name]) for name in PoissonLDSParams._fields})
    fit["trainable_params"] = params
    fit["params"] = make_poisson_lds_params(params)
    return fit


def input_trial_nlls(fit: dict) -> tuple[np.ndarray, np.ndarray]:
    y_trials = fit["y_test"]
    u_trials = fit["u_test"]
    if len(y_trials) != len(u_trials):
        raise ValueError(f"condition {fit['condition']}: y_test and u_test lengths differ")

    nlls = []
    lengths = []
    for y, u in zip(y_trials, u_trials):
        y_arr = jnp.asarray(y)
        u_arr = jnp.asarray(u)
        if y_arr.shape[0] != u_arr.shape[0]:
            raise ValueError(f"condition {fit['condition']}: y/u trial lengths differ")
        marginal_loglik = cmgs(fit["params"], EKFIntegrals(), y_arr, inputs=u_arr).marginal_loglik
        nlls.append(-float(marginal_loglik))
        lengths.append(int(y_arr.shape[0]))
    return np.asarray(nlls, dtype=float), np.asarray(lengths, dtype=int)


def no_input_trial_nlls(fit: dict) -> tuple[np.ndarray, np.ndarray]:
    nlls = []
    lengths = []
    for y in fit["y_test"]:
        y_arr = jnp.asarray(y)
        marginal_loglik = cmgs(fit["params"], EKFIntegrals(), y_arr).marginal_loglik
        nlls.append(-float(marginal_loglik))
        lengths.append(int(y_arr.shape[0]))
    return np.asarray(nlls, dtype=float), np.asarray(lengths, dtype=int)


def summarize_nll(nlls: np.ndarray, lengths: np.ndarray, prefix: str) -> dict[str, float | int]:
    if nlls.size == 0:
        raise ValueError(f"{prefix}: no test trials")
    if not np.all(np.isfinite(nlls)):
        raise FloatingPointError(f"{prefix}: non-finite NLL values")
    if np.any(lengths <= 0):
        raise ValueError(f"{prefix}: non-positive trial lengths")

    total_bins = int(lengths.sum())
    total_nll = float(nlls.sum())
    return {
        f"{prefix}_total_nll": total_nll,
        f"{prefix}_nll_per_bin": total_nll / total_bins,
        f"{prefix}_mean_trial_nll_per_bin": float(np.mean(nlls / lengths)),
    }


def row_for_condition(condition: int, input_fit: dict, no_input_fit: dict) -> dict[str, float | int]:
    input_nlls, input_lengths = input_trial_nlls(input_fit)
    no_input_nlls, no_input_lengths = no_input_trial_nlls(no_input_fit)

    if input_nlls.size != no_input_nlls.size:
        raise ValueError(
            f"condition {condition}: input model has {input_nlls.size} test trials, "
            f"no-input model has {no_input_nlls.size}"
        )
    if int(input_lengths.sum()) != int(no_input_lengths.sum()):
        raise ValueError(
            f"condition {condition}: input model has {input_lengths.sum()} test bins, "
            f"no-input model has {no_input_lengths.sum()}"
        )

    row: dict[str, float | int] = {
        "condition": int(condition),
        "n_test_trials": int(input_nlls.size),
        "n_test_bins_total": int(input_lengths.sum()),
    }
    row.update(summarize_nll(input_nlls, input_lengths, "input"))
    row.update(summarize_nll(no_input_nlls, no_input_lengths, "no_input"))
    row["delta_nll_per_bin_no_input_minus_input"] = row["no_input_nll_per_bin"] - row["input_nll_per_bin"]
    return row


def main() -> None:
    args = parse_args()
    input_payloads = load_payloads(args.input_model_dir, INPUT_MODEL_GLOB, args.conditions)
    no_input_payloads = load_payloads(args.no_input_model_dir, NO_INPUT_MODEL_GLOB, args.conditions)

    input_conditions = set(input_payloads)
    no_input_conditions = set(no_input_payloads)
    if input_conditions != no_input_conditions:
        raise ValueError(
            "condition mismatch between model sets: "
            f"input-only={sorted(input_conditions - no_input_conditions)}, "
            f"no-input-only={sorted(no_input_conditions - input_conditions)}"
        )

    rows = []
    for condition in sorted(input_conditions):
        print(f"scoring condition {condition}", flush=True)
        input_fit = restore_input_fit(input_payloads[condition])
        no_input_fit = restore_no_input_fit(no_input_payloads[condition])
        rows.append(row_for_condition(condition, input_fit, no_input_fit))

    df = pd.DataFrame(rows).sort_values("condition").reset_index(drop=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"wrote {len(df)} rows to {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
