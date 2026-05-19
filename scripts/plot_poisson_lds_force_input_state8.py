"""Generate saved plots for state-8 force-input Poisson LDS condition fits."""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib_cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.poisson_lds import (  # noqa: E402
    PoissonLDSInputParams,
    infer_latents_with_inputs,
    make_poisson_lds_input_params,
    poisson_rate,
    positive_diag,
)


DEFAULT_MODEL_DIR = SRC_DIR / "models" / "models"
DEFAULT_OUTPUT_DIR = SRC_DIR / "models" / "plots" / "poisson_lds_force_input_state8"
MODEL_GLOB = "condition_input_lds_poisson_cond*_bin50_state8_steps100_seed7.pkl"
BIN_MS = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save per-condition diagnostic plots for cached state-8 force-input Poisson LDS fits."
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--conditions", type=int, nargs="*", default=None)
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def restore_fit(payload: dict) -> dict:
    fit = dict(payload["fit"])
    if isinstance(fit["trainable_params"], dict):
        fit["trainable_params"] = PoissonLDSInputParams(
            **{name: np.asarray(fit["trainable_params"][name]) for name in PoissonLDSInputParams._fields}
        )
    fit["params"] = make_poisson_lds_input_params(fit["trainable_params"])
    return fit


def load_condition_fits(model_dir: Path, conditions: list[int] | None) -> dict[int, dict]:
    paths = sorted(model_dir.glob(MODEL_GLOB))
    if not paths:
        raise FileNotFoundError(f"no cached state-8 condition-input fits found in {model_dir}")

    fits = {}
    for path in paths:
        with path.open("rb") as f:
            payload = pickle.load(f)
        condition = int(payload["metadata"]["condition"])
        if conditions is not None and condition not in conditions:
            continue
        fits[condition] = restore_fit(payload)

    if not fits:
        raise ValueError(f"no cached fits matched requested conditions: {conditions}")
    return dict(sorted(fits.items()))


def as_2d_force(force: np.ndarray, n_bins: int) -> np.ndarray:
    arr = np.asarray(force, dtype=float).reshape(n_bins, -1)
    if arr.shape[1] != 1:
        raise ValueError(f"expected one force channel, got shape {arr.shape}")
    return arr[:, 0]


def rates_from_latents(trainable_params: PoissonLDSInputParams, latents: np.ndarray) -> np.ndarray:
    rates = [np.asarray(poisson_rate(trainable_params, z), dtype=float) for z in np.asarray(latents)]
    rates = np.stack(rates, axis=0)
    if not np.all(np.isfinite(rates)) or np.any(rates < 0):
        raise FloatingPointError("generated invalid Poisson rates")
    return rates


def simulate_latents_and_rates(
    fit: dict,
    u_trial: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    params = fit["trainable_params"]
    u = np.asarray(u_trial, dtype=float).reshape(u_trial.shape[0], -1)
    n_bins = u.shape[0]
    state_dim = params.A.shape[0]
    latents = np.zeros((n_bins, state_dim), dtype=float)

    s0_diag = np.asarray(positive_diag(params.log_s0), dtype=float)
    q_diag = np.asarray(positive_diag(params.log_q), dtype=float)
    latents[0] = np.asarray(params.m0, dtype=float) + rng.normal(scale=np.sqrt(s0_diag), size=state_dim)
    for t in range(1, n_bins):
        mean = np.asarray(params.A @ latents[t - 1] + params.B @ u[t - 1], dtype=float)
        latents[t] = mean + rng.normal(scale=np.sqrt(q_diag), size=state_dim)

    return latents, rates_from_latents(params, latents)


def save_heatmap_with_force(
    generated_rates: np.ndarray,
    ground_truth_counts: np.ndarray,
    force: np.ndarray,
    title: str,
    path: Path,
    dpi: int,
) -> None:
    n_bins = ground_truth_counts.shape[0]
    time_ms = np.arange(n_bins) * BIN_MS
    time_extent = [0, (n_bins - 1) * BIN_MS, 0.5, generated_rates.shape[1] + 0.5]
    vmax = np.nanpercentile(np.concatenate([generated_rates.ravel(), ground_truth_counts.ravel()]), 99)
    vmax = max(float(vmax), 1e-6)

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(13, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 0.8, 4, 0.8], "hspace": 0.12},
    )
    panels = [
        (axes[0], generated_rates.T, "Generated expected firing rate"),
        (axes[2], ground_truth_counts.T, "Ground-truth binned spike count"),
    ]
    for ax, data, label in panels:
        im = ax.imshow(
            data,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=time_extent,
            vmin=0,
            vmax=vmax,
        )
        ax.set_ylabel("neuron")
        ax.set_title(label)
        fig.colorbar(im, ax=ax, fraction=0.018, pad=0.01)

    for ax in [axes[1], axes[3]]:
        ax.plot(time_ms, force, color="black", lw=1.2)
        ax.set_ylabel("force")
        ax.grid(alpha=0.2)
    axes[3].set_xlabel("time (ms)")
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_transition_eigenvalues(fit: dict, path: Path, dpi: int) -> None:
    eigvals = np.linalg.eigvals(np.asarray(fit["trainable_params"].A, dtype=float))
    theta = np.linspace(0, 2 * np.pi, 361)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(np.cos(theta), np.sin(theta), color="0.7", lw=1, label="unit circle")
    ax.scatter(eigvals.real, eigvals.imag, color="tab:blue", s=45, label="eig(A)")
    ax.axhline(0, color="0.85", lw=0.8)
    ax.axvline(0, color="0.85", lw=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("real")
    ax.set_ylabel("imaginary")
    ax.set_title(f"Condition {fit['condition']} transition eigenvalues")
    ax.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_pca_projection(
    generated_rates: list[np.ndarray],
    ground_truth: list[np.ndarray],
    condition: int,
    path: Path,
    dpi: int,
) -> None:
    all_generated = np.concatenate(generated_rates, axis=0)
    all_ground_truth = np.concatenate(ground_truth, axis=0)
    pca = PCA(n_components=2).fit(np.concatenate([all_generated, all_ground_truth], axis=0))

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, (generated, truth) in enumerate(zip(generated_rates, ground_truth)):
        g_proj = pca.transform(generated)
        t_proj = pca.transform(truth)
        ax.scatter(g_proj[:, 0], g_proj[:, 1], color="tab:blue", alpha=0.35, s=10, label="generated" if i == 0 else None)
        ax.scatter(t_proj[:, 0], t_proj[:, 1], color="tab:orange", alpha=0.35, s=10, label="ground truth" if i == 0 else None)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"Condition {condition} PCA projection")
    ax.legend()
    ax.grid(alpha=0.2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def zero_like_trials(trials: list[np.ndarray]) -> list[np.ndarray]:
    return [np.zeros_like(np.asarray(trial, dtype=float)) for trial in trials]


def input_ablation_fraction(z_true: np.ndarray, z_zero: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    z_true = np.asarray(z_true, dtype=float)
    z_zero = np.asarray(z_zero, dtype=float)
    if z_true.shape != z_zero.shape:
        raise ValueError(f"latent shapes differ: true-u {z_true.shape}, zero-u {z_zero.shape}")
    return np.linalg.norm(z_true - z_zero, axis=1) / (np.linalg.norm(z_true, axis=1) + eps)


def save_latent_heatmap(
    latents: np.ndarray,
    ablation_fraction: np.ndarray,
    force: np.ndarray,
    condition: int,
    trial_index: int,
    path: Path,
    dpi: int,
) -> None:
    n_bins = latents.shape[0]
    extent = [0, (n_bins - 1) * BIN_MS, 0.5, latents.shape[1] + 0.5]
    time_ms = np.arange(n_bins) * BIN_MS
    ablation_fraction = np.asarray(ablation_fraction, dtype=float)
    if ablation_fraction.shape[0] != n_bins:
        raise ValueError(f"ablation fraction length {ablation_fraction.shape[0]} != latent length {n_bins}")

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1, 1], "hspace": 0.12},
    )
    ax = axes[0]
    im = ax.imshow(latents.T, aspect="auto", origin="lower", interpolation="nearest", extent=extent, cmap="coolwarm")
    ax.set_ylabel("latent dimension")
    ax.set_title(f"Condition {condition} test trial {trial_index}: inferred latent heatmap")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label="latent value")

    axes[1].plot(time_ms, ablation_fraction, color="tab:red", lw=1.2)
    axes[1].set_ylabel("input\nablation")
    axes[1].set_ylim(bottom=0)
    axes[1].grid(alpha=0.2)

    axes[2].plot(time_ms, force, color="black", lw=1.2)
    axes[2].set_ylabel("force")
    axes[2].set_xlabel("time (ms)")
    axes[2].grid(alpha=0.2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_latent_mean_sd(latents: list[np.ndarray], condition: int, path: Path, dpi: int) -> None:
    lengths = {z.shape[0] for z in latents}
    if len(lengths) != 1:
        min_len = min(lengths)
        latents = [z[:min_len] for z in latents]

    z_stack = np.stack(latents, axis=0)
    mean = z_stack.mean(axis=0)
    sd = z_stack.std(axis=0)
    time_ms = np.arange(mean.shape[0]) * BIN_MS
    state_dim = mean.shape[1]
    n_cols = 2
    n_rows = int(np.ceil(state_dim / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.2 * n_rows), sharex=True)
    axes = np.asarray(axes).ravel()
    for dim in range(state_dim):
        ax = axes[dim]
        ax.plot(time_ms, mean[:, dim], color="tab:blue", lw=1.4)
        ax.fill_between(time_ms, mean[:, dim] - sd[:, dim], mean[:, dim] + sd[:, dim], color="tab:blue", alpha=0.22)
        ax.axhline(0, color="0.85", lw=0.8)
        ax.set_title(f"z{dim}")
        ax.grid(alpha=0.18)
    for ax in axes[state_dim:]:
        ax.axis("off")
    for ax in axes[-n_cols:]:
        ax.set_xlabel("time (ms)")
    fig.suptitle(f"Condition {condition} inferred latent mean +/- SD across selected test trials")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_trial_averaged_latent_heatmap(
    latents: list[np.ndarray],
    force_trials: list[np.ndarray],
    condition: int,
    path: Path,
    dpi: int,
) -> None:
    if len(latents) != len(force_trials):
        raise ValueError("latents and force_trials must have the same length")
    if not latents:
        raise ValueError("at least one latent trial is required")

    min_len = min(np.asarray(z).shape[0] for z in latents)
    latent_stack = np.stack([np.asarray(z, dtype=float)[:min_len] for z in latents], axis=0)
    force_stack = np.stack([as_2d_force(np.asarray(force, dtype=float), np.asarray(force).shape[0])[:min_len] for force in force_trials], axis=0)
    mean_latent = latent_stack.mean(axis=0)
    mean_force = force_stack.mean(axis=0)

    n_bins = mean_latent.shape[0]
    time_ms = np.arange(n_bins) * BIN_MS
    extent = [0, (n_bins - 1) * BIN_MS, 0.5, mean_latent.shape[1] + 0.5]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 5),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.12},
    )
    im = axes[0].imshow(
        mean_latent.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=extent,
        cmap="coolwarm",
    )
    axes[0].set_ylabel("latent dimension")
    axes[0].set_title(f"Condition {condition}: all-test-trial averaged inferred latent heatmap")
    fig.colorbar(im, ax=axes[0], fraction=0.025, pad=0.01, label="mean latent value")

    axes[1].plot(time_ms, mean_force, color="black", lw=1.2)
    axes[1].set_ylabel("mean force")
    axes[1].set_xlabel("time (ms)")
    axes[1].grid(alpha=0.2)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def process_condition(
    condition: int,
    fit: dict,
    output_dir: Path,
    n_trials: int,
    rng: np.random.Generator,
    dpi: int,
) -> list[dict[str, str | int]]:
    condition_dir = output_dir / f"condition_{condition}"
    condition_dir.mkdir(parents=True, exist_ok=True)
    for old_plot in condition_dir.glob("*.png"):
        old_plot.unlink()

    n_available = len(fit["y_test"])
    n_selected = min(n_trials, n_available)
    selected_indices = np.sort(rng.choice(n_available, size=n_selected, replace=False))
    y_trials = [fit["y_test"][i] for i in selected_indices]
    u_trials = [fit["u_test"][i] for i in selected_indices]
    force_trials = [fit["force_test"][i] for i in selected_indices]
    inferred_latents = infer_latents_with_inputs(fit["params"], y_trials, u_trials)
    zero_input_latents = infer_latents_with_inputs(fit["params"], y_trials, zero_like_trials(u_trials))
    all_inferred_latents = infer_latents_with_inputs(fit["params"], fit["y_test"], fit["u_test"])

    generated_rates = []
    ground_truth_counts = []
    rows: list[dict[str, str | int]] = []

    for trial_index, y, u, force, z_inf, z_zero in zip(
        selected_indices,
        y_trials,
        u_trials,
        force_trials,
        inferred_latents,
        zero_input_latents,
    ):
        trial_index = int(trial_index)
        y = np.asarray(y, dtype=float)
        force_line = as_2d_force(force, y.shape[0])
        _, rates = simulate_latents_and_rates(fit, np.asarray(u, dtype=float), rng)
        generated_rates.append(rates)
        ground_truth_counts.append(y)

        heatmap_path = condition_dir / f"spike_heatmap_trial_{trial_index}.png"
        save_heatmap_with_force(
            rates,
            y,
            force_line,
            f"Condition {condition} test trial {trial_index}: generated vs ground truth",
            heatmap_path,
            dpi,
        )
        rows.append({"condition": condition, "plot_type": "spike_heatmap", "trial_index": trial_index, "path": str(heatmap_path)})

        latent_path = condition_dir / f"latent_heatmap_trial_{trial_index}.png"
        ablation_fraction = input_ablation_fraction(np.asarray(z_inf, dtype=float), np.asarray(z_zero, dtype=float))
        save_latent_heatmap(np.asarray(z_inf, dtype=float), ablation_fraction, force_line, condition, trial_index, latent_path, dpi)
        rows.append({"condition": condition, "plot_type": "latent_heatmap", "trial_index": trial_index, "path": str(latent_path)})

    eig_path = condition_dir / "transition_eigenvalues.png"
    save_transition_eigenvalues(fit, eig_path, dpi)
    rows.append({"condition": condition, "plot_type": "transition_eigenvalues", "trial_index": "", "path": str(eig_path)})

    pca_path = condition_dir / "pca_generated_vs_groundtruth.png"
    save_pca_projection(generated_rates, ground_truth_counts, condition, pca_path, dpi)
    rows.append({"condition": condition, "plot_type": "pca_generated_vs_groundtruth", "trial_index": "", "path": str(pca_path)})

    latent_summary_path = condition_dir / "latent_mean_sd.png"
    save_latent_mean_sd([np.asarray(z, dtype=float) for z in inferred_latents], condition, latent_summary_path, dpi)
    rows.append({"condition": condition, "plot_type": "latent_mean_sd", "trial_index": "", "path": str(latent_summary_path)})

    averaged_latent_path = condition_dir / "latent_trial_averaged_heatmap.png"
    save_trial_averaged_latent_heatmap(
        [np.asarray(z, dtype=float) for z in all_inferred_latents],
        fit["force_test"],
        condition,
        averaged_latent_path,
        dpi,
    )
    rows.append(
        {
            "condition": condition,
            "plot_type": "latent_trial_averaged_heatmap",
            "trial_index": "",
            "path": str(averaged_latent_path),
        }
    )
    return rows


def write_manifest(rows: list[dict[str, str | int]], output_dir: Path) -> Path:
    path = output_dir / "manifest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "plot_type", "trial_index", "path"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> None:
    args = parse_args()
    if args.n_trials <= 0:
        raise ValueError("--n-trials must be positive")

    fits = load_condition_fits(args.model_dir, args.conditions)
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, str | int]] = []
    for condition, fit in fits.items():
        print(f"processing condition {condition}", flush=True)
        rows.extend(process_condition(condition, fit, args.output_dir, args.n_trials, rng, args.dpi))

    manifest_path = write_manifest(rows, args.output_dir)
    print(f"saved {len(rows)} plot records to {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
