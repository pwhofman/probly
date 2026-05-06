"""Credal-set and conformal-prediction analyses on first-order DCIC data.

Usage:
    1. Fill in WANDB_RUN_IDS below with your (method_name, dataset) -> run_id mapping.
    2. Run from project root: uv run python src/probly_benchmark/first_order_data_comparisons.py
    3. Results saved to OUTPUT_DIR/results.json  (per-instance arrays in results_arrays.npz)
    4. TV histograms saved as PDFs in OUTPUT_DIR/plots/
    5. LaTeX tables saved to OUTPUT_DIR/tables.tex (also printed to stdout)
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from scipy import stats
from scipy.optimize import linprog
import torch
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from collections.abc import Callable

sys.path.insert(0, str(Path(__file__).parent.parent))

from probly.conformal_scores.aps._common import compute_aps_score_numpy
from probly.conformal_scores.lac._common import compute_lac_score_numpy
from probly.conformal_scores.raps._common import compute_raps_score_numpy
from probly.conformal_scores.saps._common import compute_saps_score_func_numpy
from probly_benchmark.data import _DCIC_FOLDER_NAME, DCIC_IMAGE_PATH, TRANSFORMS_TEST, _DcicZeroOrderDataset
from probly_benchmark.utils import collect_outputs_targets_raw, get_device, load_model_for_evaluation

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION -- fill in your run IDs and model settings
# ─────────────────────────────────────────────────────────────────────────────

WANDB_ENTITY = "probly"
WANDB_PROJECT = "test"
BASE_MODEL = "resnet50"
MODEL_TYPE = "logit_classifier"
PRETRAINED = True
SEED = 1
VAL_SPLIT = 0.1
BATCH_SIZE = 32
NUM_WORKERS = 4

# Map (method_name, dataset) -> wandb_run_id
# Example:
#   ("credal_relative_likelihood", "qualitymri"): "abc123xy",
WANDB_RUN_IDS: dict[tuple[str, str], str] = {
    ("credal_relative_likelihood", "qualitymri"): "e196f1oh",
    ("credal_relative_likelihood", "micebone"):   "3l1xzrgs",
    ("credal_relative_likelihood", "synthetic"):  "kdi08bpk",
    ("credal_ensembling", "qualitymri"):           "xps1ojts",
    ("credal_ensembling", "micebone"):             "26t6udc5",
    ("credal_ensembling", "synthetic"):            "05c9j9gg",
    ("credal_wrapper", "qualitymri"):           "7icmxqxs",
    ("credal_wrapper", "micebone"):             "g8snvupb",
    ("credal_wrapper", "synthetic"):            "abt7cq1t",
    # ("credal_bnn", "qualitymri"):           "8jycofwr",
    # ("credal_bnn", "micebone"):             "qhnux861",
    # ("credal_bnn", "synthetic"):            "lecoq3go",
    # ("credal_net", "qualitymri"):           "",
    # ("credal_net", "micebone"):             "wcvrdedo",
    # ("credal_net", "synthetic"):            "mrjzyn90",
}

ALPHA = 0.1
CP_N_SEEDS = 10
CP_CAL_FRAC = 0.5
RELAXED_EPSILON = 0.01

OUTPUT_DIR = Path(__file__).parent / "first_order_results"

# Display names for credal methods in tables
METHOD_DISPLAY: dict[str, str] = {
    "credal_relative_likelihood": "CreRL",
    "credal_ensembling": "CreEns",
    "credal_net": "CreNet",
    "credal_bnn": "CreBNN",
    "credal_wrapper": "CreWra",
}

CP_METHODS: dict[str, Any] = {
    "LAC": lambda p, y=None: compute_lac_score_numpy(p, y),
    "APS": lambda p, y=None: compute_aps_score_numpy(p, y, randomized=False),
    "RAPS": lambda p, y=None: compute_raps_score_numpy(p, y, randomized=False),
    "SAPS": lambda p, y=None: compute_saps_score_func_numpy(p, y, randomized=False),
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA UTILITIES
# ─────────────────────────────────────────────────────────────────────────────


def build_test_dataset(dataset: str) -> _DcicZeroOrderDataset:
    """Build a zero-order test dataset for fold 1 of a DCIC dataset."""
    folder = _DCIC_FOLDER_NAME[dataset]
    root = DCIC_IMAGE_PATH / folder
    return _DcicZeroOrderDataset(root, folds=[1], transform=TRANSFORMS_TEST["imagenet"], seed=SEED)


def get_true_probs(test_ds: _DcicZeroOrderDataset) -> np.ndarray:
    """Compute empirical p(y|x) from annotator counts for each test image."""
    num_classes = test_ds.num_classes
    true_probs = np.zeros((len(test_ds), num_classes), dtype=np.float64)
    for i, ann in enumerate(test_ds._annotation_lists):  # noqa: SLF001
        counts = np.bincount(ann, minlength=num_classes).astype(float)
        true_probs[i] = counts / counts.sum()
    return true_probs


def build_cfg(method_name: str, dataset: str, run_id: str) -> Any:  # noqa: ANN401
    """Build a minimal OmegaConf config for loading a model from W&B."""
    return OmegaConf.create(
        {
            "dataset": dataset,
            "base_model": BASE_MODEL,
            "model_type": MODEL_TYPE,
            "pretrained": PRETRAINED,
            "seed": SEED,
            "val_split": VAL_SPLIT,
            "cal_split": 0.0,
            "method": {"name": method_name},
            "wandb": {
                "run_id": run_id,
                "entity": WANDB_ENTITY,
                "project": WANDB_PROJECT,
            },
        }
    )


def load_ensemble_probs(
    method_name: str,
    dataset: str,
    run_id: str,
    test_loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Return ensemble softmax probs of shape (n_test, n_members, K)."""
    cfg = build_cfg(method_name, dataset, run_id)
    model, _, _ = load_model_for_evaluation(cfg, device)
    model.eval()

    members = list(model) if isinstance(model, torch.nn.ModuleList) else [model]
    all_logits: list[torch.Tensor] = []
    for member in members:
        member.eval()
        logits_m, _ = collect_outputs_targets_raw(member, test_loader, device)
        all_logits.append(logits_m)

    # (n_members, n_test, K) -> softmax -> (n_test, n_members, K)
    ensemble_probs = torch.stack(all_logits).softmax(dim=-1).permute(1, 0, 2).numpy()
    return ensemble_probs


# ─────────────────────────────────────────────────────────────────────────────
# CREDAL COVERAGE & TV METRICS
# ─────────────────────────────────────────────────────────────────────────────


def compute_interval_coverage(ensemble_probs: np.ndarray, true_probs: np.ndarray) -> tuple[np.ndarray, float]:
    """Check if true_probs lies within [min, max] of ensemble members for each class.

    Args:
        ensemble_probs: (n_test, n_members, K)
        true_probs: (n_test, K)

    Returns:
        per_instance: (n_test,) bool array
        coverage: float
    """
    lower = ensemble_probs.min(axis=1)  # (n_test, K)
    upper = ensemble_probs.max(axis=1)  # (n_test, K)
    per_instance = np.all((true_probs >= lower) & (true_probs <= upper), axis=-1)
    return per_instance.astype(float), float(per_instance.mean())


def compute_credal_efficiency(ensemble_probs: np.ndarray) -> tuple[np.ndarray, float]:
    """Average total-variation width of the probability interval credal set.

    Lower = smaller (more efficient) credal set.

    Args:
        ensemble_probs: (n_test, n_members, K)

    Returns:
        per_instance: (n_test,) interval width per test point (sum over classes)
        mean_efficiency: float (average over test set)
    """
    lower = ensemble_probs.min(axis=1)
    upper = ensemble_probs.max(axis=1)
    per_instance = (upper - lower).sum(axis=-1)  # sum of widths
    return per_instance, float(per_instance.mean())


def compute_credal_coverage_metrics(
    ensemble_probs: np.ndarray,
    true_probs: np.ndarray,
) -> dict[str, Any]:
    """Compute all four credal coverage / efficiency metrics.

    Args:
        ensemble_probs: (n_test, n_members, K)
        true_probs: (n_test, K) empirical p(y|x)

    Returns:
        dict with scalar metrics and per-instance arrays for significance testing.
    """
    print("  Computing convex hull coverage ...")
    per_ch = np.zeros(len(true_probs))
    covered = 0
    n_extrema = ensemble_probs.shape[1]
    c = np.zeros(n_extrema)
    bounds_lp = [(0, 1)] * n_extrema
    for i in range(ensemble_probs.shape[0]):
        a_eq = np.vstack((ensemble_probs[i].T, np.ones(n_extrema)))
        b_eq = np.concatenate((true_probs[i], [1]))
        res = linprog(c=c, A_eq=a_eq, b_eq=b_eq, bounds=bounds_lp)
        per_ch[i] = float(res.success)
        covered += res.success
    ch_coverage = float(covered / ensemble_probs.shape[0])

    print(f"    CH coverage: {ch_coverage:.4f}")

    print(f"  Computing relaxed convex hull coverage (epsilon={RELAXED_EPSILON}) ...")
    per_ch_relaxed = np.zeros(len(true_probs))
    n_classes = ensemble_probs.shape[2]
    c_relaxed = np.concatenate([np.zeros(n_extrema), np.ones(2 * n_classes)])
    bounds_relaxed = [(0, 1)] * n_extrema + [(0, None)] * (2 * n_classes)
    covered_relaxed = 0
    for i in range(ensemble_probs.shape[0]):
        a_eq_top = np.hstack([ensemble_probs[i].T, np.eye(n_classes), -np.eye(n_classes)])
        a_eq_bot = np.concatenate([np.ones(n_extrema), np.zeros(2 * n_classes)])
        a_eq = np.vstack([a_eq_top, a_eq_bot])
        b_eq = np.concatenate([true_probs[i], [1]])
        res = linprog(c=c_relaxed, A_eq=a_eq, b_eq=b_eq, bounds=bounds_relaxed)
        ok = res.success and res.fun <= RELAXED_EPSILON
        per_ch_relaxed[i] = float(ok)
        covered_relaxed += ok
    ch_relaxed_coverage = float(covered_relaxed / ensemble_probs.shape[0])

    print(f"    Relaxed CH coverage: {ch_relaxed_coverage:.4f}")

    print("  Computing interval coverage ...")
    per_interval, interval_coverage = compute_interval_coverage(ensemble_probs, true_probs)

    print(f"    Interval coverage: {interval_coverage:.4f}")

    print("  Computing credal efficiency ...")
    per_efficiency, mean_efficiency = compute_credal_efficiency(ensemble_probs)

    print(f"    Credal efficiency (interval width): {mean_efficiency:.4f}")

    return {
        "ch_coverage": ch_coverage,
        "ch_relaxed_coverage": ch_relaxed_coverage,
        "interval_coverage": interval_coverage,
        "credal_efficiency": mean_efficiency,
        "per_ch": per_ch.tolist(),
        "per_ch_relaxed": per_ch_relaxed.tolist(),
        "per_interval": per_interval.tolist(),
        "per_efficiency": per_efficiency.tolist(),
    }


def compute_tv_metrics(ensemble_probs: np.ndarray, true_probs: np.ndarray) -> dict[str, Any]:
    """Compute TV distance between ensemble mean and true distribution.

    Args:
        ensemble_probs: (n_test, n_members, K)
        true_probs: (n_test, K)

    Returns:
        dict with mean, std, and per-instance TV distances.
    """
    ensemble_mean = ensemble_probs.mean(axis=1)  # (n_test, K)
    tv_per_instance = 0.5 * np.abs(ensemble_mean - true_probs).sum(axis=-1)  # (n_test,)
    return {
        "mean": float(tv_per_instance.mean()),
        "std": float(tv_per_instance.std()),
        "per_instance": tv_per_instance.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CONFORMAL PREDICTION (ZERO-ORDER)
# ─────────────────────────────────────────────────────────────────────────────


def _conformal_quantile(cal_scores: np.ndarray, alpha: float) -> float:
    n = len(cal_scores)
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(cal_scores, level))


def run_conformal_single(
    score_fn: Callable[..., np.ndarray],
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    test_true_probs: np.ndarray,
    alpha: float,
) -> dict[str, float]:
    """Run one split-conformal trial and return the three metrics."""
    cal_scores = score_fn(cal_probs, cal_labels)
    q = _conformal_quantile(cal_scores, alpha)

    all_test_scores = score_fn(test_probs)  # (n_test, K)
    pred_sets = all_test_scores <= q  # (n_test, K)

    n_test = len(test_labels)
    marginal_cov = float(pred_sets[np.arange(n_test), test_labels].mean())
    avg_set_size = float(pred_sets.sum(axis=-1).mean())
    prob_mass_in_set = (pred_sets * test_true_probs).sum(axis=-1)
    cond_sat = float((prob_mass_in_set >= 1 - alpha).mean())

    return {
        "marginal_coverage": marginal_cov,
        "avg_set_size": avg_set_size,
        "cond_satisfaction": cond_sat,
    }


def sample_zero_order(true_probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample one hard label per image from the empirical annotator distribution."""
    num_classes = true_probs.shape[1]
    return np.array([rng.choice(num_classes, p=true_probs[i]) for i in range(len(true_probs))])


def run_cp_all_seeds(
    ensemble_mean: np.ndarray,
    true_probs_test: np.ndarray,
    alpha: float = ALPHA,
    n_seeds: int = CP_N_SEEDS,
    cal_frac: float = CP_CAL_FRAC,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Run CP for all score functions and seeds.

    Returns:
        {cp_name: {"marginal_coverage": {"mean", "std"}, "avg_set_size": ..., "cond_satisfaction": ...}}
    """
    seed_results: dict[str, list[dict[str, float]]] = defaultdict(list)

    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        zero_order_labels = sample_zero_order(true_probs_test, rng)

        n = len(zero_order_labels)
        idx = rng.permutation(n)
        n_cal = int(n * cal_frac)
        cal_idx, test_idx = idx[:n_cal], idx[n_cal:]

        cal_probs = ensemble_mean[cal_idx]
        cal_labels = zero_order_labels[cal_idx]
        test_probs_split = ensemble_mean[test_idx]
        test_labels = zero_order_labels[test_idx]
        test_true_probs_split = true_probs_test[test_idx]

        for cp_name, score_fn in CP_METHODS.items():
            metrics = run_conformal_single(
                score_fn, cal_probs, cal_labels, test_probs_split, test_labels, test_true_probs_split, alpha
            )
            seed_results[cp_name].append(metrics)

    aggregated: dict[str, dict[str, dict[str, Any]]] = {}
    for cp_name, seed_list in seed_results.items():
        aggregated[cp_name] = {}
        for metric in ["marginal_coverage", "avg_set_size", "cond_satisfaction"]:
            vals = [s[metric] for s in seed_list]
            aggregated[cp_name][metric] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "per_seed": vals,
            }
    return aggregated


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────


def plot_tv_histograms(all_results: dict[tuple[str, str], dict[str, Any]], output_dir: Path) -> None:
    """Save per-dataset TV distance histograms as PDFs, one curve per credal method."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    datasets = sorted({ds for _, ds in all_results})
    methods = sorted({m for m, _ in all_results})

    for dataset in datasets:
        fig, ax = plt.subplots(figsize=(7, 4))
        for method in methods:
            key = (method, dataset)
            if key not in all_results:
                continue
            tv = np.array(all_results[key]["tv"]["per_instance"])
            label = METHOD_DISPLAY.get(method, method)
            ax.hist(tv, bins=30, alpha=0.6, label=label, density=True)

        ax.set_xlabel("TV distance (ensemble mean vs. true distribution)")
        ax.set_ylabel("Density")
        ax.set_title(f"{dataset} - TV distance distribution")
        ax.legend()
        plt.tight_layout()
        out_path = plots_dir / f"tv_hist_{dataset}.pdf"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# RESULTS PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────


def save_results(all_results: dict[tuple[str, str], dict[str, Any]], output_dir: Path) -> None:
    """Persist scalar summary to JSON and per-instance arrays to npz."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON-serialisable summary (no numpy arrays)
    summary: dict[str, Any] = {}
    for (method, dataset), res in all_results.items():
        key = f"{method}__{dataset}"
        summary[key] = {
            "credal_coverage": {k: v for k, v in res["credal_coverage"].items() if not k.startswith("per_")},
            "tv": {k: v for k, v in res["tv"].items() if k != "per_instance"},
            "cp": {
                cp_name: {metric: {k: v for k, v in vals.items() if k != "per_seed"} for metric, vals in cp_res.items()}
                for cp_name, cp_res in res["cp"].items()
            },
        }

    with (output_dir / "results.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # Full results with per-instance arrays (for significance testing)
    arrays: dict[str, np.ndarray] = {}
    for (method, dataset), res in all_results.items():
        prefix = f"{method}__{dataset}"
        cov = res["credal_coverage"]
        tv = res["tv"]
        arrays[f"{prefix}__per_ch"] = np.array(cov["per_ch"])
        arrays[f"{prefix}__per_ch_relaxed"] = np.array(cov["per_ch_relaxed"])
        arrays[f"{prefix}__per_interval"] = np.array(cov["per_interval"])
        arrays[f"{prefix}__per_efficiency"] = np.array(cov["per_efficiency"])
        arrays[f"{prefix}__per_tv"] = np.array(tv["per_instance"])
        for cp_name, cp_res in res["cp"].items():
            for metric, vals in cp_res.items():
                arrays[f"{prefix}__{cp_name}__{metric}__per_seed"] = np.array(vals["per_seed"])

    np.savez(output_dir / "results_arrays.npz", **arrays)  # ty: ignore[invalid-argument-type]
    print(f"Results saved to {output_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# STATISTICAL TESTING
# ─────────────────────────────────────────────────────────────────────────────


def bold_mask(values_list: list[np.ndarray], higher_better: bool = True) -> list[bool]:
    """Return which entries should be bolded (not significantly worse than best, p > 0.05).

    Uses a paired t-test against the best method's per-instance values.
    """
    if len(values_list) <= 1:
        return [True] * len(values_list)

    agg = np.array([v.mean() for v in values_list])
    best_idx = int(np.argmax(agg) if higher_better else np.argmin(agg))
    best_vals = values_list[best_idx]

    mask = [False] * len(values_list)
    mask[best_idx] = True
    for i, v in enumerate(values_list):
        if i == best_idx:
            continue
        _, p = stats.ttest_rel(best_vals, v)
        if p > 0.05:
            mask[i] = True
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# LATEX TABLE GENERATION
# ─────────────────────────────────────────────────────────────────────────────

_METRIC_CNAMES = {
    "ch_coverage": "CH Cov",
    "ch_relaxed_coverage": rf"CH Cov ($\varepsilon$={RELAXED_EPSILON})",
    "interval_coverage": "Int Cov",
    "credal_efficiency": r"Efficiency$\downarrow$",
}
_PER_METRIC_KEYS = {
    "ch_coverage": "per_ch",
    "ch_relaxed_coverage": "per_ch_relaxed",
    "interval_coverage": "per_interval",
    "credal_efficiency": "per_efficiency",
}
_HIGHER_BETTER = {
    "ch_coverage": True,
    "ch_relaxed_coverage": True,
    "interval_coverage": True,
    "credal_efficiency": False,
}


def _fmt(val: float, bold: bool) -> str:
    s = f"{val:.4f}"
    return rf"\textbf{{{s}}}" if bold else s


def _fmt_mean_std(mean: float, std: float, bold: bool) -> str:
    m_str = f"{mean:.4f}"
    s_str = f"{std:.4f}"
    if bold:
        return rf"\textbf{{{m_str}}} $\pm$ \textbf{{{s_str}}}"
    return rf"{m_str} $\pm$ {s_str}"


def make_table_1(all_results: dict[tuple[str, str], dict[str, Any]]) -> str:
    """Credal coverage table."""
    datasets = sorted({ds for _, ds in all_results})
    methods = sorted({m for m, _ in all_results})
    cov_metrics = ["ch_coverage", "ch_relaxed_coverage", "interval_coverage", "credal_efficiency"]

    col_header = " & ".join(_METRIC_CNAMES[m] for m in cov_metrics)
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Credal set coverage and efficiency on first-order DCIC datasets.}",
        r"\label{tab:credal_coverage}",
        r"\begin{tabular}{ll" + "c" * len(cov_metrics) + r"}",
        r"\toprule",
        r"Dataset & Method & " + col_header + r" \\",
        r"\midrule",
    ]

    for d_idx, dataset in enumerate(datasets):
        ds_methods = [m for m in methods if (m, dataset) in all_results]
        if not ds_methods:
            continue

        # Pre-compute bold masks per metric (once per dataset)
        bold_per_metric: dict[str, list[bool]] = {}
        for metric in cov_metrics:
            per_key = _PER_METRIC_KEYS[metric]
            higher = _HIGHER_BETTER[metric]
            arrays = [np.array(all_results[(m, dataset)]["credal_coverage"][per_key]) for m in ds_methods]
            bold_per_metric[metric] = bold_mask(arrays, higher_better=higher)

        row_data: dict[str, dict[str, tuple[float, bool]]] = {}
        for i, method in enumerate(ds_methods):
            row_data[method] = {}
            for metric in cov_metrics:
                val = all_results[(method, dataset)]["credal_coverage"][metric]
                row_data[method][metric] = (val, bold_per_metric[metric][i])

        n_methods = len(ds_methods)
        for i, method in enumerate(ds_methods):
            prefix = rf"\multirow{{{n_methods}}}{{*}}{{{dataset}}}" if i == 0 else ""
            mname = METHOD_DISPLAY.get(method, method)
            cells = " & ".join(_fmt(row_data[method][metric][0], row_data[method][metric][1]) for metric in cov_metrics)
            lines.append(f"  {prefix} & {mname} & {cells} \\\\")

        if d_idx < len(datasets) - 1:
            lines.append(r"\midrule")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def make_table_2(all_results: dict[tuple[str, str], dict[str, Any]]) -> str:
    """TV calibration error table."""
    datasets = sorted({ds for _, ds in all_results})
    methods = sorted({m for m, _ in all_results})

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Calibration error (TV distance, ensemble mean vs.\ true distribution).}",
        r"\label{tab:calibration_error}",
        r"\begin{tabular}{llc}",
        r"\toprule",
        r"Dataset & Method & TV (mean $\pm$ std) \\",
        r"\midrule",
    ]

    for d_idx, dataset in enumerate(datasets):
        ds_methods = [m for m in methods if (m, dataset) in all_results]
        if not ds_methods:
            continue

        tv_arrays = [np.array(all_results[(m, dataset)]["tv"]["per_instance"]) for m in ds_methods]
        bm = bold_mask(tv_arrays, higher_better=False)

        n_methods = len(ds_methods)
        for i, method in enumerate(ds_methods):
            prefix = rf"\multirow{{{n_methods}}}{{*}}{{{dataset}}}" if i == 0 else ""
            mname = METHOD_DISPLAY.get(method, method)
            tv = all_results[(method, dataset)]["tv"]
            cell = _fmt_mean_std(tv["mean"], tv["std"], bm[i])
            lines.append(f"  {prefix} & {mname} & {cell} \\\\")

        if d_idx < len(datasets) - 1:
            lines.append(r"\midrule")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


_CP_METRIC_NAMES = {
    "cond_satisfaction": r"Cond.\ Sat.",
    "avg_set_size": r"Set Size$\downarrow$",
    "marginal_coverage": r"Marg.\ Cov.",
}
_CP_HIGHER_BETTER = {
    "cond_satisfaction": True,
    "avg_set_size": False,
    "marginal_coverage": True,
}
_CP_METRIC_ORDER = ["cond_satisfaction", "avg_set_size", "marginal_coverage"]


def make_table_3(all_results: dict[tuple[str, str], dict[str, Any]]) -> str:
    """Conformal prediction table grouped by credal method."""
    datasets = sorted({ds for _, ds in all_results})
    methods = sorted({m for m, _ in all_results})
    cp_names = list(CP_METHODS.keys())

    col_header = " & ".join(cp_names)
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Conformal prediction results ($\alpha=0.1$, 10 seeds) for each credal method.}",
        r"\label{tab:conformal_prediction}",
        r"\begin{tabular}{llc" + "c" * len(cp_names) + r"}",
        r"\toprule",
        r"Dataset & Metric & " + col_header + r" \\",
        r"\midrule",
    ]

    for m_idx, method in enumerate(methods):
        mname = METHOD_DISPLAY.get(method, method)
        if m_idx > 0:
            lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{{3 + len(cp_names)}}}{{l}}{{\textit{{{mname}}}}} \\")

        for d_idx, dataset in enumerate(datasets):
            if (method, dataset) not in all_results:
                continue
            cp_res = all_results[(method, dataset)]["cp"]

            n_metrics = len(_CP_METRIC_ORDER)
            for met_idx, cp_metric in enumerate(_CP_METRIC_ORDER):
                higher = _CP_HIGHER_BETTER[cp_metric]
                # Per-seed arrays for each CP method for significance testing
                seed_arrays = [np.array(cp_res[cp_name][cp_metric]["per_seed"]) for cp_name in cp_names]
                bm = bold_mask(seed_arrays, higher_better=higher)

                prefix = rf"\multirow{{{n_metrics}}}{{*}}{{{dataset}}}" if met_idx == 0 else ""
                met_label = _CP_METRIC_NAMES[cp_metric]
                cells = " & ".join(
                    _fmt_mean_std(
                        cp_res[cp_name][cp_metric]["mean"],
                        cp_res[cp_name][cp_metric]["std"],
                        bm[j],
                    )
                    for j, cp_name in enumerate(cp_names)
                )
                lines.append(f"  {prefix} & {met_label} & {cells} \\\\")

            if d_idx < len([d for d in datasets if (method, d) in all_results]) - 1:
                lines.append(r"\cmidrule{1-" + str(3 + len(cp_names)) + r"}")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────


def _evaluate_one(
    method: str,
    dataset: str,
    run_id: str,
    device: torch.device,
) -> dict[str, Any]:
    """Run all evaluations for a single (method, dataset) W&B run."""
    print(f"\n{'=' * 60}")
    print(f"Method: {method}  |  Dataset: {dataset}  |  Run: {run_id}")
    print("=" * 60)

    print("Building test dataset ...")
    test_ds = build_test_dataset(dataset)
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    print("Computing true first-order distributions ...")
    true_probs_test = get_true_probs(test_ds)
    print(f"  True probs shape: {true_probs_test.shape}")

    print("Loading model and computing ensemble predictions ...")
    ensemble_probs = load_ensemble_probs(method, dataset, run_id, test_loader, device)
    print(f"  Ensemble probs shape: {ensemble_probs.shape}  (n_test, n_members, K)")

    print("\n[1] Credal coverage metrics ...")
    credal_cov = compute_credal_coverage_metrics(ensemble_probs, true_probs_test)

    print("\n[2] TV calibration error ...")
    tv_metrics = compute_tv_metrics(ensemble_probs, true_probs_test)
    print(f"  Mean TV: {tv_metrics['mean']:.4f} +/- {tv_metrics['std']:.4f}")

    print("\n[3] Conformal prediction (zero-order, 10 seeds) ...")
    ensemble_mean = ensemble_probs.mean(axis=1)
    cp_results = run_cp_all_seeds(ensemble_mean, true_probs_test)
    for cp_name, cp_res in cp_results.items():
        mc = cp_res["marginal_coverage"]
        ss = cp_res["avg_set_size"]
        cs = cp_res["cond_satisfaction"]
        print(
            f"  {cp_name}: marginal={mc['mean']:.3f}+/-{mc['std']:.3f}  "
            f"set_size={ss['mean']:.3f}+/-{ss['std']:.3f}  "
            f"cond_sat={cs['mean']:.3f}+/-{cs['std']:.3f}"
        )

    return {"credal_coverage": credal_cov, "tv": tv_metrics, "cp": cp_results}


def _print_tables(table1: str, table2: str, table3: str) -> None:
    """Print all three LaTeX tables to stdout."""
    for title, table in [
        ("TABLE 1: Credal Coverage", table1),
        ("TABLE 2: Calibration Error", table2),
        ("TABLE 3: Conformal Prediction", table3),
    ]:
        print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
        print(table)


def main() -> None:
    """Entry point: run evaluations for all configured (method, dataset) pairs."""
    if not WANDB_RUN_IDS:
        print("WANDB_RUN_IDS is empty. Fill in the configuration section and re-run.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    all_results: dict[tuple[str, str], dict[str, Any]] = {
        (method, dataset): _evaluate_one(method, dataset, run_id, device)
        for (method, dataset), run_id in WANDB_RUN_IDS.items()
    }

    print("\nSaving results ...")
    save_results(all_results, OUTPUT_DIR)

    print("Generating TV histograms ...")
    plot_tv_histograms(all_results, OUTPUT_DIR)

    print("\nGenerating LaTeX tables ...")
    table1 = make_table_1(all_results)
    table2 = make_table_2(all_results)
    table3 = make_table_3(all_results)

    tables_tex = f"{table1}\n\n{table2}\n\n{table3}"
    (OUTPUT_DIR / "tables.tex").write_text(tables_tex)
    print(f"Tables saved to {OUTPUT_DIR}/tables.tex\n")

    _print_tables(table1, table2, table3)


if __name__ == "__main__":
    main()
