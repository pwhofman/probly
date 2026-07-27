"""Credal-set and conformal analyses on first-order DCIC data using probly built-ins.

Usage (run from the repo root, not this directory -- needs the probly_benchmark
package and its dependencies):
    1. Fill in WANDB_RUN_IDS below.
    2. uv run python experiments/first_order_data/paper_comparisons.py
    3. Results -> OUTPUT_DIR/results.json  +  results_arrays.npz
    4. Plots   -> OUTPUT_DIR/plots/
    5. Tables  -> OUTPUT_DIR/tables.tex (also stdout)
"""

from __future__ import annotations

from collections import defaultdict
import itertools
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from scipy import stats
import torch
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from collections.abc import Callable

from probly.conformal_scores.aps._common import compute_aps_score_numpy
from probly.conformal_scores.lac._common import compute_lac_score_numpy
from probly.conformal_scores.raps._common import compute_raps_score_numpy
from probly.conformal_scores.saps._common import compute_saps_score_func_numpy
from probly.metrics._common import CREDAL_ROUND_DECIMALS
from probly.metrics.array import (
    _convex_hull_lp_coverage,
    _credal_containment_coverage,
    _credal_interval_efficiency,
)
from probly.plot.config import PlotConfig, _apply_rc_defaults
from probly.utils.quantile._common import calculate_quantile
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

WANDB_RUN_IDS: dict[tuple[str, str], str] = {
    ("credal_relative_likelihood", "qualitymri"): "e196f1oh",
    ("credal_relative_likelihood", "micebone"): "3l1xzrgs",
    ("credal_relative_likelihood", "synthetic"): "kdi08bpk",
    ("credal_ensembling", "qualitymri"): "xps1ojts",
    ("credal_ensembling", "micebone"): "26t6udc5",
    ("credal_ensembling", "synthetic"): "05c9j9gg",
    ("credal_wrapper", "qualitymri"): "7icmxqxs",
    ("credal_wrapper", "micebone"): "g8snvupb",
    ("credal_wrapper", "synthetic"): "abt7cq1t",
}

ALPHA = 0.1
CP_N_SEEDS = 10
CP_CAL_FRAC = 0.5
ECE_N_SEEDS = 10
ECE_N_BINS = 15
CH_EPSILON = 0.0005  # matches CREDAL_ROUND_DECIMALS=3 tolerance

# Which credal method's ensemble mean feeds the LAC/APS/RAPS/SAPS conformal
# prediction table (make_table_3). CreEns is a plain deep ensemble, the most
# natural point-probability source for conformal prediction.
CP_TABLE_METHOD = "credal_ensembling"

OUTPUT_DIR = Path(__file__).parent / "first_order_results_probly"

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

_CFG = PlotConfig()
_apply_rc_defaults()

# ─────────────────────────────────────────────────────────────────────────────
# DATA UTILITIES
# ─────────────────────────────────────────────────────────────────────────────


def build_test_dataset(dataset: str) -> _DcicZeroOrderDataset:
    """Build a zero-order test dataset for fold 1 of a DCIC dataset."""
    folder = _DCIC_FOLDER_NAME[dataset]
    root = DCIC_IMAGE_PATH / folder
    return _DcicZeroOrderDataset(root, folds=[1], transform=TRANSFORMS_TEST["imagenet"], seed=SEED)


def get_true_probs(test_ds: _DcicZeroOrderDataset) -> np.ndarray:
    """Empirical p(y|x) from annotator counts; shape (N, K)."""
    nc = test_ds.num_classes
    true_probs = np.zeros((len(test_ds), nc), dtype=np.float64)
    for i, ann in enumerate(test_ds._annotation_lists):  # noqa: SLF001
        counts = np.bincount(ann, minlength=nc).astype(float)
        true_probs[i] = counts / counts.sum()
    return true_probs


def load_ensemble_probs(
    method_name: str,
    dataset: str,
    run_id: str,
    test_loader: DataLoader,  # type: ignore[type-arg]
    device: torch.device,
) -> np.ndarray:
    """Return ensemble softmax probs; shape (N, M, K)."""
    cfg = OmegaConf.create(
        {
            "dataset": dataset,
            "base_model": BASE_MODEL,
            "model_type": MODEL_TYPE,
            "pretrained": PRETRAINED,
            "seed": SEED,
            "val_split": VAL_SPLIT,
            "cal_split": 0.0,
            "method": {"name": method_name},
            "wandb": {"run_id": run_id, "entity": WANDB_ENTITY, "project": WANDB_PROJECT},
        }
    )
    model, _, _ = load_model_for_evaluation(cfg, device)
    # credal_relative_likelihood builds its ensemble as a plain list of members
    # rather than an nn.ModuleList; each member is eval()'d individually below.
    members = list(model) if isinstance(model, torch.nn.ModuleList | list) else [model]
    logits_list: list[torch.Tensor] = []
    for m in members:
        m.eval()
        logits_m, _ = collect_outputs_targets_raw(m, test_loader, device)
        logits_list.append(logits_m)
    return torch.stack(logits_list).softmax(dim=-1).permute(1, 0, 2).numpy()  # (N, M, K)


# ─────────────────────────────────────────────────────────────────────────────
# CREDAL COVERAGE & EFFICIENCY
# ─────────────────────────────────────────────────────────────────────────────


def compute_credal_metrics(ensemble_probs: np.ndarray, true_probs: np.ndarray) -> dict[str, Any]:
    """Convex-hull coverage (eps=CH_EPSILON), interval coverage, efficiency.

    Uses probly's _convex_hull_lp_coverage, _credal_containment_coverage,
    _credal_interval_efficiency.

    Args:
        ensemble_probs: (N, M, K)
        true_probs: (N, K)

    Returns:
        Dict with scalar metrics and per-instance arrays.
    """
    print("  Convex hull coverage (LP) ...")
    ch_cov = float(_convex_hull_lp_coverage(ensemble_probs, true_probs, epsilon=CH_EPSILON))
    print(f"    CH coverage (eps={CH_EPSILON}): {ch_cov:.4f}")

    lower = ensemble_probs.min(axis=1)  # (N, K)
    upper = ensemble_probs.max(axis=1)  # (N, K)

    print("  Interval coverage ...")
    int_cov = float(_credal_containment_coverage(lower, upper, true_probs))
    print(f"    Interval coverage: {int_cov:.4f}")

    print("  Credal efficiency ...")
    eff = float(_credal_interval_efficiency(lower, upper))
    print(f"    Efficiency (higher=better): {eff:.4f}")

    # Per-instance arrays for significance testing
    r = CREDAL_ROUND_DECIMALS
    lower_r = np.round(lower, decimals=r)
    upper_r = np.round(upper, decimals=r)
    per_interval = np.all((lower_r <= true_probs) & (true_probs <= upper_r), axis=-1).astype(float)
    per_eff = 1.0 - (upper_r - lower_r).mean(axis=-1)

    # Per-instance CH (reuse LP loop result via aggregate only; approximate per-instance as indicator)
    # We recompute per-instance here for significance testing
    from scipy.optimize import linprog  # noqa: PLC0415

    n, n_v, n_k = ensemble_probs.shape
    c_rel = np.concatenate([np.zeros(n_v), np.ones(2 * n_k)])
    bounds_rel = [(0, 1)] * n_v + [(0, None)] * (2 * n_k)
    per_ch = np.zeros(n)
    for i in range(n):
        a_top = np.hstack([ensemble_probs[i].T, np.eye(n_k), -np.eye(n_k)])
        a_bot = np.concatenate([np.ones(n_v), np.zeros(2 * n_k)])
        a_eq = np.vstack([a_top, a_bot])
        b_eq = np.concatenate([true_probs[i], [1]])
        res = linprog(c=c_rel, A_eq=a_eq, b_eq=b_eq, bounds=bounds_rel)
        per_ch[i] = float(res.success and res.fun <= CH_EPSILON)

    return {
        "ch_coverage": ch_cov,
        "interval_coverage": int_cov,
        "efficiency": eff,
        "per_ch": per_ch.tolist(),
        "per_interval": per_interval.tolist(),
        "per_eff": per_eff.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION - L1 AND ECE
# ─────────────────────────────────────────────────────────────────────────────


def _ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = ECE_N_BINS) -> float:
    """Standard ECE with equal-width confidence bins."""
    confs = probs.max(axis=-1)
    preds = probs.argmax(axis=-1)
    acc = (preds == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for lo, hi in itertools.pairwise(edges):
        mask = (confs >= lo) & (confs < hi)
        if mask.sum():
            ece += mask.sum() / n * abs(acc[mask].mean() - confs[mask].mean())
    return float(ece)


def compute_calibration_metrics(
    ensemble_probs: np.ndarray,
    true_probs: np.ndarray,
    n_ece_seeds: int = ECE_N_SEEDS,
) -> dict[str, Any]:
    """L1 distance (ensemble mean vs true) and ECE (zero-order labels).

    Args:
        ensemble_probs: (N, M, K)
        true_probs: (N, K)
        n_ece_seeds: Number of label-sampling seeds for ECE.

    Returns:
        Dict with l1 mean/std (per_instance) and ece mean/std (per_seed).
    """
    mean_probs = ensemble_probs.mean(axis=1)  # (N, K)
    per_l1 = np.abs(mean_probs - true_probs).sum(axis=-1)  # L1, not TV
    per_tv = 0.5 * per_l1  # TV(target, ensemble) = 0.5 * L1

    ece_vals = []
    num_classes = true_probs.shape[1]
    for seed in range(n_ece_seeds):
        rng = np.random.default_rng(seed)
        labels = np.array([rng.choice(num_classes, p=true_probs[i]) for i in range(len(true_probs))])
        ece_vals.append(_ece(mean_probs, labels))

    return {
        "l1_mean": float(per_l1.mean()),
        "l1_std": float(per_l1.std()),
        "per_l1": per_l1.tolist(),
        "tv_mean": float(per_tv.mean()),
        "tv_std": float(per_tv.std()),
        "per_tv": per_tv.tolist(),
        "ece_mean": float(np.mean(ece_vals)),
        "ece_std": float(np.std(ece_vals)),
        "ece_per_seed": ece_vals,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CONFORMAL PREDICTION
# ─────────────────────────────────────────────────────────────────────────────


def _run_cp_single(
    score_fn: Callable[..., np.ndarray],
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    test_true_probs: np.ndarray,
    alpha: float,
) -> dict[str, float]:
    """One split-conformal trial; uses probly's calculate_quantile."""
    cal_scores = score_fn(cal_probs, cal_labels)
    q = calculate_quantile(np.asarray(cal_scores, dtype=float), alpha)
    all_scores = score_fn(test_probs)  # (n_test, K)
    pred_sets = all_scores <= q
    n = len(test_labels)
    marginal = float(pred_sets[np.arange(n), test_labels].mean())
    set_size = float(pred_sets.sum(axis=-1).mean())
    mass = (pred_sets * test_true_probs).sum(axis=-1)
    cond_sat = float((mass >= 1 - alpha).mean())
    return {"marginal_coverage": marginal, "avg_set_size": set_size, "cond_satisfaction": cond_sat}


def run_cp_all_seeds(
    ensemble_mean: np.ndarray,
    true_probs_test: np.ndarray,
    alpha: float = ALPHA,
    n_seeds: int = CP_N_SEEDS,
    cal_frac: float = CP_CAL_FRAC,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Run CP for all score functions and seeds."""
    seed_results: dict[str, list[dict[str, float]]] = defaultdict(list)
    num_classes = true_probs_test.shape[1]
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        labels = np.array([rng.choice(num_classes, p=true_probs_test[i]) for i in range(len(true_probs_test))])
        n = len(labels)
        idx = rng.permutation(n)
        n_cal = int(n * cal_frac)
        ci, ti = idx[:n_cal], idx[n_cal:]
        for cp_name, fn in CP_METHODS.items():
            seed_results[cp_name].append(
                _run_cp_single(
                    fn,
                    ensemble_mean[ci],
                    labels[ci],
                    ensemble_mean[ti],
                    labels[ti],
                    true_probs_test[ti],
                    alpha,
                )
            )
    agg: dict[str, dict[str, dict[str, Any]]] = {}
    for cp_name, sl in seed_results.items():
        agg[cp_name] = {}
        for metric in ["marginal_coverage", "avg_set_size", "cond_satisfaction"]:
            vals = [s[metric] for s in sl]
            agg[cp_name][metric] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "per_seed": vals}
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

_HIST_COLOR = _CFG.color_negative  # "#1e88e5"


def _tv_hist(l1_distances: np.ndarray, title: str, out_path: Path) -> None:
    """Save one TV/L1 histogram PDF using the probly Fira Sans style."""
    plt.rcParams.update(
        {
            "font.family": "Fira Sans",
            "font.size": 8,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.labelweight": "semibold",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "font.weight": "light",
        }
    )
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.hist(l1_distances, bins=10, color=_HIST_COLOR, alpha=0.7, edgecolor="white")
    mean_val = l1_distances.mean()
    ax.axvline(mean_val, linestyle="--", color="k", linewidth=1.5)
    ax.set_xlabel("L1 Distance", fontweight="semibold")
    ax.set_xticks(np.arange(0, 2.5, 0.5))
    ax.set_title(title, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_l1_histograms(all_results: dict[tuple[str, str], dict[str, Any]], output_dir: Path) -> None:
    """Save one PDF histogram per (dataset, credal-method) pair."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for (method, dataset), res in all_results.items():
        l1 = np.array(res["calibration"]["per_l1"])
        mname = METHOD_DISPLAY.get(method, method)
        title = f"{dataset} / {mname}"
        out = plots_dir / f"l1_hist_{dataset}_{method}.pdf"
        _tv_hist(l1, title, out)
        print(f"  Saved: {out}")


def plot_cp_scatter(all_results: dict[tuple[str, str], dict[str, Any]], output_dir: Path) -> None:
    """Scatter: x=set size, y=cond coverage violation, per (credal_method, dataset).

    One subplot per (credal_method x dataset) combination.
    """
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    methods = sorted({m for m, _ in all_results})
    datasets = sorted({d for _, d in all_results})
    cp_names = list(CP_METHODS.keys())
    colors = [_CFG.color(i) for i in range(len(cp_names))]

    ncols = len(datasets)
    nrows = len(methods)
    if nrows == 0 or ncols == 0:
        return

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.5 * nrows), squeeze=False)
    plt.rcParams.update(
        {
            "font.family": "Fira Sans",
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.labelweight": "semibold",
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "font.weight": "light",
        }
    )

    for r, method in enumerate(methods):
        for c, dataset in enumerate(datasets):
            ax = axes[r][c]
            key = (method, dataset)
            if key not in all_results:
                ax.set_visible(False)
                continue
            cp_res = all_results[key]["cp"]
            for j, cp_name in enumerate(cp_names):
                ss_mean = cp_res[cp_name]["avg_set_size"]["mean"]
                ss_std = cp_res[cp_name]["avg_set_size"]["std"]
                sat_mean = cp_res[cp_name]["cond_satisfaction"]["mean"]
                sat_std = cp_res[cp_name]["cond_satisfaction"]["std"]
                viol_mean = (1 - ALPHA) - sat_mean  # violation
                ax.errorbar(
                    ss_mean,
                    viol_mean,
                    xerr=ss_std,
                    yerr=sat_std,
                    fmt="o",
                    color=colors[j],
                    label=cp_name,
                    capsize=3,
                    markersize=5,
                    linewidth=1,
                )
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
            ax.set_title(f"{METHOD_DISPLAY.get(method, method)} / {dataset}", fontsize=8)
            ax.set_xlabel("Avg Set Size", fontweight="semibold")
            ax.set_ylabel("Cov. Violation", fontweight="semibold")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if r == 0 and c == ncols - 1:
                ax.legend(fontsize=7, frameon=False)

    plt.tight_layout()
    out = plots_dir / "cp_scatter.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# RESULTS PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────


def save_results(all_results: dict[tuple[str, str], dict[str, Any]], output_dir: Path) -> None:
    """Save scalar JSON summary + per-instance npz arrays."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    for (method, dataset), res in all_results.items():
        key = f"{method}__{dataset}"
        cov = res["credal_coverage"]
        cal = res["calibration"]
        summary[key] = {
            "credal_coverage": {k: v for k, v in cov.items() if not k.startswith("per_")},
            "calibration": {k: v for k, v in cal.items() if not k.startswith("per_") and k != "ece_per_seed"},
            "cp": {
                cp: {m: {kk: vv for kk, vv in vals.items() if kk != "per_seed"} for m, vals in cr.items()}
                for cp, cr in res["cp"].items()
            },
        }
        arrays[f"{key}__per_ch"] = np.array(cov["per_ch"])
        arrays[f"{key}__per_interval"] = np.array(cov["per_interval"])
        arrays[f"{key}__per_eff"] = np.array(cov["per_eff"])
        arrays[f"{key}__per_l1"] = np.array(cal["per_l1"])
        arrays[f"{key}__per_tv"] = np.array(cal["per_tv"])
        for cp_name, cp_res in res["cp"].items():
            for metric, vals in cp_res.items():
                arrays[f"{key}__{cp_name}__{metric}"] = np.array(vals["per_seed"])

    with (output_dir / "results.json").open("w") as f:
        json.dump(summary, f, indent=2)
    np.savez(output_dir / "results_arrays.npz", **arrays)  # ty: ignore[invalid-argument-type]
    print(f"  Results saved to {output_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# STATISTICAL TESTING
# ─────────────────────────────────────────────────────────────────────────────


def bold_mask(values_list: list[np.ndarray], higher_better: bool = True) -> list[bool]:
    """Paired t-test; bold if not significantly worse than best (p > 0.05)."""
    if len(values_list) <= 1:
        return [True] * len(values_list)
    agg = np.array([v.mean() for v in values_list])
    best_idx = int(np.argmax(agg) if higher_better else np.argmin(agg))
    mask = [False] * len(values_list)
    mask[best_idx] = True
    for i, v in enumerate(values_list):
        if i == best_idx:
            continue
        _, p = stats.ttest_rel(values_list[best_idx], v)
        if p > 0.05:
            mask[i] = True
    return mask


def bold_max_only(values: list[float], higher_better: bool = True) -> list[bool]:
    """Bold only the strict best value(s); no significance test.

    Used for Table 1's credal-coverage metrics (ch_coverage/interval_coverage),
    which are rare-event binary indicators per instance (often ~1-2% positive
    rate) -- a paired t-test is underpowered for those regardless of sample
    size and over-bolds non-best values, unlike the continuous per-seed/
    per-instance metrics bold_mask is used for elsewhere.
    """
    best = max(values) if higher_better else min(values)
    return [v == best for v in values]


# ─────────────────────────────────────────────────────────────────────────────
# LATEX TABLE GENERATION
# ─────────────────────────────────────────────────────────────────────────────


def _fmt_val(val: float, bold: bool, decimals: int = 3) -> str:
    r"""Format a single scalar as a LaTeX math cell, e.g. ``$0.877$`` or ``$\mathbf{0.877}$``."""
    body = f"{val:.{decimals}f}"
    if bold:
        body = rf"\mathbf{{{body}}}"
    return f"${body}$"


def _fmt_pm(mean: float, std: float, bold: bool, decimals: int = 3, *, strip_zero: bool = False) -> str:
    r"""Format a mean +/- std pair as ``$mean_{\pm std}$``, optionally bolded and without a leading zero."""
    m = f"{mean:.{decimals}f}"
    s = f"{std:.{decimals}f}"
    if strip_zero:
        m = m.removeprefix("0")
        s = s.removeprefix("0")
    body = f"{m}_{{\\pm {s}}}"
    if bold:
        body = rf"\mathbf{{{body}}}"
    return f"${body}$"


def make_table_1(all_results: dict[tuple[str, str], dict[str, Any]]) -> str:
    """Credal-set coverage & efficiency table: CH Cov / Int. Cov. / Efficiency, grouped by dataset."""
    datasets = sorted({d for _, d in all_results})
    methods = sorted({m for m, _ in all_results})
    metrics = [
        ("ch_coverage", True),
        ("interval_coverage", True),
        ("efficiency", True),
    ]

    header = [
        r"\begin{tabular}{@{}l@{\hspace{2pt}}cccc@{}}",
        r"\toprule",
        r"\multicolumn{2}{@{}l}{\textbf{Method}}",
        r"&\makecell[c]{\textbf{ConvexHull}\\[-1pt]\textbf{Coverage ($\uparrow$)}}",
        r"&\makecell[c]{\textbf{Interval}\\[-1pt]\textbf{Coverage ($\uparrow$)}}",
        r"&\textbf{Efficiency} \textbf{($\uparrow$)}\\",
        r"\hdashline\noalign{\vskip0.45em}",
    ]

    dataset_blocks: list[str] = []
    ds_with_data = [d for d in datasets if any((m, d) in all_results for m in methods)]
    for d_idx, dataset in enumerate(ds_with_data):
        ds_methods = [m for m in methods if (m, dataset) in all_results]
        bold_per: dict[str, list[bool]] = {}
        for key, higher in metrics:
            vals = [all_results[(m, dataset)]["credal_coverage"][key] for m in ds_methods]
            bold_per[key] = bold_max_only(vals, higher_better=higher)
        dataset_blocks += [
            r"\rowcolor[gray]{0.92}[0pt][0pt]",
            rf"\multicolumn{{5}}{{@{{}}l@{{}}}}{{\textbf{{{dataset}}}}}\\",
            r"\addlinespace[0.15em]",
        ]
        for i, method in enumerate(ds_methods):
            mname = METHOD_DISPLAY.get(method, method)
            cells = "&".join(
                _fmt_val(all_results[(method, dataset)]["credal_coverage"][key], bold_per[key][i])
                for key, _ in metrics
            )
            dataset_blocks.append(f"{mname}&&{cells}\\\\")
        if d_idx < len(ds_with_data) - 1:
            dataset_blocks.append(r"\addlinespace[0.4em]")

    footer = [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(header + dataset_blocks + footer)


def make_table_tv(all_results: dict[tuple[str, str], dict[str, Any]]) -> str:
    """TV-distance comparison table: dataset rows x credal-method columns."""
    datasets = sorted({d for _, d in all_results})
    methods = sorted({m for m, _ in all_results})
    col_hdr = " & ".join(rf"\textbf{{{METHOD_DISPLAY.get(m, m)}}} ($\downarrow$)" for m in methods)

    lines = [
        r"\begin{tabular}{@{}l" + "c" * len(methods) + r"@{}}",
        r"\toprule",
        rf"\textbf{{Dataset}} & {col_hdr} \\",
        r"\hdashline\noalign{\vskip 0.45em}",
        r"\midrule",
    ]
    for dataset in datasets:
        ds_methods = [m for m in methods if (m, dataset) in all_results]
        arrs = [np.array(all_results[(m, dataset)]["calibration"]["per_tv"]) for m in ds_methods]
        bm = bold_mask(arrs, higher_better=False)
        cells = " & ".join(
            _fmt_pm(
                all_results[(m, dataset)]["calibration"]["tv_mean"],
                all_results[(m, dataset)]["calibration"]["tv_std"],
                bm[i],
                strip_zero=True,
            )
            for i, m in enumerate(ds_methods)
        )
        lines.append(f"{dataset} & {cells} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


_CP_METRIC_ORDER = [
    ("cond_satisfaction", r"Cond.\ Sat.", True),
    ("avg_set_size", "Set Size", False),
    ("marginal_coverage", r"Marg.\ Cov.", True),
]

_CP_HEADER = (
    r"\multicolumn{2}{@{}l}{\textbf{Metric}} & \textbf{LAC} \cite{sadinle2019least}"
    r"& \textbf{APS} \cite{romanoAPS2020}& \textbf{RAPS} \cite{angelopoulosRAPS2021} "
    r"& \textbf{SAPS} \cite{huang2024conformal} \\"
)


def make_table_3(all_results: dict[tuple[str, str], dict[str, Any]], cp_method: str = CP_TABLE_METHOD) -> str:
    """CP table: Cond. Sat. / Set Size / Marg. Cov. for LAC/APS/RAPS/SAPS, using cp_method's ensemble mean."""
    datasets = [d for d in sorted({d for _, d in all_results}) if (cp_method, d) in all_results]
    cp_names = list(CP_METHODS.keys())

    header = [
        r"\begin{tabular}{@{}l@{\hspace{2pt}}c c c c c@{}}",
        r"\toprule",
        _CP_HEADER,
        r"\hdashline\noalign{\vskip 0.45em}",
    ]

    dataset_blocks: list[str] = []
    for d_idx, dataset in enumerate(datasets):
        cp_res = all_results[(cp_method, dataset)]["cp"]
        dataset_blocks += [
            r"\rowcolor[gray]{0.92}[0pt][0pt]",
            rf"\multicolumn{{6}}{{@{{}}l@{{}}}}{{\textbf{{{dataset}}}}} \\",
            r"\addlinespace[0.15em]",
        ]
        for cp_metric, label, higher in _CP_METRIC_ORDER:
            # never bold marginal coverage
            if cp_metric == "marginal_coverage":
                bm = [False] * len(cp_names)
            else:
                seed_arrs = [np.array(cp_res[cn][cp_metric]["per_seed"]) for cn in cp_names]
                bm = bold_mask(seed_arrs, higher_better=higher)
            arrow = r"$\uparrow$" if higher else r"$\downarrow$"
            cells = " & ".join(
                _fmt_pm(cp_res[cn][cp_metric]["mean"], cp_res[cn][cp_metric]["std"], bm[j])
                for j, cn in enumerate(cp_names)
            )
            dataset_blocks.append(rf"{label} & {arrow}   & {cells} \\")
        if d_idx < len(datasets) - 1:
            dataset_blocks.append(r"\addlinespace[0.4em]")

    footer = [r"\bottomrule", r"\vspace{-0.4cm}", r"\end{tabular}"]
    return "\n".join(header + dataset_blocks + footer)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────


def _evaluate_one(method: str, dataset: str, run_id: str, device: torch.device) -> dict[str, Any]:
    """Run all evaluations for one (method, dataset) W&B run."""
    print(f"\n{'=' * 60}")
    print(f"Method: {method}  |  Dataset: {dataset}  |  Run: {run_id}")
    print("=" * 60)
    test_ds = build_test_dataset(dataset)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)
    true_probs = get_true_probs(test_ds)
    print(f"  True probs: {true_probs.shape}")
    ensemble_probs = load_ensemble_probs(method, dataset, run_id, test_loader, device)
    print(f"  Ensemble: {ensemble_probs.shape}")

    print("\n[1] Credal coverage ...")
    cov = compute_credal_metrics(ensemble_probs, true_probs)

    print("\n[2] Calibration (L1 + ECE) ...")
    cal = compute_calibration_metrics(ensemble_probs, true_probs)
    print(f"  L1: {cal['l1_mean']:.4f} +/- {cal['l1_std']:.4f}  |  ECE: {cal['ece_mean']:.4f} +/- {cal['ece_std']:.4f}")

    print("\n[3] Conformal prediction ...")
    ensemble_mean = ensemble_probs.mean(axis=1)
    cp = run_cp_all_seeds(ensemble_mean, true_probs)
    for cp_name, cr in cp.items():
        mc = cr["marginal_coverage"]
        ss = cr["avg_set_size"]
        cs = cr["cond_satisfaction"]
        print(
            f"  {cp_name}: marg={mc['mean']:.3f}+/-{mc['std']:.3f}  "
            f"size={ss['mean']:.3f}+/-{ss['std']:.3f}  "
            f"cond_sat={cs['mean']:.3f}+/-{cs['std']:.3f}"
        )

    return {"credal_coverage": cov, "calibration": cal, "cp": cp}


def main() -> None:
    """Entry point."""
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

    print("Generating L1 histograms ...")
    plot_l1_histograms(all_results, OUTPUT_DIR)

    print("Generating CP scatter plots ...")
    plot_cp_scatter(all_results, OUTPUT_DIR)

    print("\nGenerating LaTeX tables ...")
    t1 = make_table_1(all_results)
    t2 = make_table_tv(all_results)
    t3 = make_table_3(all_results)
    tables_tex = f"{t1}\n\n{t2}\n\n{t3}"
    (OUTPUT_DIR / "tables.tex").write_text(tables_tex)
    print(f"Tables saved to {OUTPUT_DIR}/tables.tex")

    for title, table in [
        ("TABLE 1: Credal Coverage & Efficiency", t1),
        ("TABLE 2: TV Distance", t2),
        ("TABLE 3: Conformal Prediction", t3),
    ]:
        print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
        print(table)


if __name__ == "__main__":
    main()
