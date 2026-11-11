"""Accuracy-rejection curves for selective prediction, one figure per target.

Reads only the local cache. Run ``python -m paper_plots.fetch_sp`` first.

Without arguments, every method present in the SP cache is overlaid::

    uv run python -m paper_plots.sp_arc

To restrict to a subset of UQ methods, pass them via ``--methods``. ``base``
is *always* included (drawn behind, in neutral grey)::

    uv run python -m paper_plots.sp_arc --methods dropout ensemble laplace
"""

from __future__ import annotations

import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np

from paper_plots import cache
from paper_plots.aggregate import sp_ranking
from paper_plots.metadata import EXCLUDED_METHODS, dataset_display, method_label
from paper_plots.paths import PAPER_ARTIFACTS
from paper_plots.style import PlotConfig
from paper_plots.targets import ENTITY, TARGETS, Target

MEASURE = "default"
DECOMPOSITION = "total"
# Shared single-panel default with the other "squarish" plots
# (coverage_efficiency, al_curves, ood_hist).
FIGSIZE: tuple[float, float] = (4.5, 3.6)
# Default top-K and bottom-K UQ methods to plot per target. Both default to
# ``None`` = every cached method. When set, the union is plotted (top first,
# then any bottom methods not already in the top). ``base`` is always plotted
# regardless. Overridden by ``--methods`` on the CLI.
TOP_K: int | None = None
BOTTOM_K: int | None = None


def _curve_for(target: Target, method: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int] | None:
    """Return ``(x, mean_acc, std_acc, auroc_acc, n_seeds)`` for ``method`` or ``None``."""
    bin_losses_key = f"sp/{MEASURE}/{DECOMPOSITION}/bin_losses"
    auroc_key = f"sp/{MEASURE}/{DECOMPOSITION}/auroc"
    records = cache.load_records(ENTITY, target.project, target.dataset, target.base_model, method)
    bin_losses_list = [r["summary"][bin_losses_key] for r in records if bin_losses_key in r["summary"]]
    aurocs = [r["summary"][auroc_key] for r in records if auroc_key in r["summary"]]
    if not bin_losses_list or not aurocs:
        return None
    stack = np.stack([np.asarray(bl, dtype=float) for bl in bin_losses_list])
    if stack.size == 0:
        return None
    mean_acc = 1.0 - stack.mean(axis=0)
    std_acc = stack.std(axis=0)
    auroc_acc = 1.0 - float(np.mean(aurocs))
    x = np.linspace(0.0, 1.0, mean_acc.shape[0])
    return x, mean_acc, std_acc, auroc_acc, stack.shape[0]


def _render_one(target: Target, methods: list[str] | None = None) -> None:  # noqa: C901, PLR0912, PLR0915
    cfg = PlotConfig()
    available = [
        m
        for m in cache.list_methods(ENTITY, target.project, target.dataset, target.base_model)
        if m not in EXCLUDED_METHODS
    ]
    if not available:
        warnings.warn(f"no cached SP runs for {target.dataset}/{target.base_model}", stacklevel=2)
        return

    available_set = set(available)
    if methods is None:
        if TOP_K is None and BOTTOM_K is None:
            uq_methods = [m for m in available if m != "base"]
        else:
            # Top-K from the SP Acc-AUC ranking (descending) plus bottom-K from
            # the same ranking, merged preserving order and deduplicated.
            ranked = [r["method"] for r in sp_ranking(target) if r["method"] != "base"]
            ranked = [m for m in ranked if m in available_set]
            top = ranked[:TOP_K] if TOP_K else []
            bot = ranked[-BOTTOM_K:] if BOTTOM_K else []
            seen: set[str] = set()
            uq_methods = []
            for m in (*top, *bot):
                if m not in seen:
                    seen.add(m)
                    uq_methods.append(m)
    else:
        # Drop unknown names with a warning so typos don't fail silently.
        uq_methods: list[str] = []
        for m in methods:
            if m == "base":
                continue
            if m in available_set:
                uq_methods.append(m)
            else:
                warnings.warn(
                    f"requested method {m!r} not in cache for {target.dataset}; skipping",
                    stacklevel=2,
                )

    fig, ax = plt.subplots(figsize=FIGSIZE)
    drawn = 0

    # ``base`` always plotted, neutral grey, dashed, behind the UQ curves.
    if "base" in available_set:
        result = _curve_for(target, "base")
        if result is not None:
            x, mean_acc, std_acc, auroc_acc, n_seeds = result
            ax.plot(
                x,
                mean_acc,
                label=f"{method_label('base')} (AUROC={auroc_acc:.3f})",
                color=cfg.color_neutral,
                linestyle="--",
                linewidth=cfg.line_width,
                zorder=1,
            )
            if n_seeds > 1:
                ax.fill_between(
                    x,
                    mean_acc - std_acc,
                    mean_acc + std_acc,
                    color=cfg.color_neutral,
                    alpha=cfg.fill_alpha * 0.5,
                    linewidth=0,
                    zorder=0,
                )
            drawn += 1

    # UQ methods (everything except base).
    for idx, method in enumerate(uq_methods):
        result = _curve_for(target, method)
        if result is None:
            continue
        x, mean_acc, std_acc, auroc_acc, n_seeds = result
        color = cfg.color(idx)
        ax.plot(
            x,
            mean_acc,
            label=f"{method_label(method)} (AUROC={auroc_acc:.3f})",
            color=color,
            linewidth=cfg.line_width,
            zorder=2,
        )
        if n_seeds > 1:
            ax.fill_between(
                x,
                mean_acc - std_acc,
                mean_acc + std_acc,
                color=color,
                alpha=cfg.fill_alpha,
                linewidth=0,
                zorder=1,
            )
        drawn += 1

    if drawn == 0:
        warnings.warn(f"no usable bin_losses for {target.dataset}", stacklevel=2)
        plt.close(fig)
        return

    ax.set_xlabel("Rejection rate")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0.0, 1.0)
    ax.set_title(f"Accuracy-rejection curve on {dataset_display(target.dataset)}")
    ax.grid(visible=True, linestyle=cfg.grid_linestyle, alpha=cfg.grid_alpha, color=cfg.color_gridline)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=8, ncol=1)
    fig.tight_layout()

    out_dir = PAPER_ARTIFACTS / f"{target.dataset}_methods"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"sp_arc_{target.dataset}.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


def main(methods: list[str] | None = None) -> None:
    for target in TARGETS:
        _render_one(target, methods)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Selective-prediction accuracy-rejection curves.")
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="UQ method names to plot (e.g. dropout ensemble laplace). 'base' is always included. "
        "Default: every method present in the SP cache (or top-K if --top-k is set).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="When --methods is omitted, plot the top-K UQ methods by SP Acc-AUC per target.",
    )
    parser.add_argument(
        "--bottom-k",
        type=int,
        default=None,
        help="When --methods is omitted, also plot the bottom-K UQ methods by SP Acc-AUC. "
        "Combined with --top-k by union (deduplicated).",
    )
    args = parser.parse_args()
    if args.top_k is not None:
        TOP_K = args.top_k
    if args.bottom_k is not None:
        BOTTOM_K = args.bottom_k
    main(methods=args.methods)
