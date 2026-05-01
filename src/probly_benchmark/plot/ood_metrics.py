"""OOD detection metrics comparison: grouped bar chart and LaTeX table."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf

from probly_benchmark.paths import FIGURE_PATH
from probly_benchmark.plot.utils import fetch_ood_runs, resolve_label

if TYPE_CHECKING:
    from matplotlib.figure import Figure

_CONFIG_DIR = Path(__file__).parent.parent / "configs"


def _make_bar_chart(
    method_labels: list[str],
    per_method: list[dict[str, tuple[float, float]]],
    metric_keys: list[str],
) -> Figure:
    """Build a grouped bar chart with one subplot per metric.

    Args:
        method_labels: Display labels for the methods on the x-axis.
        per_method: Aggregated metrics per method as ``{metric: (mean, std)}``.
        metric_keys: Ordered list of metrics to plot, one subplot each.

    Returns:
        The matplotlib Figure containing all metric subplots.
    """
    n_metrics = len(metric_keys)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 4), squeeze=False)
    x = np.arange(len(method_labels))

    for ax, metric in zip(axes[0], metric_keys, strict=True):
        means = [per_method[i].get(metric, (float("nan"), 0.0))[0] for i in range(len(method_labels))]
        stds = [per_method[i].get(metric, (float("nan"), 0.0))[1] for i in range(len(method_labels))]
        ax.bar(x, means, yerr=stds, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(method_labels, rotation=30, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(metric)
        ax.yaxis.grid(visible=True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

    fig.tight_layout()
    return fig


def _make_latex_table(
    method_labels: list[str],
    per_method: list[dict[str, tuple[float, float]]],
    metric_keys: list[str],
    higher_is_better: list[str],
) -> str:
    r"""Render a booktabs LaTeX table comparing methods on each metric.

    Cells are formatted as ``mean $\pm$ std`` to three decimals. The best
    value per column is bolded; for metrics in ``higher_is_better`` higher
    means are best, otherwise lower means are best.

    Args:
        method_labels: Row labels (one per method).
        per_method: Aggregated metrics per method as ``{metric: (mean, std)}``.
        metric_keys: Column metrics in the order they should appear.
        higher_is_better: Metric names for which a higher value is better.

    Returns:
        The full LaTeX table as a string.
    """
    best_idx_per_metric: dict[str, int] = {}
    for metric in metric_keys:
        means = [per_method[i].get(metric, (float("nan"), 0.0))[0] for i in range(len(method_labels))]
        valid = [(i, m) for i, m in enumerate(means) if not np.isnan(m)]
        if not valid:
            continue
        if metric in higher_is_better:
            best_idx_per_metric[metric] = max(valid, key=lambda t: t[1])[0]
        else:
            best_idx_per_metric[metric] = min(valid, key=lambda t: t[1])[0]

    col_spec = "l" + "c" * len(metric_keys)
    lines: list[str] = []
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    header = "Method & " + " & ".join(metric_keys) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")
    for i, label in enumerate(method_labels):
        cells = [label]
        for metric in metric_keys:
            mean_std = per_method[i].get(metric)
            if mean_std is None or np.isnan(mean_std[0]):
                cells.append("--")
                continue
            mean, std = mean_std
            cell = f"{mean:.3f} $\\pm$ {std:.3f}"
            if best_idx_per_metric.get(metric) == i:
                cell = r"\textbf{" + cell + "}"
            cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


@hydra.main(version_base=None, config_path="../plot_configs", config_name="ood_metrics")
def main(cfg: DictConfig) -> tuple[Figure, str]:
    """Compare OOD detection metrics across methods as a bar chart and table.

    For each method in the config, all matching seeds are fetched and the
    scalar OOD metrics are aggregated to mean and standard deviation. The
    results are rendered as a grouped bar chart (one subplot per metric) and
    as a booktabs LaTeX table with the best value per column bolded.

    Args:
        cfg: Hydra config composed from an ``ood_metrics`` comparison config.

    Returns:
        A tuple ``(fig, table)`` of the matplotlib Figure and the LaTeX table
        string.
    """
    recipe_raw = OmegaConf.load(_CONFIG_DIR / "recipe" / f"{cfg.recipe}.yaml")
    recipe = recipe_raw if isinstance(recipe_raw, DictConfig) else DictConfig({})
    ood_detection_defaults_raw = OmegaConf.load(_CONFIG_DIR / "ood_detection.yaml")
    ood_detection_defaults = (
        ood_detection_defaults_raw if isinstance(ood_detection_defaults_raw, DictConfig) else DictConfig({})
    )
    dataset: str = cfg.get("dataset") or recipe.dataset
    base_model: str = cfg.get("base_model") or recipe.base_model
    ood_dataset: str = (
        cfg.get("ood_dataset") or recipe.get("ood_dataset") or ood_detection_defaults.get("ood_dataset", "")
    )

    higher_is_better: list[str] = list(cfg.get("higher_is_better", ["auroc", "aupr"]))

    method_labels: list[str] = []
    per_method: list[dict[str, tuple[float, float]]] = []
    metric_keys_set: set[str] = set()

    for entry in cfg.methods:
        runs = fetch_ood_runs(
            cfg.wandb.entity,
            cfg.wandb.project,
            entry,
            dataset,
            ood_dataset,
            base_model,
            list(cfg.seeds) if cfg.get("seeds") else None,
        )

        per_metric_values: dict[str, list[float]] = {}
        for run in runs:
            for k, v in run["metrics"].items():
                per_metric_values.setdefault(k, []).append(v)

        aggregated: dict[str, tuple[float, float]] = {}
        for k, values in per_metric_values.items():
            arr = np.asarray(values, dtype=float)
            aggregated[k] = (float(arr.mean()), float(arr.std()))
            metric_keys_set.add(k)

        method_labels.append(resolve_label(entry))
        per_method.append(aggregated)

    metric_keys = sorted(metric_keys_set)

    fig = _make_bar_chart(method_labels, per_method, metric_keys)
    table = _make_latex_table(method_labels, per_method, metric_keys, higher_is_better)

    if cfg.get("filename") and cfg.get("filename_prefix"):
        FIGURE_PATH.mkdir(parents=True, exist_ok=True)
        stem = Path(cfg.filename).stem
        prefix = cfg.filename_prefix
        fig.savefig(FIGURE_PATH / f"{prefix}_{stem}.pdf")
        (FIGURE_PATH / f"{prefix}_{stem}.tex").write_text(table)

    if cfg.get("show", False):
        plt.show()

    return fig, table


if __name__ == "__main__":
    main()
