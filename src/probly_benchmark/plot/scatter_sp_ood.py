r"""SP vs OOD scatter plots — one PDF per ID dataset plus a combined view.

For each ID dataset (CIFAR-10, ImageNet, ...) seen in the input ranking
JSONs, draws a 2D scatter where:

- x-axis: SP performance (Acc-AUC).
- y-axis: OOD performance (mean AUROC across this dataset's near + far
  OOD datasets).

Marker shape encodes the method family:

- Methods whose name contains any ``credal_keywords`` substring (default:
  "credal") render as triangles.
- All other methods render as filled circles.

A ``both`` variant aggregates per-method scores across every dataset
(simple mean over datasets where the method has both SP and OOD data),
producing a single scatter that compares all methods regardless of ID
dataset.

Each marker is annotated with the method's display label in Fira Sans
*thin italic* just to the right of the marker. Axis limits are
auto-scaled with a small padding margin; tweak the ``inputs`` data and
the consuming script will recompute on next run.

Usage::

    uv run scatter_sp_ood.py \
        inputs='[/path/to/cifar10_methods, /path/to/imagenet_methods]' \
        save_path=/path/to/paper_figures
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf

from probly.plot.config import PlotConfig
from probly_benchmark.plot.utils import display_name, resolve_save_path

if TYPE_CHECKING:
    from collections.abc import Iterable

    from matplotlib.figure import Figure


def _normalize_inputs(value: object) -> list[Path]:
    """Coerce ``inputs`` (string or list) to a list of absolute Paths."""
    if isinstance(value, str):
        items: list[str] = [value]
    elif isinstance(value, (list, tuple)):
        items = [str(v) for v in value]
    else:
        msg = f"`inputs` must be a string or list of strings; got {type(value).__name__}."
        raise TypeError(msg)
    return [Path(p).expanduser() for p in items]


def _load_rankings(
    inputs: list[Path],
) -> tuple[dict[str, list[dict]], dict[tuple[str, str], list[dict]]]:
    """Scan ``inputs`` for ``bar_sp_*`` and ``bar_ood_*`` ranking JSONs."""
    sp: dict[str, list[dict]] = {}
    ood: dict[tuple[str, str], list[dict]] = {}
    for root in inputs:
        if not root.exists():
            continue
        for path in sorted(root.glob("bar_sp_*.ranking.json")):
            stem = path.name[: -len(".ranking.json")]
            ds = stem[len("bar_sp_") :]
            sp[ds] = json.loads(path.read_text())
        for path in sorted(root.glob("bar_ood_*_*.ranking.json")):
            stem = path.name[: -len(".ranking.json")]
            rest = stem[len("bar_ood_") :]
            band, _, ds = rest.partition("_")
            if band in ("near", "far") and ds:
                ood[(band, ds)] = json.loads(path.read_text())
    return sp, ood


def _per_band_points(
    sp_ranking: list[dict],
    band_ranking: list[dict],
) -> dict[str, dict]:
    """Build ``{method: {label, sp, ood}}`` for one ID dataset and one OOD band.

    Methods missing either the SP score or that band's OOD score are dropped.
    """
    label_lookup: dict[str, str] = {entry["method"]: entry["label"] for entry in sp_ranking}
    label_lookup.update({entry["method"]: entry["label"] for entry in band_ranking})

    sp_lookup = {entry["method"]: entry["mean"] for entry in sp_ranking}
    ood_lookup = {entry["method"]: entry["mean"] for entry in band_ranking}

    out: dict[str, dict] = {}
    for method in sp_lookup.keys() & ood_lookup.keys():
        out[method] = {
            "label": label_lookup.get(method, method),
            "sp": sp_lookup[method],
            "ood": ood_lookup[method],
        }
    return out


def _combined_band_points(per_dataset: dict[str, dict[str, dict]]) -> dict[str, dict]:
    """Average per-method (sp, ood) across every dataset that has both."""
    bag: dict[str, dict] = {}
    for points in per_dataset.values():
        for method, p in points.items():
            entry = bag.setdefault(method, {"label": p["label"], "sps": [], "oods": []})
            entry["sps"].append(p["sp"])
            entry["oods"].append(p["ood"])
    return {
        method: {
            "label": p["label"],
            "sp": sum(p["sps"]) / len(p["sps"]),
            "ood": sum(p["oods"]) / len(p["oods"]),
        }
        for method, p in bag.items()
        if p["sps"] and p["oods"]
    }


def _is_credal(method: str, keywords: Iterable[str]) -> bool:
    name = method.lower()
    return any(kw.lower() in name for kw in keywords)


def _autopad(values: list[float], frac: float = 0.05) -> tuple[float, float]:
    """Return ``(low, high)`` bracketing ``values`` with ``frac`` padding."""
    lo, hi = min(values), max(values)
    pad = max((hi - lo) * frac, 0.005)
    return lo - pad, hi + pad


def _draw_scatter(
    points: dict[str, dict],
    *,
    title: str,
    ylabel: str,
    plot_config: PlotConfig,
    credal_keywords: list[str],
    marker_size: float,
    annotation_offset_pt: tuple[float, float],
    annotation_fontsize: float,
) -> Figure:
    """Render the scatter for one variant (single dataset / band, or combined)."""
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    if not points:
        ax.set_title(title + " (no data)")
        ax.set_axis_off()
        return fig

    color = plot_config.color(0)
    sp_values: list[float] = []
    ood_values: list[float] = []
    for method, p in points.items():
        marker = "^" if _is_credal(method, credal_keywords) else "o"
        ax.scatter(
            p["sp"],
            p["ood"],
            marker=marker,
            s=marker_size,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        ax.annotate(
            p["label"],
            xy=(p["sp"], p["ood"]),
            xytext=annotation_offset_pt,
            textcoords="offset points",
            fontsize=annotation_fontsize,
            fontstyle="italic",
            fontweight=300,  # Fira Sans Light
            color="#333333",
            zorder=4,
        )
        sp_values.append(p["sp"])
        ood_values.append(p["ood"])

    ax.set_xlim(*_autopad(sp_values))
    ax.set_ylim(*_autopad(ood_values))
    ax.set_xlabel("SP performance (Acc-AUC)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(
        visible=True,
        linestyle=plot_config.grid_linestyle,
        alpha=plot_config.grid_alpha,
        color=plot_config.color_gridline,
    )
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


@hydra.main(version_base=None, config_path="../plot_configs", config_name="scatter_sp_ood")
def main(cfg: DictConfig) -> dict[str, Path]:
    """Render SP vs OOD scatter plots — per dataset + a combined ``both`` view.

    Args:
        cfg: Hydra config composed from ``scatter_sp_ood``.

    Returns:
        Mapping ``{variant_key: pdf_path}`` for every PDF written.
    """
    inputs_resolved = OmegaConf.to_container(cfg.inputs, resolve=True) if cfg.get("inputs") is not None else None
    if inputs_resolved is None:
        msg = "`inputs` is required (directory or list of directories)."
        raise ValueError(msg)
    inputs = _normalize_inputs(inputs_resolved)
    out_dir = resolve_save_path(cfg.get("save_path"))
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_config = PlotConfig()
    credal_keywords = list(cfg.get("credal_keywords") or ["credal"])
    marker_size = float(cfg.get("marker_size", 90))
    offset_raw = cfg.get("annotation_offset_pt") or (6, 5)
    offset_obj = OmegaConf.to_container(offset_raw, resolve=True) if OmegaConf.is_config(offset_raw) else offset_raw
    if not isinstance(offset_obj, (list, tuple)) or len(offset_obj) != 2:
        msg = "`annotation_offset_pt` must be a 2-element [dx, dy] list."
        raise ValueError(msg)
    annotation_offset_pt = (float(offset_obj[0]), float(offset_obj[1]))
    annotation_fontsize = float(cfg.get("annotation_fontsize", 9))

    sp, ood = _load_rankings(inputs)
    bands = ("near", "far")

    # Outer key is the OOD band, inner keys are ID datasets, leaf value is
    # ``{method: {label, sp, ood}}``.
    per_band: dict[str, dict[str, dict[str, dict]]] = {band: {} for band in bands}
    datasets = sorted(set(sp.keys()) | {ds for _band, ds in ood})
    for band in bands:
        for ds in datasets:
            per_band[band][ds] = _per_band_points(
                sp_ranking=sp.get(ds, []),
                band_ranking=ood.get((band, ds), []),
            )

    written: dict[str, Path] = {}

    for band in bands:
        band_label = f"{band.title()}-OOD"
        ylabel = f"{band_label} performance (AUROC)"
        for ds, points in per_band[band].items():
            if not points:
                continue
            fig = _draw_scatter(
                points,
                title=f"SP vs {band_label} on {display_name(ds)}",
                ylabel=ylabel,
                plot_config=plot_config,
                credal_keywords=credal_keywords,
                marker_size=marker_size,
                annotation_offset_pt=annotation_offset_pt,
                annotation_fontsize=annotation_fontsize,
            )
            path = out_dir / f"scatter_sp_vs_ood_{band}_{ds}.pdf"
            fig.savefig(path)
            plt.close(fig)
            written[f"{band}_{ds}"] = path
            print(f"Wrote {path}  ({len(points)} methods)")

        non_empty = {ds: pts for ds, pts in per_band[band].items() if pts}
        if len(non_empty) >= 2:
            combined = _combined_band_points(non_empty)
            ds_list = ", ".join(display_name(ds) for ds in sorted(non_empty))
            fig = _draw_scatter(
                combined,
                title=f"SP vs {band_label} across {ds_list}",
                ylabel=ylabel,
                plot_config=plot_config,
                credal_keywords=credal_keywords,
                marker_size=marker_size,
                annotation_offset_pt=annotation_offset_pt,
                annotation_fontsize=annotation_fontsize,
            )
            path = out_dir / f"scatter_sp_vs_ood_{band}_both.pdf"
            fig.savefig(path)
            plt.close(fig)
            written[f"{band}_both"] = path
            print(f"Wrote {path}  ({len(combined)} methods, averaged over {len(non_empty)} datasets)")

    _ = int(cfg.get("decimals", 3))  # reserved for future tick formatting
    return written


if __name__ == "__main__":
    main()
