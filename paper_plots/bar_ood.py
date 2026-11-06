"""Render the near/far OOD ranked bar plots for every target from cached runs.

Reads only the local cache. Run ``python -m paper_plots.fetch_ood`` first to
populate / refresh it. Then::

    uv run python -m paper_plots.bar_ood
"""

from __future__ import annotations

import warnings

from paper_plots import ood_groups
from paper_plots.aggregate import ood_band_ranking
from paper_plots.metadata import dataset_display
from paper_plots.paths import PAPER_ARTIFACTS
from paper_plots.style import draw_bars
from paper_plots.targets import TARGETS, Target

# Smaller figure so two bar plots sit comfortably side by side.
FIGSIZE: tuple[float, float] = (7.5, 3.6)

# Per-dataset method exclusion (OoD plots only). Methods listed here are
# dropped from the OoD bar before rendering for the given dataset; they still
# appear in SP / AL / tables.
PER_DATASET_EXCLUDED_OOD: dict[str, set[str]] = {
    "imagenet": {"dare"},
}


def _render_one(target: Target) -> None:
    out_dir = PAPER_ARTIFACTS / f"{target.dataset}_methods"
    out_dir.mkdir(parents=True, exist_ok=True)
    band_lookup = {
        "near": ood_groups.near_ood_for(target.dataset),
        "far": ood_groups.far_ood_for(target.dataset),
    }
    excluded = ood_groups.EXCLUDED_METHODS | PER_DATASET_EXCLUDED_OOD.get(target.dataset, set())
    for band_name, band_datasets in band_lookup.items():
        ranking = ood_band_ranking(target, band_name)
        if not ranking:
            warnings.warn(
                f"no {band_name}-OOD AUROCs cached for {target.dataset}",
                stacklevel=2,
            )
            continue
        base_entry = next((r for r in ranking if r["method"] == "base"), None)
        bars = [r for r in ranking if r["method"] != "base" and r["method"] not in excluded]
        title_band = "Near-OoD" if band_name == "near" else "Far-OoD"
        fig = draw_bars(
            bars,
            ylabel="OoD AUROC",
            title=f"{title_band} detection on {dataset_display(target.dataset)}",
            subtitle=(
                f"(avg over {len(band_datasets)} datasets)"
                if len(band_datasets) > 1
                else f"({dataset_display(band_datasets[0])})"
            ),
            figsize=FIGSIZE,
            baseline=base_entry,
        )
        out = out_dir / f"bar_ood_{band_name}_{target.dataset}.pdf"
        fig.savefig(out)
        print(f"wrote {out}")


def main() -> None:
    for target in TARGETS:
        _render_one(target)


if __name__ == "__main__":
    main()
