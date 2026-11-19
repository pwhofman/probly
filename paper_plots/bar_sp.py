"""Render the SP ranked bar plot for every target from cached runs.

Reads only the local cache. Run ``python -m paper_plots.fetch_sp`` first to
populate / refresh it. Then::

    uv run python -m paper_plots.bar_sp
"""

from __future__ import annotations

import warnings

from paper_plots.aggregate import sp_ranking
from paper_plots.metadata import dataset_display
from paper_plots.paths import PAPER_ARTIFACTS
from paper_plots.style import draw_bars
from paper_plots.targets import TARGETS, Target

# Smaller figure so two bar plots sit comfortably side by side.
FIGSIZE: tuple[float, float] = (7.5, 3.6)


def _render_one(target: Target) -> None:
    ranking = sp_ranking(target)
    if not ranking:
        warnings.warn(
            f"no cached SP rankings for {target.dataset}/{target.base_model}; run fetch_sp first",
            stacklevel=2,
        )
        return
    base_entry = next((r for r in ranking if r["method"] == "base"), None)
    bars = [r for r in ranking if r["method"] != "base"]
    fig = draw_bars(
        bars,
        ylabel="Area under acc.-rejection curve",
        title=f"Selective prediction on {dataset_display(target.dataset)}",
        figsize=FIGSIZE,
        baseline=base_entry,
    )
    out_dir = PAPER_ARTIFACTS / f"{target.dataset}_methods"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"bar_sp_{target.dataset}.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


def main() -> None:
    for target in TARGETS:
        _render_one(target)


if __name__ == "__main__":
    main()
