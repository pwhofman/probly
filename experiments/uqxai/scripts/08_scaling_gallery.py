"""Draw the simplex attribution plot of the same explanations with different scales, side by side.

The first triangle is the plot as it was: every feature, distance proportional to the contrast
magnitude. The others hide the features below the noise floor and differ in how the magnitude above
the floor becomes a distance to the center. The noise floor is measured where a run has two
coalition samples (``03_explain.py --seed 1``), and a conservative bound otherwise.
"""

from __future__ import annotations

import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from uqxai import plots, simplex
from uqxai.model import IMAGE_SIZE
from uqxai.paths import ASSET_DIR, RESULT_DIR
from uqxai.results import load_explanation

DEFAULT_RUNS = [
    "electric/dog_guitar_patch16_mean",
    "labrador/dog_guitar_patch16_mean",
    "labrador/ambiguous_labrador_patch16_mean",
]
# Title and options of each triangle; the noise floor is filled in per explanation.
VARIANTS = {
    "linear, all features\n(as before)": {"noise_floor": 0.0},
    "linear\n(distance = magnitude above the floor)": {},
    "square root\n(weak features outwards)": {"scale": "power", "exponent": 0.5},
    "square\n(weak features to the center)": {"scale": "power", "exponent": 2.0},
    "softmax\n(strongest features at the boundary)": {"scale": "softmax"},
    "direction only\n(size = magnitude)": {"scale": "direction"},
}


def main() -> None:
    """Draw one row per explanation and one column per scale."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS, help="result directories below results/")
    parser.add_argument("--readout", default="probability")
    args = parser.parse_args()

    with mpl.rc_context(plots.STYLE):
        columns = len(VARIANTS) + 1
        fig = plt.figure(figsize=(3.9 * columns, 4.3 * len(args.runs)))
        grid = fig.add_gridspec(len(args.runs), columns, wspace=0.62, hspace=0.5)
        for row, run in enumerate(args.runs):
            explanation = load_explanation(RESULT_DIR / run, args.readout)
            summary, values, floor = explanation.summary, explanation.values, explanation.noise_floor
            image = Image.open(ASSET_DIR / summary["image"]).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
            above = int(np.sum(simplex.magnitude(values) > floor))
            source = "measured, 2 samples" if explanation.n_samples > 1 else "bound, 1 sample"

            ax_map = fig.add_subplot(grid[row, 0])
            plots.simplex_map(
                ax_map, np.asarray(image) / 255, explanation.cells, values, embedding={"noise_floor": floor}
            )
            ax_map.set_title(" /\n".join(summary["classes"]), fontsize=9)
            ax_map.text(
                0.5, -0.04, f"noise floor {floor:.2g} ({source})\n{above} of {len(values)} features above it",
                transform=ax_map.transAxes, ha="center", va="top", fontsize=8, color=plots.INK_SECONDARY,
            )  # fmt: skip
            for column, (title, options) in enumerate(VARIANTS.items(), start=1):
                ax = fig.add_subplot(grid[row, column], projection="ternary")
                plots.ternary_plot(ax, values, summary["classes"], embedding={"noise_floor": floor, **options})
                ax.set_title(title, fontsize=9, pad=22)
        handles, _ = ax.get_legend_handles_labels()
        fig.legend(
            handles[:1], ["features below the noise floor (one dot for all of them)"], loc="lower center",
            frameon=False, fontsize=9,
        )  # fmt: skip
        fig.suptitle(
            f"Scales of the simplex attribution plot, {args.readout} readout", x=0.02, y=0.94, ha="left", fontsize=11,
            color=plots.INK_SECONDARY,
        )  # fmt: skip
        path = RESULT_DIR / f"scaling_gallery_{args.readout}.png"
        fig.savefig(path, dpi=130, bbox_inches="tight")
    print(path)


if __name__ == "__main__":
    main()
