"""Draw the simplex attribution plot of the same explanations in different ways, side by side.

``--vary fade`` (the default) leaves every dot where it is, at a distance from the center that is
proportional to its contrast magnitude, and varies how the size and the opacity of the dots and
superpixels follow the magnitude. ``--vary scale`` hides the features below the noise floor and
varies how the magnitude above the floor becomes a distance to the center.

The noise floor is measured where a run has two coalition samples (``03_explain.py --seed 1``),
and a conservative bound otherwise.
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
SCALES = {
    "linear, all features\n(as before)": {"noise_floor": 0.0},
    "linear\n(distance = magnitude above the floor)": {},
    "square root\n(weak features outwards)": {"scale": "power", "exponent": 0.5},
    "square\n(weak features to the center)": {"scale": "power", "exponent": 2.0},
    "softmax\n(strongest features at the boundary)": {"scale": "softmax"},
    "direction only\n(size = magnitude)": {"scale": "direction"},
}
# Title and options of the visibility; ``None`` is the look as before, "floor" the noise floor.
FADES: dict[str, dict[str, float | str] | None] = {
    "as before\n(every feature at least 35% opaque)": None,
    "linear\n(size and opacity = magnitude)": {},
    "quadratic\n(magnitude squared)": {"exponent": 2.0},
    "noise floor\n(invisible below, linear above)": {"noise_floor": "floor"},
    "noise floor, square root\n(invisible below, soon visible above)": {"noise_floor": "floor", "exponent": 0.5},
}


def main() -> None:
    """Draw one row per explanation and one column per scale."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS, help="result directories below results/")
    parser.add_argument("--readout", default="probability")
    parser.add_argument("--vary", choices=["fade", "scale"], default="fade")
    args = parser.parse_args()

    variants = FADES if args.vary == "fade" else SCALES
    with mpl.rc_context(plots.STYLE):
        fig = plt.figure(figsize=(3.9 * len(variants), 7.4 * len(args.runs)))
        grid = fig.add_gridspec(
            2 * len(args.runs), len(variants), height_ratios=[0.62, 1] * len(args.runs), wspace=0.62, hspace=0.55
        )
        for row, run in enumerate(args.runs):
            explanation = load_explanation(RESULT_DIR / run, args.readout)
            summary, values, floor = explanation.summary, explanation.values, explanation.noise_floor
            image = Image.open(ASSET_DIR / summary["image"]).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
            above = int(np.sum(simplex.magnitude(values) > floor))
            source = "measured, 2 samples" if explanation.n_samples > 1 else "bound, 1 sample"
            for column, (title, options) in enumerate(variants.items()):
                if args.vary == "fade":
                    fade = options and {k: floor if v == "floor" else v for k, v in options.items()}
                    style: dict = {"fade": fade}
                else:
                    style = {"embedding": {"noise_floor": floor, **options}}
                ax_map = fig.add_subplot(grid[2 * row, column])
                plots.simplex_map(ax_map, np.asarray(image) / 255, explanation.cells, values, **style)
                ax_map.set_title(title, fontsize=9)
                ax = fig.add_subplot(grid[2 * row + 1, column], projection="ternary")
                plots.ternary_plot(ax, values, summary["classes"], **style)
            fig.text(
                0.02, ax_map.get_position().y1 + 0.075 / len(args.runs),
                f"{' / '.join(summary['classes'])}: noise floor {floor:.2g} ({source}), "
                f"{above} of {len(values)} features above it",
                fontsize=10, fontweight="bold",
            )  # fmt: skip
        what = "Size and opacity" if args.vary == "fade" else "Scales"
        fig.suptitle(
            f"{what} of the simplex attribution plot, {args.readout} readout", x=0.02, y=0.925, ha="left",
            fontsize=11, color=plots.INK_SECONDARY,
        )  # fmt: skip
        path = RESULT_DIR / f"{args.vary}_gallery_{args.readout}.png"
        fig.savefig(path, dpi=130, bbox_inches="tight")
    print(path)


if __name__ == "__main__":
    main()
