"""Draw the explanations of several problems side by side: one column per sub-ensemble."""

from __future__ import annotations

import argparse
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from uqxai import plots
from uqxai.classes import PROBLEMS
from uqxai.model import IMAGE_SIZE
from uqxai.paths import ASSET_DIR, RESULT_DIR


def main() -> None:
    """Draw the image maps and the simplex attribution plots of the chosen problems."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", nargs="+", choices=list(PROBLEMS), default=list(PROBLEMS))
    parser.add_argument("--run", default="dog_guitar_patch16_mean", help="result directory within each problem")
    parser.add_argument("--readout", default="probability")
    args = parser.parse_args()

    with mpl.rc_context(plots.STYLE):
        fig = plt.figure(figsize=(4.4 * len(args.problems), 8.2))
        grid = fig.add_gridspec(2, len(args.problems), height_ratios=[0.8, 1], hspace=0.22, wspace=0.55)
        for column, name in enumerate(args.problems):
            run_dir = RESULT_DIR / name / args.run
            summary = json.loads((run_dir / "summary.json").read_text())
            stored = np.load(run_dir / "attributions.npz")
            attributions = stored[f"{args.readout}_values"]
            image = np.asarray(Image.open(ASSET_DIR / summary["image"]).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)))
            is_probability = args.readout in ("probability", "open_world")
            full, empty = stored[f"{args.readout}_full"], stored[f"{args.readout}_empty"]

            ax_map = fig.add_subplot(grid[0, column])
            plots.simplex_map(ax_map, image / 255, stored["cells"], attributions)
            prediction = ", ".join(f"{value:.2f}" for value in summary["prediction"].values())
            ax_map.set_title(f"{' / '.join(summary['classes'])}\nprediction {prediction}", fontsize=9)

            ax_ternary = fig.add_subplot(grid[1, column], projection="ternary")
            plots.ternary_plot(
                ax_ternary,
                attributions,
                summary["classes"],
                prediction=full / full.sum() if is_probability else None,
                baseline=empty / empty.sum() if is_probability else None,
                members=stored["member_probabilities"] if args.readout == "probability" else None,
            )
        handles, labels = ax_ternary.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", ncols=len(handles), frameon=False, fontsize=8)
        fig.suptitle(
            f"Simplex attribution plots of the {args.readout} readout, one sub-ensemble per column",
            x=0.02, ha="left", fontsize=10, color=plots.INK_SECONDARY,
        )  # fmt: skip
        path = RESULT_DIR / f"comparison_{args.run}_{args.readout}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
    print(path)


if __name__ == "__main__":
    main()
