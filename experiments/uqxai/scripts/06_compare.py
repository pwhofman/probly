"""Draw explanations side by side: one column per explanation.

Without ``--cases``, the columns are the problems for one image. With ``--cases``, there is one
figure per problem of ``assets/cases.json`` whose columns are its uncertainty cases, and
``results/cases.json`` summarizes where the features of each explanation point.
"""

from __future__ import annotations

import argparse
import json
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from uqxai import plots, simplex
from uqxai.classes import PROBLEMS
from uqxai.model import IMAGE_SIZE
from uqxai.paths import ASSET_DIR, CASES_MANIFEST, RESULT_DIR, run_dir

if TYPE_CHECKING:
    from pathlib import Path

CASE_TITLES = {
    "certain": "Certain",
    "conflict": "Uncertain: regions disagree",
    "ambiguous": "Uncertain: one object, two classes",
}


def draw(path: Path, columns: list[tuple[str, Path]], readout: str, title: str) -> None:
    """Draw the image, the image map and the simplex attribution plot of each explanation."""
    with mpl.rc_context(plots.STYLE):
        fig = plt.figure(figsize=(4.4 * len(columns), 11.6))
        grid = fig.add_gridspec(3, len(columns), height_ratios=[0.8, 0.8, 1], hspace=0.16, wspace=0.55)
        for column, (heading, directory) in enumerate(columns):
            summary = json.loads((directory / "summary.json").read_text())
            stored = np.load(directory / "attributions.npz")
            attributions = stored[f"{readout}_values"]
            if readout == "probability":  # the common part of probabilities is estimation error
                attributions = simplex.contrast(attributions)
            image = Image.open(ASSET_DIR / summary["image"]).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
            image = np.asarray(image) / 255
            is_probability = readout in ("probability", "open_world")
            full, empty = stored[f"{readout}_full"], stored[f"{readout}_empty"]

            ax_image = fig.add_subplot(grid[0, column])
            ax_image.imshow(image)
            ax_image.set_axis_off()
            prediction = " / ".join(f"{value:.2f}" for value in summary["prediction"].values())
            ax_image.set_title(
                f"{heading}\nprediction {prediction}\nuncertainty {summary['total_uncertainty']:.2f} nats", fontsize=9
            )
            plots.simplex_map(fig.add_subplot(grid[1, column]), image, stored["cells"], attributions)
            plots.ternary_plot(
                fig.add_subplot(grid[2, column], projection="ternary"),
                attributions,
                summary["classes"],
                prediction=full / full.sum() if is_probability else None,
                baseline=empty / empty.sum() if is_probability else None,
                members=stored["member_probabilities"] if readout == "probability" else None,
            )
        handles, labels = fig.axes[-1].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", ncols=len(handles), frameon=False, fontsize=8)
        fig.suptitle(title, x=0.02, y=0.96, ha="left", fontsize=10, color=plots.INK_SECONDARY)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(path)


def case_summary(directory: Path, background: float) -> dict[str, object]:
    """Summarize the prediction and the structure of one explanation."""
    summary = json.loads((directory / "summary.json").read_text())
    stored = np.load(directory / "attributions.npz")
    result: dict[str, object] = {
        "classes": summary["classes"],
        "prediction": list(summary["prediction"].values()),
        "prediction_without_features": stored["probability_empty"].tolist(),
        "total_uncertainty": summary["total_uncertainty"],
        "epistemic_uncertainty": summary["epistemic_uncertainty"],
    }
    for readout in ("probability", "mean_logit"):
        raw = stored[f"{readout}_values"]
        attributions = simplex.contrast(raw) if readout == "probability" else raw
        threshold = background * simplex.magnitude(attributions).max()
        result[readout] = simplex.structure(attributions, threshold)
        result[readout]["common_share"] = float(
            (simplex.common_magnitude(raw) ** 2).sum() / (simplex.magnitude(raw) ** 2).sum()
        )
    return result


def main() -> None:
    """Draw the comparison of the problems, or of the uncertainty cases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", nargs="+", choices=list(PROBLEMS), default=["electric", "banjo", "hat"])
    parser.add_argument("--run", default="dog_guitar_patch16_mean", help="result directory within each problem")
    parser.add_argument("--readout", default="probability")
    parser.add_argument("--cases", action="store_true", help="compare the uncertainty cases of assets/cases.json")
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--baseline", default="mean")
    parser.add_argument("--background", type=float, default=0.1, help="background: below this share of the largest")
    args = parser.parse_args()

    if not args.cases:
        columns = [(" / ".join(PROBLEMS[name].class_names), RESULT_DIR / name / args.run) for name in args.problems]
        title = f"Simplex attribution plots of the {args.readout} readout, one sub-ensemble per column"
        draw(RESULT_DIR / f"comparison_{args.run}_{args.readout}.png", columns, args.readout, title)
        return

    cases = json.loads(CASES_MANIFEST.read_text())
    report: dict[str, dict[str, object]] = {}
    for problem in dict.fromkeys(case["problem"] for case in cases):
        columns = []
        for name, heading in CASE_TITLES.items():
            case = next(c for c in cases if c["problem"] == problem and c["case"] == name)
            directory = run_dir(problem, ASSET_DIR / case["image"], args.patch_size, args.baseline)
            columns.append((heading, directory))
            report.setdefault(problem, {})[name] = case_summary(directory, args.background)
        classes = " / ".join(PROBLEMS[problem].class_names)
        title = f"Three cases of uncertainty, {args.readout} readout of the model {classes}"
        draw(RESULT_DIR / f"cases_{problem}_{args.readout}.png", columns, args.readout, title)
    (RESULT_DIR / "cases.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
