"""Draw the figures of an explanation computed by ``03_explain.py``."""

from __future__ import annotations

import argparse
import json
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from uqxai import plots, simplex
from uqxai.model import IMAGE_SIZE
from uqxai.paths import ASSET_DIR, RESULT_DIR
from uqxai.readouts import UNCERTAINTY_NAMES

if TYPE_CHECKING:
    from pathlib import Path

READOUT_TITLES = {
    "probability": "class probabilities of the sub-ensemble",
    "mean_logit": "class logits of the sub-ensemble",
    "open_world": "probabilities of the same classes in the 1000-class ImageNet model",
}


def simplex_figure(
    path: Path,
    image: np.ndarray,
    stored: dict[str, np.ndarray],
    readout: str,
    class_names: list[str],
    background: float,
) -> dict[str, object]:
    """Draw the image map, the simplex attribution plot and the regime plane of one readout."""
    raw = stored[f"{readout}_values"]
    # Exact Shapley values of class probabilities sum to zero over the classes: the common part of
    # the estimate is error and is removed. Its size is reported as an estimate of that error.
    attributions = simplex.contrast(raw) if readout == "probability" else raw
    threshold = background * simplex.magnitude(attributions).max()
    is_probability = readout in ("probability", "open_world")
    fig = plt.figure(figsize=(13.5, 4.6))
    grid = fig.add_gridspec(1, 3, width_ratios=[0.85, 1.05, 1.25], wspace=0.4)

    ax_map = fig.add_subplot(grid[0])
    plots.simplex_map(ax_map, image, stored["cells"], attributions)
    ax_map.set_title("Superpixels, colored like their dots")

    ax_ternary = fig.add_subplot(grid[1], projection="ternary")
    plots.ternary_plot(
        ax_ternary,
        attributions,
        class_names,
        prediction=stored[f"{readout}_full"] / stored[f"{readout}_full"].sum() if is_probability else None,
        baseline=stored[f"{readout}_empty"] / stored[f"{readout}_empty"].sum() if is_probability else None,
        members=stored["member_probabilities"] if readout == "probability" else None,
        prediction_note=" (renormalized to the three classes)" if readout == "open_world" else "",
    )
    ax_ternary.set_title("Simplex attribution plot", pad=28)
    handles, labels = ax_ternary.get_legend_handles_labels()
    if handles:
        ax_ternary.legend(
            handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False, fontsize=7.5, ncols=1
        )

    ax_plane = fig.add_subplot(grid[2])
    plots.regime_plane(ax_plane, attributions, threshold)
    ax_plane.set_title("Magnitude: common vs. contrast")

    fig.suptitle(f"Explained: {READOUT_TITLES[readout]}", x=0.01, ha="left", fontsize=10, color=plots.INK_SECONDARY)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    regime = simplex.regimes(attributions, threshold)
    return {
        "background_threshold": threshold,
        "regime_counts": dict(zip(simplex.REGIME_NAMES, np.bincount(regime, minlength=3).tolist(), strict=True)),
        "share_of_squared_magnitude_that_is_common": float(
            (simplex.common_magnitude(raw) ** 2).sum() / (simplex.magnitude(raw) ** 2).sum()
        ),
        "mean_supported_classes_of_non_background": float(
            simplex.supported_classes(attributions)[regime != simplex.BACKGROUND].mean()
        ),
    }


def maps_figure(
    path: Path, image: np.ndarray, cells: np.ndarray, values: np.ndarray, names: list[str], title: str
) -> None:
    """Draw the image and one signed attribution map per output, on a shared color scale."""
    limit = np.abs(values).max()
    fig, axes = plt.subplots(1, len(names) + 1, figsize=(3.1 * (len(names) + 1) + 0.6, 3.5))
    axes[0].imshow(image)
    axes[0].set_axis_off()
    axes[0].set_title("Image")
    for ax, name, column in zip(axes[1:], names, values.T, strict=True):
        plots.signed_map(ax, image, cells, column, limit=limit)
        ax.set_title(name)
    colorbar = fig.colorbar(
        mpl.cm.ScalarMappable(mpl.colors.Normalize(-limit, limit), plots.DIVERGING), ax=axes, fraction=0.02, pad=0.02
    )
    colorbar.outline.set_visible(False)
    colorbar.set_label("Shapley value")
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, color=plots.INK_SECONDARY)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Draw all figures of one result directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="result directory below results/, e.g. electric/dog_guitar_patch16_mean")
    parser.add_argument(
        "--background", type=float, default=0.1, help="background: magnitude below this share of the largest"
    )
    args = parser.parse_args()

    run_dir = RESULT_DIR / args.run
    summary = json.loads((run_dir / "summary.json").read_text())
    stored = dict(np.load(run_dir / "attributions.npz"))
    class_names = summary["classes"]
    image = np.asarray(Image.open(ASSET_DIR / summary["image"]).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))) / 255

    regimes = {}
    with mpl.rc_context(plots.STYLE):
        for readout in READOUT_TITLES:
            if f"{readout}_values" in stored:
                regimes[readout] = simplex_figure(
                    run_dir / f"simplex_{readout}.png", image, stored, readout, class_names, args.background
                )
        maps_figure(
            run_dir / "class_maps.png", image, stored["cells"],
            simplex.contrast(stored["probability_values"]), class_names,
            "Shapley values of the class probabilities",
        )  # fmt: skip
        if "uncertainty_values" in stored:
            maps_figure(
                run_dir / "uncertainty_maps.png", image, stored["cells"], stored["uncertainty_values"],
                [f"{name} uncertainty" for name in UNCERTAINTY_NAMES],
                "Shapley values of the uncertainty (entropy, nats)",
            )  # fmt: skip
    (run_dir / "regimes.json").write_text(json.dumps(regimes, indent=2) + "\n")
    print(json.dumps(regimes, indent=2))


if __name__ == "__main__":
    main()
