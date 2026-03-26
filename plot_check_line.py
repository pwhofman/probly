"""Quick sanity-check script for outlined line plotting."""

from __future__ import annotations

import logging
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from probly.plot.utils import _plot_line_with_outline

logger = logging.getLogger(__name__)

OUTPUT_PATH = pathlib.Path(__file__).with_suffix(".png")


def main() -> None:
    """Generate a small plot that shows overlapping outlined lines."""
    x = np.linspace(0.0, 1.0, 200)
    y1 = x
    y2 = x + 0.02
    y3 = x - 0.02

    fig, ax = plt.subplots(figsize=(6, 4))
    _plot_line_with_outline(ax, x, y1, color="tab:blue", linewidth=2, label="line 1")
    _plot_line_with_outline(ax, x, y2, color="tab:orange", linewidth=2, label="line 2")
    _plot_line_with_outline(ax, x, y3, color="tab:green", linewidth=2, label="line 3")

    ax.set_title("Outlined line visibility check")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150)
    logger.info("Wrote %s", OUTPUT_PATH)
    plt.show()


if __name__ == "__main__":
    main()
