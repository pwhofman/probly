"""Quick sanity-check script for the plotting module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from probly.plot.credal_plot import plot_credal_set
from probly.representation.credal_set.array import ArrayProbabilityIntervalsCredalSet

if TYPE_CHECKING:
    from probly.plot._base import PlotFunction

if __name__ == "__main__":
    credal_set = ArrayProbabilityIntervalsCredalSet(
        lower_bounds=np.array([0.1, 0.2, 0.3]), upper_bounds=np.array([0.4, 0.5, 0.6])
    )
    print(credal_set)  # noqa: T201

    plot_credal_set(credal_set)
    plt.show()

    a: PlotFunction[ArrayProbabilityIntervalsCredalSet] = plot_credal_set
