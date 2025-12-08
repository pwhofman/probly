"""Geometry-related visualization utilities for credal sets."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

def radar_factory(num_vars: int, frame: str = "polygon") -> np.ndarray:
    """
    Create radar chart angles and register a custom radar projection.
    """

    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):
        name = "radar"
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location("N")

        def fill(self, *args, closed=True, **kwargs):
            return super().fill(*args, closed=closed, **kwargs)

        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)
            return lines

        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                line.set_data(np.append(x, x[0]), np.append(y, y[0]))

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            if frame == "polygon":
                return RegularPolygon((0.5, 0.5), num_vars, radius=0.5)
            return Circle((0.5, 0.5), 0.5)

        def _gen_axes_spines(self):
            if frame == "polygon":
                spine = Spine(
                    axes=self,
                    spine_type="circle",
                    path=Path.unit_regular_polygon(num_vars),
                )
                spine.set_transform(
                    Affine2D().scale(0.5).translate(0.5, 0.5) + self.transAxes
                )
                return {"polar": spine}
            return super()._gen_axes_spines()

    register_projection(RadarAxes)

    return np.append(theta, theta[0])

class CredalVisualizer:
    """Collection of geometric plots for credal predictions."""

    def __init__(self) -> None:
        pass

def spider_plot(
    self,
    lower: np.ndarray | None,
    upper: np.ndarray | None,
    mle: np.ndarray | None,
    labels: list[str],
    title: str = "Credal Prediction",
    rmax: float = 1.0,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    General radar (spider) plot for credal predictions.
    """

    K = len(labels)

    if mle is None and (lower is None or upper is None):
        raise ValueError(
            "Either 'mle' or both 'lower' and 'upper' must be provided."
        )

    theta = radar_factory(K)

    if ax is None:
        _, ax = plt.subplots(
            figsize=(6, 6),
            subplot_kw=dict(projection="radar"),
        )

    ax.set_rgrids(np.linspace(0, rmax, 6)[1:])
    ax.set_ylim(0.0, rmax)
    ax.set_varlabels(labels)

    if lower is not None and upper is not None:
        lower_c = np.append(lower, lower[0])
        upper_c = np.append(upper, upper[0])
        ax.fill(theta, upper_c, alpha=0.25, label="Credal set")
        ax.fill(theta, lower_c, color="white")

    if mle is not None:
        idx = np.argmax(mle)
        ax.scatter(
            [theta[idx]],
            [mle[idx]],
            s=80,
            color="red",
            label="MLE",
        )

    ax.set_title(title, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))

    return ax
        