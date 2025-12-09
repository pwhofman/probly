"""Plotting for >3 classes."""

from __future__ import annotations

from typing import Any

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
import matplotlib.pyplot as plt
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import numpy as np


def radar_factory(num_vars: int, frame: str = "circle") -> np.ndarray:  # noqa: C901
    """Create a radar chart with `num_vars` axes."""
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path: Path) -> Path:
            # Note: _interpolation_steps is internal logic needed for this projection hack
            if path._interpolation_steps > 1:  # noqa: SLF001
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):
        name = "radar"
        PolarTransform = RadarTransform

        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location("N")

        def fill(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            """Override fill to handle closed polygons by default."""
            closed = kwargs.pop("closed", True)
            return super().fill(closed=closed, *args, **kwargs)  # noqa: B026

        def plot(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)
            return lines

        def _close_line(self, line: Any) -> None:  # noqa: ANN401
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels: list[str]) -> None:
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self) -> Any:  # noqa: ANN401
            if frame == "circle":
                return Circle((0.5, 0.5), 0.5)
            if frame == "polygon":
                return RegularPolygon((0.5, 0.5), num_vars, radius=0.5, edgecolor="k")
            msg = f"Unknown value for 'frame': {frame}"
            raise ValueError(msg)

        def _gen_axes_spines(self) -> Any:  # noqa: ANN401
            if frame == "circle":
                return super()._gen_axes_spines()
            if frame == "polygon":
                spine = Spine(axes=self, spine_type="circle", path=Path.unit_regular_polygon(num_vars))
                spine.set_transform(Affine2D().scale(0.5).translate(0.5, 0.5) + self.transAxes)
                return {"polar": spine}
            msg = f"Unknown value for 'frame': {frame}"
            raise ValueError(msg)

    register_projection(RadarAxes)
    return theta


class MultiVisualizer:
    """Class to create multidimensional plots."""

    def spider_plot(
        self,
        probs: np.ndarray,
        labels: list[str] | None = None,
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        """General radar (spider) plot for credal predictions.

        Args:
        probs: NumPy array with probabilities.
        labels: labels for the classes.
        ax: Axes on which to create the radar chart.
        """
        n_classes = probs.shape[-1]

        if labels is None:
            labels = [f"C{i + 1}" for i in range(n_classes)]

        if len(labels) != n_classes:
            msg = f"Number of labels ({len(labels)}) must match number of classes ({n_classes})."
            raise ValueError(msg)

        # Use the factory from spider.py
        theta = radar_factory(n_classes, frame="polygon")

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "radar"})

        # Setup Axis
        ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_ylim(0.0, 1.0)
        ax.set_varlabels(labels)

        probs_flat = probs.flatten()
        max_class = np.argmax(probs_flat)
        ax.scatter(theta[max_class], probs_flat[max_class], s=80, color="red", label="MLE")

        ax.set_title(f"Spider Plot ({n_classes} Classes)", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()

        return ax
