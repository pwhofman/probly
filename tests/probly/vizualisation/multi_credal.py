# mypy: ignore-errors
from __future__ import annotations

from collections.abc import Sequence

from matplotlib.lines import Line2D
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
import matplotlib.pyplot as plt
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import numpy as np
from numpy.typing import NDArray


# ======================================================================
#   RADAR FACTORY
# ======================================================================
def radar_factory(num_vars: int, frame: str = "circle") -> NDArray[np.float64]:
    """Create a radar chart with ``num_vars`` axes.

    Parameters
    ----------
    num_vars:
        Number of variables (axes) for the radar chart.
    frame:
        ``"polygon"`` for a polygon frame, ``"circle"`` for a circular frame.
    """
    theta: NDArray[np.float64] = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):
        """Transform that keeps gridlines as straight segments."""

        def transform_path_non_affine(self, path: Path) -> Path:
            """Transform gridline paths so they interpolate as polygons."""
            # Avoid direct access to private attributes (_interpolation_steps).
            steps = getattr(path, "_interpolation_steps", 1)
            if steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):
        """Custom polar axes used for radar charts."""

        name = "radar"
        PolarTransform = RadarTransform

        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            # Place the first axis at the top (north).
            self.set_theta_zero_location("N")

        def fill(
            self,
            *args: object,
            closed: bool = True,
            **kwargs: object,
        ) -> object:
            """Override fill so that polygons are closed by default."""
            return super().fill(*args, closed=closed, **kwargs)

        def plot(
            self,
            *args: object,
            **kwargs: object,
        ) -> object:
            """Override plot so that lines are automatically closed."""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)
            return lines

        def _close_line(self, line: Line2D) -> None:
            """Ensure the line forms a closed polygon."""
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels: Sequence[str]) -> None:
            """Set the labels for each axis."""
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self) -> Circle | RegularPolygon:
            """Return the patch used as background/frame for the axes."""
            if frame == "circle":
                return Circle((0.5, 0.5), 0.5)
            if frame == "polygon":
                return RegularPolygon(
                    (0.5, 0.5),
                    num_vars,
                    radius=0.5,
                    edgecolor="k",
                )
            msg = f"Unknown frame type: {frame}"
            raise ValueError(msg)

        def _gen_axes_spines(self) -> dict[str, Spine]:
            """Generate the outer spine for the axes."""
            if frame == "circle":
                return super()._gen_axes_spines()
            if frame == "polygon":
                spine = Spine(self, "circle", Path.unit_regular_polygon(num_vars))
                spine.set_transform(
                    Affine2D().scale(0.5).translate(0.5, 0.5) + self.transAxes,
                )
                return {"polar": spine}
            msg = f"Unknown frame type: {frame}"
            raise ValueError(msg)

    register_projection(RadarAxes)
    return theta


# ======================================================================
#   GENERAL FUNCTION FOR ANY SPIDER PLOT
# ======================================================================
def spider_plot(
    labels: Sequence[str],
    datasets: Sequence[tuple[str, Sequence[float]]],
    frame: str = "polygon",
    title: str = "Spider Plot",
) -> None:
    """Create a general spider / radar plot.

    Parameters
    ----------
    labels:
        List of axis labels, e.g. ``["A", "B", "C", "D", "E"]``.
    datasets:
        List of ``(name, values)`` tuples.
        ``name`` is shown in the legend.
        ``values`` is a sequence of floats with ``len(values) == len(labels)``.
    frame:
        Either ``"polygon"`` or ``"circle"``.
    title:
        Title shown at the top of the plot.
    """
    n_axes = len(labels)
    theta = radar_factory(n_axes, frame=frame)

    _fig, ax = plt.subplots(
        figsize=(6, 6),
        subplot_kw={"projection": "radar"},
    )

    # Default radius grid (for probabilities 0-1).
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(0, 1)

    # Set axis labels.
    ax.set_varlabels(labels)

    # Automatic color cycling.
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Plot each dataset.
    for (name, values), color in zip(datasets, colors, strict=False):
        values_arr = np.asarray(values, dtype=float)

        if len(values_arr) != n_axes:
            msg = f"Dataset '{name}' has length {len(values_arr)}, expected {n_axes} values."
            raise ValueError(msg)

        # Maximum-likelihood class index and coordinates.
        idx = int(np.argmax(values_arr))
        angle = theta[idx]
        radius = float(values_arr[idx])

        # Draw the MLE point.
        ax.scatter([angle], [radius], s=80, color=color, label=name)

        # Draw polygon line + filled area.
        ax.plot(theta, values_arr, color=color, linewidth=2)
        ax.fill(theta, values_arr, alpha=0.2, color=color)

    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    plt.title(title, pad=20)
    plt.tight_layout()
    plt.show()


# ======================================================================
#   EXAMPLE USAGE
# ======================================================================
if __name__ == "__main__":
    labels_example = ["A", "B", "C", "D", "E"]  # any number of labels works!
    datasets_example = [
        ("Run 1", [0.1, 0.5, 0.2, 0.3, 0.7]),
        ("Run 2", [0.3, 0.6, 0.1, 0.4, 0.2]),
        ("Run 3", [0.8, 0.1, 0.05, 0.2, 0.3]),
        ("Run 4", [0.1, 0.1, 0.1, 0.1, 0.1]),
        ("Run 5", [0.2, 0.7, 0.1, 0.4, 0.1]),
    ]

    spider_plot(labels_example, datasets_example, title="General Credal/Spider Plot")
