"""Plotting for >3 classes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from matplotlib.patches import Circle, Polygon, RegularPolygon
from matplotlib.path import Path
import matplotlib.patheffects as PathEffects
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
import matplotlib.pyplot as plt
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import numpy as np

import probly.visualization.config as cfg

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from numpy.typing import NDArray


def radar_factory(num_vars: int, frame: str = "circle") -> np.ndarray:  # noqa: C901
    """Create a radar chart with `num_vars` axes.

    Args:
        num_vars: Number of variables (axes) in the radar chart.
        frame: Shape of the frame, either ``"circle"`` or ``"polygon"``.

    Returns:
        Array of angles (theta) corresponding to each variable axis.
    """
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path: Path) -> Path:
            interpolation_steps = getattr(path, "_interpolation_steps", 1)
            if interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):
        name = "radar"
        PolarTransform = RadarTransform

        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)  # type: ignore[arg-type]
            self.set_theta_zero_location("N")

        def fill(self, *args: object, **kwargs: object) -> list[Polygon]:
            """Override fill to handle closed polygons by default.

            Returns:
                List of Polygon objects created by the fill operation.
            """
            closed = kwargs.pop("closed", True)
            return super().fill(*args, closed=closed, **kwargs)

        def plot(self, *args: Any, **kwargs: Any) -> list[Line2D]:  # noqa: ANN401
            """Plot lines on the radar chart and automatically close them.

            Returns:
                List of Line2D objects added to the axes.
            """
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)
            return lines

        def _close_line(self, line: Line2D) -> None:
            x_raw, y_raw = line.get_data()

            x: NDArray[np.floating[Any]] = np.asarray(x_raw, dtype=float)
            y: NDArray[np.floating[Any]] = np.asarray(y_raw, dtype=float)

            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels: list[str]) -> None:
            """Set labels for each variable axis.

            Args:
                labels: List of axis labels in angular order.
            """
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self) -> Patch:
            if frame == "circle":
                return Circle((0.5, 0.5), 0.5)
            if frame == "polygon":
                return RegularPolygon((0.5, 0.5), num_vars, radius=0.5, edgecolor=cfg.BLACK)
            msg = f"Unknown value for 'frame': {frame}"
            raise ValueError(msg)

        def _gen_axes_spines(self) -> dict[str, Spine]:
            if frame == "circle":
                return cast("dict[str, Spine]", super()._gen_axes_spines())
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

    def spider_plot(  # noqa: PLR0915
        self,
        probs: np.ndarray,
        labels: list[str],
        title: str,
        mle_flag: bool,
        credal_flag: bool,
        ax: Axes | None = None,
    ) -> Axes:
        """General radar (spider) plot for credal predictions.

        Args:
            probs: NumPy array of shape (N, C) with class probabilities per sample,
                where N is the number of samples and C is the number of classes.
            labels: Labels for the classes. Must have length C.
            title: Title of the plot.
            mle_flag: If True, a point for the mean probability vector (MLE-style summary)
                is shown.
            credal_flag: If True, the credal band (min-max envelope across samples) is shown.
            ax: Optional Matplotlib Axes to draw into. If provided, it should be a radar
                projection Axes. If None, a new figure and radar Axes are created.

        Returns:
            The Matplotlib Axes containing the spider plot.
        """
        n_classes = int(probs.shape[-1])
        theta = radar_factory(n_classes, frame="polygon")

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "radar"})

        ax_any = cast("Any", ax)

        ax_any.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0])
        ax_any.set_ylim(0.0, 1.0)
        ax_any.set_varlabels(labels)

        ref_theta = 0.5 * (theta[0] + theta[1])
        ax_any.set_rlabel_position(np.degrees(ref_theta))
        ax_any.set_yticklabels([])

        def spiderplot_axis_with_ticks(
            ax_in: Axes,
            theta_in: np.ndarray,
            n_vars: int,
            tick_values: list[float] | None = None,
            draw_tick_marks: bool = True,
        ) -> None:
            """Draw reference axis between class 1 and 2 and place 0..1 labels scaled to polygon boundary.

            Args:
                ax_in: Matplotlib Axes on which to draw the reference axis.
                theta_in: Array of angular positions for each class.
                n_vars: Number of variables (classes).
                tick_values: Optional list of tick values in the interval [0, 1].
                draw_tick_marks: Whether to draw small tick marks along the axis.
            """
            # For a regular n-gon frame, the radius to the middle of an edge is cos(pi/n)
            ref_theta2 = 0.5 * (theta_in[0] + theta_in[1])
            r_max = float(np.cos(np.pi / n_vars))

            if tick_values is None:
                tick_values = [0.2, 0.4, 0.6, 0.8, 1.0]

            axis_color = cfg.BLACK

            (axis_line,) = ax_in.plot(
                [ref_theta2, ref_theta2],
                [0.0, r_max],
                color=axis_color,
                alpha=0.8,
                linewidth=0.5,
                zorder=2,
            )
            axis_line.set_clip_path(ax_in.patch)

            for t in tick_values:
                if t < 0 or t > 1:
                    continue
                r = t * r_max

                if draw_tick_marks and t not in (0.0, 1.0):
                    dtheta = 0.015
                    (tm,) = ax_in.plot(
                        [ref_theta2 - dtheta, ref_theta2 + dtheta],
                        [r, r],
                        color=axis_color,
                        linewidth=1.5,
                        zorder=5,
                    )
                    tm.set_clip_path(ax_in.patch)

                txt = ax_in.text(
                    ref_theta2,
                    r,
                    f"{t:g}",
                    color=cfg.WHITE,
                    fontsize=7,
                    ha="center",
                    va="center",
                    zorder=7,
                )
                txt.set_clip_path(ax_in.patch)
                txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground=cfg.BLACK)])

        max_class = np.argmax(probs, axis=1)
        max_probs = np.max(probs, axis=1)

        ax_any.scatter(theta[max_class], max_probs, color=cfg.BLUE, label="Probabilities", zorder=3)

        if mle_flag:
            mean_probs = probs.mean(axis=0)
            mean_max_class = int(np.argmax(mean_probs))
            ax_any.scatter(
                theta[mean_max_class],
                mean_probs[mean_max_class],
                s=80,
                color=cfg.RED,
                label="MLE",
                zorder=4,
            )

        if credal_flag:
            lower = probs.min(axis=0)
            upper = probs.max(axis=0)
            lower_c = np.append(lower, lower[0])
            upper_c = np.append(upper, upper[0])
            theta_c = np.append(theta, theta[0])

            ax_any.fill_between(
                theta_c,
                lower_c,
                upper_c,
                alpha=0.30,
                label="Credal band (lower-upper)",
                zorder=2,
            )

            ax_any.plot(
                theta_c,
                lower_c,
                linestyle=cfg.MIN_MAX_LINESTYLE_1,
                color=cfg.RED,
                linewidth=1.5,
                label="Lower bound",
            )
            ax_any.plot(
                theta_c,
                upper_c,
                linestyle=cfg.MIN_MAX_LINESTYLE_2,
                color=cfg.BLUE,
                linewidth=1.5,
                label="Upper bound",
            )

        spiderplot_axis_with_ticks(ax, theta, n_vars=n_classes, draw_tick_marks=True)
        ax.set_title(title, pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()

        return ax
