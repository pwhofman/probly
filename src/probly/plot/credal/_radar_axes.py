"""Custom RadarAxes projection for spider/radar charts."""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Any

from matplotlib.patches import RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import numpy as np

if TYPE_CHECKING:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from numpy.typing import NDArray


@cache
def _get_radar_axes(num_vars: int) -> type[PolarAxes]:
    """Create and cache a RadarAxes subclass for the given number of variables.

    Each call with the same ``num_vars`` returns the same class. The class is
    registered with matplotlib as ``"radar_{num_vars}"``.

    Args:
        num_vars: Number of spokes in the radar chart.

    Returns:
        A RadarAxes class registered with matplotlib.
    """
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):  # ty: ignore[unsupported-base]
        def transform_path_non_affine(self, path: Path) -> Path:
            if path._interpolation_steps > 1:  # ty: ignore[unresolved-attribute]  # noqa: SLF001
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):
        name = f"radar_{num_vars}"
        PolarTransform = RadarTransform

        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)  # ty: ignore[invalid-argument-type]
            self.set_theta_zero_location("N")

        def fill(self, *args: object, **kwargs: Any) -> list[Any]:  # noqa: ANN401
            """Override fill so that the polygon is closed by default."""
            closed = kwargs.pop("closed", True)
            return super().fill(*args, closed=closed, **kwargs)

        def plot(self, *args: Any, **kwargs: Any) -> list[Line2D]:  # noqa: ANN401
            """Override plot so that lines are closed by default."""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)
            return lines

        def _close_line(self, line: Line2D) -> None:
            x_raw, y_raw = line.get_data()
            x: NDArray[np.floating[Any]] = np.asarray(x_raw, dtype=float)
            y: NDArray[np.floating[Any]] = np.asarray(y_raw, dtype=float)
            if len(x) > 0 and x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels: list[str]) -> None:
            """Set labels for each spoke.

            Args:
                labels: One label per spoke, in angular order.
            """
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self) -> Patch:
            return RegularPolygon(
                (0.5, 0.5),
                num_vars,
                radius=0.5,
                edgecolor="k",
            )

        def _gen_axes_spines(self) -> dict[str, Spine]:
            spine = Spine(
                axes=self,
                spine_type="circle",
                path=Path.unit_regular_polygon(num_vars),
            )
            spine.set_transform(
                Affine2D().scale(0.5).translate(0.5, 0.5) + self.transAxes,
            )
            return {"polar": spine}

    register_projection(RadarAxes)
    return RadarAxes
