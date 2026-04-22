"""Spider (radar) plot drawing for credal sets with 4+ classes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from flextype import flexdispatch
from probly.representation.credal_set.array import (
    ArrayCategoricalCredalSet,
    ArrayConvexCredalSet,
    ArrayDiscreteCredalSet,
    ArrayDistanceBasedCredalSet,
    ArrayProbabilityIntervalsCredalSet,
    ArraySingletonCredalSet,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from probly.plot.config import PlotConfig


def _setup_spider_axes(
    ax: Axes,
    labels: list[str],
    config: PlotConfig,
    *,
    gridlines: bool = True,
) -> None:
    """Configure a RadarAxes for credal set plotting.

    Args:
        ax: A RadarAxes instance.
        labels: One label per spoke.
        config: Plot configuration.
        gridlines: Whether to show concentric gridlines.
    """
    ax_radar = cast("Any", ax)
    ax_radar.set_ylim(0.0, 1.0)
    ax_radar.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_radar.set_yticklabels([])
    ax_radar.set_varlabels(labels)

    if gridlines:
        ax.yaxis.grid(True, color=config.color_gridline, alpha=config.grid_alpha)
    else:
        ax.yaxis.grid(False)

    ax.spines["polar"].set_color(config.color_gridline)
    ax.spines["polar"].set_linewidth(1)


def _get_theta(num_classes: int) -> np.ndarray:
    """Return evenly spaced spoke angles for *num_classes* variables."""
    return np.linspace(0, 2 * np.pi, num_classes, endpoint=False)


def _draw_spider_bar(
    ax: Axes,
    angle: float,
    r_low: float,
    r_high: float,
    width: float,
    color: object,
    alpha: float,
    *,
    edgecolor: object | None = None,
    linewidth: float = 0.5,
) -> None:
    """Draw a constant-width rectangular bar along a spoke.

    The bar has uniform visual thickness regardless of its radial position.
    It is computed in Cartesian space and converted back to polar.

    Args:
        ax: The RadarAxes to draw on.
        angle: Spoke angle in radians.
        r_low: Inner radius of the bar.
        r_high: Outer radius of the bar.
        width: Visual width of the bar in data units.
        color: Fill color.
        alpha: Fill transparency.
        edgecolor: Edge color (defaults to *color*).
        linewidth: Edge line width.
    """
    c, s = np.cos(angle), np.sin(angle)
    p0 = np.array([r_low * c, r_low * s])
    p1 = np.array([r_high * c, r_high * s])

    # Unit tangent perpendicular to the spoke
    u_tan = np.array([-s, c])
    offset = (width / 2.0) * u_tan

    # Rectangle corners in Cartesian
    q0a = p0 - offset
    q0b = p0 + offset
    q1a = p1 + offset
    q1b = p1 - offset

    corners = [q0a, q0b, q1a, q1b]
    ths = [float(np.arctan2(q[1], q[0])) for q in corners]
    rs = [float(np.hypot(q[0], q[1])) for q in corners]

    ax.fill(
        ths,
        rs,
        facecolor=color,
        alpha=alpha,
        edgecolor=edgecolor if edgecolor is not None else color,
        linewidth=linewidth,
        zorder=2,
    )


def _segment_intersection(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    p4: np.ndarray,
) -> np.ndarray | None:
    """Compute the intersection point of two 2D line segments.

    Segments are ``p1-p2`` and ``p3-p4``. Returns the intersection point if
    both parameters are strictly in ``(0, 1)``, otherwise ``None``.
    """
    d1 = p2 - p1
    d2 = p4 - p3
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-12:
        return None
    d3 = p3 - p1
    t = (d3[0] * d2[1] - d3[1] * d2[0]) / cross
    s = (d3[0] * d1[1] - d3[1] * d1[0]) / cross
    if 0.0 < t < 1.0 and 0.0 < s < 1.0:
        return p1 + t * d1
    return None


def _ray_segment_r(
    cos_a: float,
    sin_a: float,
    p_start: np.ndarray,
    p_end: np.ndarray,
) -> float | None:
    """Distance from origin where a ray hits a line segment.

    The ray direction is ``(cos_a, sin_a)``.

    Args:
        cos_a: Cosine of the ray angle.
        sin_a: Sine of the ray angle.
        p_start: Segment start in Cartesian.
        p_end: Segment end in Cartesian.

    Returns:
        Distance along the ray, or ``None`` if no hit.
    """
    dx = p_end[0] - p_start[0]
    dy = p_end[1] - p_start[1]
    denom = dy * cos_a - dx * sin_a
    if abs(denom) < 1e-12:
        return None
    s = (p_start[0] * sin_a - p_start[1] * cos_a) / denom
    if s < -1e-9 or s > 1.0 + 1e-9:
        return None
    r = cos_a * (p_start[0] + s * dx) + sin_a * (p_start[1] + s * dy)
    if r < -1e-9:
        return None
    return float(r)


def _member_r_at_angle(
    angle: float,
    cart_x: np.ndarray,
    cart_y: np.ndarray,
    spoke_pair: tuple[int, int],
    n_members: int,
) -> np.ndarray:
    """Evaluate all members' radii at a given angle via ray-casting.

    Only tests the segment between the given spoke pair.

    Args:
        angle: Ray angle in radians.
        cart_x: Cartesian x-coordinates, shape ``(n_members, n_classes)``.
        cart_y: Cartesian y-coordinates, shape ``(n_members, n_classes)``.
        spoke_pair: Indices ``(i, j)`` of the adjacent spokes.
        n_members: Number of members.

    Returns:
        Array of r-values per member (``nan`` where no intersection).
    """
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    i, j = spoke_pair
    r_vals = np.full(n_members, np.nan)
    for k in range(n_members):
        p_start = np.array([cart_x[k, i], cart_y[k, i]])
        p_end = np.array([cart_x[k, j], cart_y[k, j]])
        r = _ray_segment_r(cos_a, sin_a, p_start, p_end)
        if r is not None:
            r_vals[k] = r
    return r_vals


def _draw_convex_spider_envelope(
    ax: Axes,
    theta: np.ndarray,
    pts: np.ndarray,
    color: object,
    alpha: float,
) -> None:
    """Draw the upper/lower envelope of member distributions.

    Computes Cartesian segment intersections to find exact crossing points
    where member lines switch between being outermost/innermost. Between
    crossings, the fill polygon edges follow the member lines exactly (both
    are straight Cartesian segments).

    Args:
        ax: The RadarAxes to draw on.
        theta: Spoke angles.
        pts: Member distributions, shape ``(n_members, num_classes)``.
        color: Fill color.
        alpha: Fill transparency.
    """
    n_members, n_classes = pts.shape
    two_pi = 2.0 * np.pi
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    cart_x = pts * cos_t[np.newaxis, :]
    cart_y = pts * sin_t[np.newaxis, :]

    # Collect events: (angle, spoke_pair, r_values_per_member)
    # Each event is a point where the envelope might change shape.
    events: list[tuple[float, tuple[int, int], np.ndarray]] = []

    for i in range(n_classes):
        j = (i + 1) % n_classes
        spoke_pair = (i, j)

        # Spoke endpoint i
        events.append((theta[i], spoke_pair, pts[:, i].copy()))

        # Crossings between spokes i and j
        for a in range(n_members):
            pa_i = np.array([cart_x[a, i], cart_y[a, i]])
            pa_j = np.array([cart_x[a, j], cart_y[a, j]])
            for b in range(a + 1, n_members):
                pb_i = np.array([cart_x[b, i], cart_y[b, i]])
                pb_j = np.array([cart_x[b, j], cart_y[b, j]])
                ix = _segment_intersection(pa_i, pa_j, pb_i, pb_j)
                if ix is not None:
                    angle_ix = float(np.arctan2(ix[1], ix[0])) % two_pi
                    # For the wrap-around sector (last spoke to spoke 0),
                    # ensure the angle sorts after the last spoke.
                    if j == 0 and angle_ix < theta[i]:
                        angle_ix += two_pi
                    r_vals = _member_r_at_angle(angle_ix, cart_x, cart_y, spoke_pair, n_members)
                    events.append((angle_ix, spoke_pair, r_vals))

    # Sort by angle
    events.sort(key=lambda e: e[0] % two_pi)

    # Build upper and lower envelope vertices
    upper_angles: list[float] = []
    upper_radii: list[float] = []
    lower_angles: list[float] = []
    lower_radii: list[float] = []

    for angle, _pair, r_vals in events:
        valid = r_vals[~np.isnan(r_vals)]
        if len(valid) == 0:
            continue
        upper_angles.append(angle)
        upper_radii.append(float(np.max(valid)))
        lower_angles.append(angle)
        lower_radii.append(float(np.min(valid)))

    if not upper_angles:
        return

    # Close the envelopes
    upper_angles.append(upper_angles[0] + two_pi)
    upper_radii.append(upper_radii[0])
    lower_angles.append(lower_angles[0] + two_pi)
    lower_radii.append(lower_radii[0])

    # Fill between upper and lower traces
    tt = np.array(upper_angles + lower_angles[::-1])
    rr = np.array(upper_radii + lower_radii[::-1])
    ax.fill(tt, rr, facecolor=color, alpha=alpha, edgecolor="none", zorder=1)


def _draw_intervals_on_spokes(
    ax: Axes,
    theta: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    color: object,
    config: PlotConfig,
    *,
    label: str | None = None,
) -> None:
    """Draw constant-width bars on each spoke.

    Args:
        ax: The RadarAxes to draw on.
        theta: Spoke angles.
        lower: Lower bounds per class.
        upper: Upper bounds per class.
        color: Color for bars.
        config: Plot configuration.
        label: Optional legend label.
    """
    bar_width = config.spider_bar_width * cast("Any", ax).get_rmax()
    for t, lo, hi in zip(theta, lower, upper, strict=True):
        _draw_spider_bar(ax, float(t), float(lo), float(hi), bar_width, color, config.fill_alpha)

    # Invisible artist for the legend entry
    if label is not None:
        ax.fill([], [], facecolor=color, alpha=config.fill_alpha, label=label)


# ---------------------------------------------------------------------------
# flexdispatch handlers
# ---------------------------------------------------------------------------


@flexdispatch
def _draw_credal_set_spider(
    data: ArrayCategoricalCredalSet,
    ax: Axes,
    config: PlotConfig,
    series_labels: list[str] | None,
) -> None:
    """Draw a credal set on spider (radar) axes.

    Args:
        data: The credal set to draw.
        ax: A RadarAxes instance.
        config: Plot configuration.
        series_labels: Optional per-series legend labels.

    Raises:
        NotImplementedError: If no handler is registered for the given type.
    """
    msg = f"Unsupported credal set type: {type(data).__name__}"
    raise NotImplementedError(msg)


@_draw_credal_set_spider.register(ArraySingletonCredalSet)
def _draw_singleton_spider(
    data: ArraySingletonCredalSet,
    ax: Axes,
    config: PlotConfig,
    series_labels: list[str] | None,
) -> None:
    theta = _get_theta(data.num_classes)
    arr = data.array.reshape(-1)
    n_sets = arr.shape[0]
    for idx in range(n_sets):
        color = config.color(idx)
        label = series_labels[idx] if series_labels is not None and idx < len(series_labels) else None
        values = arr[idx].unnormalized_probabilities
        ax.plot(theta, values, color=color, linewidth=config.line_width, label=label, zorder=3)
        ax.scatter(theta, values, color=color, s=config.marker_size, zorder=4)


@_draw_credal_set_spider.register(ArrayProbabilityIntervalsCredalSet)
def _draw_intervals_spider(
    data: ArrayProbabilityIntervalsCredalSet,
    ax: Axes,
    config: PlotConfig,
    series_labels: list[str] | None,
) -> None:
    data = data.reshape(-1)
    theta = _get_theta(data.num_classes)
    lower_all = data.lower()
    upper_all = data.upper()
    n_sets = lower_all.shape[0]
    for idx in range(n_sets):
        color = config.color(idx)
        label = series_labels[idx] if series_labels is not None and idx < len(series_labels) else None
        lower = lower_all[idx]
        upper = upper_all[idx]
        if np.allclose(lower, upper):
            ax.plot(theta, lower, color=color, linewidth=config.line_width, label=label, zorder=3)
            ax.scatter(theta, lower, color=color, s=config.marker_size, zorder=4)
        else:
            _draw_intervals_on_spokes(ax, theta, lower, upper, color, config, label=label)


@_draw_credal_set_spider.register(ArrayDistanceBasedCredalSet)
def _draw_distance_based_spider(
    data: ArrayDistanceBasedCredalSet,
    ax: Axes,
    config: PlotConfig,
    series_labels: list[str] | None,
) -> None:
    data = data.reshape(-1)
    theta = _get_theta(data.num_classes)
    lower_all = data.lower()
    upper_all = data.upper()
    nominal_all = data.nominal.unnormalized_probabilities
    n_sets = lower_all.shape[0]
    for idx in range(n_sets):
        color = config.color(idx)
        label = series_labels[idx] if series_labels is not None and idx < len(series_labels) else None
        _draw_intervals_on_spokes(ax, theta, lower_all[idx], upper_all[idx], color, config, label=label)
        ax.scatter(theta, nominal_all[idx], color=color, s=config.marker_size, zorder=4)


@_draw_credal_set_spider.register(ArrayConvexCredalSet)
def _draw_convex_set_spider(
    data: ArrayConvexCredalSet,
    ax: Axes,
    config: PlotConfig,
    series_labels: list[str] | None,
) -> None:
    theta = _get_theta(data.num_classes)
    arr = data.reshape(-1)
    n_sets = arr.shape[0]
    for idx in range(n_sets):
        color = config.color(idx)
        label = series_labels[idx] if series_labels is not None and idx < len(series_labels) else None
        pts = arr[idx].array.unnormalized_probabilities

        for member in pts:
            ax.plot(theta, member, color=color, linewidth=config.line_width, alpha=0.6, zorder=3)

        _draw_convex_spider_envelope(ax, theta, pts, color, config.fill_alpha)

        if label is not None:
            ax.fill([], [], facecolor=color, alpha=config.fill_alpha, label=label)


@_draw_credal_set_spider.register(ArrayDiscreteCredalSet)
def _draw_discrete_set_spider(
    data: ArrayDiscreteCredalSet,
    ax: Axes,
    config: PlotConfig,
    series_labels: list[str] | None,
) -> None:
    theta = _get_theta(data.num_classes)
    arr = data.reshape(-1)
    n_sets = arr.shape[0]
    for idx in range(n_sets):
        label = series_labels[idx] if series_labels is not None and idx < len(series_labels) else None
        pts = arr[idx].array.unnormalized_probabilities

        for m_idx, member in enumerate(pts):
            # Each member gets a unique color (unlike convex sets which use one color per batch element)
            color = config.color(idx * pts.shape[0] + m_idx)
            m_label = label if m_idx == 0 else None
            ax.plot(theta, member, color=color, linewidth=config.line_width, zorder=3, label=m_label)
            ax.scatter(theta, member, color=color, s=config.marker_size, zorder=4)
