"""Figures of the simplex attribution plot and of the three regimes of features."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from matplotlib.colors import LinearSegmentedColormap, to_rgb
import mpltern  # noqa: F401  # registers the "ternary" projection
import numpy as np

from uqxai import simplex

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes

# One color per class, fixed: the color follows the class through every figure.
CLASS_COLORS = ("#eb6834", "#2a78d6", "#1baf7a")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
NEUTRAL = "#c9c8c2"
# Diverging map for signed attributions: two hues and a neutral midpoint.
DIVERGING = LinearSegmentedColormap.from_list("uqxai_diverging", ["#2a78d6", "#f0efec", "#e34948"])

STYLE = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.labelcolor": INK_SECONDARY,
    "axes.edgecolor": INK_MUTED,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "font.size": 9,
}


def simplex_colors(points: np.ndarray, colors: Sequence[str] = CLASS_COLORS) -> np.ndarray:
    """Color points of the simplex: a class's color at its vertex, neutral gray at the barycenter.

    Args:
        points: Barycentric coordinates of shape ``(n, n_classes)``.
        colors: One color per class.

    Returns:
        RGB colors of shape ``(n, 3)``.
    """
    palette = np.array([to_rgb(color) for color in colors])
    excess = points - points.min(axis=1, keepdims=True)
    saturation = excess.sum(axis=1, keepdims=True)
    return np.clip((1 - saturation) * np.array(to_rgb(NEUTRAL)) + excess @ palette, 0, 1)


def relative_magnitude(attributions: np.ndarray, noise_floor: float = 0.0) -> np.ndarray:
    """Return each feature's magnitude above the noise floor as a fraction of the largest one."""
    magnitudes = np.clip(simplex.magnitude(attributions) - noise_floor, 0, None)
    return magnitudes / max(magnitudes.max(), 1e-12)


def ternary_plot(
    ax: Axes,
    attributions: np.ndarray,
    class_names: Sequence[str],
    *,
    prediction: np.ndarray | None = None,
    baseline: np.ndarray | None = None,
    members: np.ndarray | None = None,
    mode: str = "contrast",
    prediction_note: str = "",
    embedding: dict[str, Any] | None = None,
) -> None:
    """Draw the simplex attribution plot of a three-class problem.

    Every feature is a dot. Its direction from the center says which classes it speaks for and
    against, its distance to the center grows with the standard deviation of its attributions
    across the classes (in proportion to it by default), and its size shows its magnitude. The
    model's predictions are drawn in the same triangle, where they are probabilities.

    With a noise floor in ``embedding``, the features whose magnitude lies below it are not drawn
    one by one: a single gray dot at the center stands for them, and the legend gives their number.

    Args:
        ax: An axes with the ``"ternary"`` projection.
        attributions: Attributions of shape ``(n_features, 3)``.
        class_names: Names of the three classes.
        prediction: Predicted probabilities of the full image, of shape ``(3,)``.
        baseline: Predicted probabilities of the image without any feature, of shape ``(3,)``.
        members: Predicted probabilities of each ensemble member, of shape ``(n_members, 3)``.
        mode: How attributions become points; see :func:`uqxai.simplex.simplex_embedding`.
        prediction_note: Text appended to the legend entries of the predictions.
        embedding: Options of :func:`uqxai.simplex.simplex_embedding`: ``scale``, ``noise_floor``,
            ``exponent`` and ``sharpness``.
    """
    embedding = embedding or {}
    points = simplex.simplex_embedding(attributions, mode=mode, **embedding)
    size = relative_magnitude(attributions, embedding.get("noise_floor", 0.0))
    order = np.argsort(size)  # strong features on top
    order = order[size[order] > 0]
    if len(order) < len(size):
        ax.scatter(
            [1 / 3], [1 / 3], [1 / 3], s=42, c=NEUTRAL, edgecolors=INK_MUTED, linewidths=0.8, zorder=2,
            label=f"{len(size) - len(order)} features below the noise floor",
        )  # fmt: skip
    ax.scatter(
        *points[order].T,
        s=6 + 170 * size[order],
        c=np.column_stack([simplex_colors(points[order]), 0.35 + 0.65 * size[order]]),
        edgecolors=SURFACE,
        linewidths=0.6,
        zorder=3,
    )
    if members is not None:
        ax.scatter(
            *members.T, marker="x", s=22, c=INK_SECONDARY, linewidths=0.9, zorder=4, label="members' predictions"
        )
    if baseline is not None:
        ax.scatter(
            *baseline[None].T, marker="o", s=70, facecolors="none", edgecolors=INK, linewidths=1.3, zorder=5,
            label=f"prediction, no features{prediction_note}",
        )  # fmt: skip
    if prediction is not None:
        ax.scatter(
            *prediction[None].T, marker="*", s=230, c=INK, edgecolors=SURFACE, linewidths=0.8, zorder=6,
            label=f"prediction, full image{prediction_note}",
        )  # fmt: skip
    for setter, name, color in zip(
        (ax.set_tlabel, ax.set_llabel, ax.set_rlabel), class_names, CLASS_COLORS, strict=True
    ):
        setter(
            name,
            color=INK,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.25", "fc": color, "ec": "none", "alpha": 0.25},
        )
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ticks = [0.2, 0.4, 0.6, 0.8]
    for axis in (ax.taxis, ax.laxis, ax.raxis):
        axis.set_ticks(ticks)
        axis.set_tick_params(colors=INK_MUTED, labelsize=7, length=2)
    for spine in ax.spines.values():
        spine.set_color(INK_MUTED)


def regime_plane(ax: Axes, attributions: np.ndarray, background_threshold: float, mode: str = "contrast") -> None:
    """Draw every feature in the plane of its common and its contrast magnitude.

    The distance to the origin is the feature's magnitude. Background features lie inside the
    circle, class-specific features above the diagonals (their attributions vary across the
    classes), and shared features below them (their attributions are alike for all classes).

    Args:
        ax: The axes.
        attributions: Attributions of shape ``(n_features, n_classes)``.
        background_threshold: Magnitude below which a feature is background.
        mode: How the colors of the features are computed; see :func:`ternary_plot`.
    """
    x, y = simplex.common_magnitude(attributions), simplex.contrast_magnitude(attributions)
    size = relative_magnitude(attributions)
    order = np.argsort(size)
    limit = 1.12 * max(np.abs(x).max(), y.max(), background_threshold)
    angles = np.linspace(0, np.pi, 100)
    ax.plot([-limit, 0, limit], [limit, 0, limit], color=INK_MUTED, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
    ax.fill_between(
        background_threshold * np.cos(angles), background_threshold * np.sin(angles), color=GRID, zorder=2, linewidth=0
    )
    ax.scatter(
        x[order],
        y[order],
        s=10 + 90 * size[order],
        c=simplex_colors(simplex.simplex_embedding(attributions, mode=mode))[order],
        edgecolors=SURFACE,
        linewidths=0.6,
        zorder=3,
    )
    regime = simplex.regimes(attributions, background_threshold)
    shared_for = int(np.sum((regime == simplex.SHARED) & (x > 0)))
    shared_against = int(np.sum((regime == simplex.SHARED) & (x < 0)))
    labels = [
        (-0.42 * limit, 0.93 * limit, "center", f"class-specific ({np.sum(regime == simplex.SPECIFIC)})"),
        (0.98 * limit, 0.36 * limit, "right", f"shared,\nfor all classes ({shared_for})"),
        (-0.98 * limit, 0.36 * limit, "left", f"shared,\nagainst all classes ({shared_against})"),
        (0.0, -0.085 * limit, "center", f"background ({np.sum(regime == simplex.BACKGROUND)})"),
    ]
    for label_x, label_y, align, text in labels:
        ax.text(label_x, label_y, text, ha=align, va="center", color=INK_SECONDARY, fontsize=8)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-0.15 * limit, limit)
    ax.set_aspect("equal")
    ax.axhline(0, color=INK_MUTED, linewidth=0.8, zorder=1)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_position(("outward", 4))
    ax.set_xlabel(r"common magnitude  $\sqrt{C}\,\mathrm{mean}_c\,\phi_i^c$  (all classes alike)")
    ax.set_ylabel(r"contrast magnitude  $\sqrt{C\,\mathrm{var}_c\,\phi_i^c}$")


def _grayscale(image: np.ndarray) -> np.ndarray:
    gray = image @ np.array([0.299, 0.587, 0.114])
    return np.repeat((0.35 + 0.65 * gray)[..., None], 3, axis=2)


def simplex_map(
    ax: Axes,
    image: np.ndarray,
    cells: np.ndarray,
    attributions: np.ndarray,
    mode: str = "contrast",
    embedding: dict[str, Any] | None = None,
) -> None:
    """Color every superpixel of the image like its dot in the simplex attribution plot.

    Args:
        ax: The axes.
        image: The image as RGB values in ``[0, 1]``, of shape ``(height, width, 3)``.
        cells: Index of the superpixel of each pixel, of shape ``(height, width)``.
        attributions: Attributions of shape ``(n_features, n_classes)``.
        mode: How attributions become points of the simplex; see :func:`ternary_plot`.
        embedding: Options of :func:`uqxai.simplex.simplex_embedding`; see :func:`ternary_plot`.
    """
    embedding = embedding or {}
    colors = simplex_colors(simplex.simplex_embedding(attributions, mode=mode, **embedding))[cells]
    opacity = (0.92 * relative_magnitude(attributions, embedding.get("noise_floor", 0.0)) ** 0.6)[cells][..., None]
    ax.imshow((1 - opacity) * _grayscale(image) + opacity * colors)
    ax.set_axis_off()


def signed_map(ax: Axes, image: np.ndarray, cells: np.ndarray, values: np.ndarray, limit: float | None = None) -> None:
    """Overlay signed attributions of one output on the image with a diverging color map.

    Args:
        ax: The axes.
        image: The image as RGB values in ``[0, 1]``, of shape ``(height, width, 3)``.
        cells: Index of the superpixel of each pixel, of shape ``(height, width)``.
        values: Attributions of shape ``(n_features,)``.
        limit: Attribution at which the color saturates; defaults to the largest absolute value.
    """
    limit = limit or max(np.abs(values).max(), 1e-12)
    relative = np.clip(values / limit, -1, 1)[cells]
    colors = DIVERGING(0.5 + 0.5 * relative)[..., :3]
    opacity = (0.92 * np.abs(relative) ** 0.6)[..., None]
    ax.imshow((1 - opacity) * _grayscale(image) + opacity * colors)
    ax.set_axis_off()
