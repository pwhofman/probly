"""=========================================
Second order: a sample of distributions
=========================================

A sampled second-order representation is a finite collection of first-order
distributions -- ensemble members, dropout passes, posterior weight samples. In
``probly`` that is an
:class:`~probly.representation.distribution.array_categorical.ArrayCategoricalDistributionSample`.

Drawn on the 3-simplex, each member is one point. How far the points are apart is the
epistemic part: tightly clustered members mean the odds are pinned down, scattered
members mean the model does not know the odds. Both inputs below have almost the same
mean prediction, which is the number a first-order representation would have reported.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from probly.plot import PlotConfig
from probly.representation.distribution import (
    ArrayCategoricalDistributionSample,
    ArrayProbabilityCategoricalDistribution,
)

CLASSES = ["cat", "dog", "fox"]
config = PlotConfig()
rng = np.random.default_rng(0)

# Ten members per input, for two inputs. Shape: (members, instances, classes)
center = np.array([[0.45, 0.35, 0.20], [0.45, 0.35, 0.20]])
spread = np.array([0.01, 0.30])  # input 0 agrees, input 1 does not
members = np.clip(center + spread[None, :, None] * rng.normal(size=(10, 2, 3)), 1e-6, None)
members /= members.sum(axis=-1, keepdims=True)

sample = ArrayCategoricalDistributionSample(
    array=ArrayProbabilityCategoricalDistribution(array=members),
    sample_axis=0,
)

print("Number of members:", sample.sample_size)
print("Mean prediction:\n", sample.sample_mean().probabilities)
print("Spread between members (per class):\n", members.std(axis=0))


# %%
# Barycentric coordinates place a three-class distribution inside a triangle: each corner
# is one class taking all the mass.
def to_cartesian(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map points on the 3-simplex to plot coordinates."""
    x = probabilities[..., 1] + 0.5 * probabilities[..., 2]
    y = (np.sqrt(3) / 2) * probabilities[..., 2]
    return x, y


def draw_simplex(ax: plt.Axes) -> None:
    """Draw the triangle outline and label its corners with the class names."""
    corners = np.eye(3)
    corner_x, corner_y = to_cartesian(corners)
    ax.plot([*corner_x, corner_x[0]], [*corner_y, corner_y[0]], color=config.color_neutral, lw=1)
    for name, x, y in zip(CLASSES, corner_x, corner_y, strict=True):
        ax.annotate(name, (x, y), ha="center", va="center", xytext=(0, 12 if y > 0 else -12), textcoords="offset points")
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.15, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")


fig, axes = plt.subplots(1, 2, figsize=(7.5, 4))
titles = ["Input 0: members agree", "Input 1: members disagree"]
for index, (ax, title) in enumerate(zip(axes, titles, strict=True)):
    draw_simplex(ax)
    member_x, member_y = to_cartesian(members[:, index, :])
    ax.scatter(member_x, member_y, color=config.categorical_palette[0], s=28, zorder=3, label="member")
    mean_x, mean_y = to_cartesian(sample.sample_mean().probabilities[index])
    ax.scatter(mean_x, mean_y, color=config.color_positive, s=70, marker="X", zorder=4, label="mean")
    ax.set_title(title)
axes[0].legend(loc="upper left", frameon=False, fontsize=9)
fig.suptitle("Second order, sampled: ten distributions on the simplex")
fig.tight_layout()
plt.show()

# %%
# The mean predictions almost coincide, so a first-order representation would report the
# same thing for both inputs. The spread between the members is the information the
# second-order rung adds -- and because the members are a finite draw, every measure read
# off them is an estimate whose value moves with the member count.
