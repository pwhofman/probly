"""==================================================
Second order: a parameterized Dirichlet
==================================================

The other encoding of the second-order rung: instead of drawing members, state the
second-order distribution in closed form. Over the simplex the usual choice is a
Dirichlet, here an
:class:`~probly.representation.distribution.array_dirichlet.ArrayDirichletDistribution`,
which a single forward pass can produce.

The concentration parameters carry both readings at once: their normalization is the mean
prediction, and their total mass is how much evidence the model claims to have.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import dirichlet

from probly.plot import PlotConfig
from probly.representation.distribution import ArrayDirichletDistribution

CLASSES = ["cat", "dog", "fox"]
config = PlotConfig()

# Two inputs with the same mean prediction but very different evidence.
# Shape: (instances, classes)
distribution = ArrayDirichletDistribution(alphas=np.array([[18.0, 14.0, 8.0], [0.9, 0.7, 0.4]]))

print("Shape (batch dims):", distribution.shape)
print("Concentration parameters:\n", distribution.alphas)
print("Mean prediction:\n", distribution.mean.probabilities)
print("Total evidence per input:", distribution.alphas.sum(axis=-1))

# %%
# Ten draws from the second-order distribution are exactly the members a sampled
# representation would have had to produce by running the model ten times.
draws = distribution.sample(num_samples=10, rng=np.random.default_rng(0))
print("Draw shape (members, instances, classes):", draws.samples.probabilities.shape)


# %%
# On the simplex the density is a surface rather than a scatter: concentrated mass means
# the odds are pinned down, mass pushed out to the corners means they are not. The
# surfaces are drawn on a log scale, each panel with its own color range, because the two
# densities differ by orders of magnitude.
def to_cartesian(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map points on the 3-simplex to plot coordinates."""
    x = probabilities[..., 1] + 0.5 * probabilities[..., 2]
    y = (np.sqrt(3) / 2) * probabilities[..., 2]
    return x, y


resolution = 300
first, second = np.meshgrid(np.linspace(0, 1, resolution), np.linspace(0, 1, resolution))
inside = (first + second) <= 1
grid = np.stack([first[inside], second[inside], 1.0 - first[inside] - second[inside]], axis=-1)
grid = np.clip(grid, 1e-6, None)
grid /= grid.sum(axis=-1, keepdims=True)
grid_x, grid_y = to_cartesian(grid)

fig, axes = plt.subplots(1, 2, figsize=(7.5, 4))
titles = ["Input 0: much evidence", "Input 1: little evidence"]
for index, (ax, title) in enumerate(zip(axes, titles, strict=True)):
    density = dirichlet.pdf(grid.T, distribution.alphas[index])
    log_density = np.log(np.clip(density, 1e-12, None))
    levels = np.linspace(*np.percentile(log_density, [2, 99]), 60)
    ax.tricontourf(grid_x, grid_y, log_density, levels=levels, cmap="viridis", extend="both")

    corners = np.eye(3)
    corner_x, corner_y = to_cartesian(corners)
    ax.plot([*corner_x, corner_x[0]], [*corner_y, corner_y[0]], color=config.color_neutral, lw=1)
    for name, x, y in zip(CLASSES, corner_x, corner_y, strict=True):
        ax.annotate(name, (x, y), ha="center", va="center", xytext=(0, 12 if y > 0 else -12), textcoords="offset points")
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.15, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title)
fig.suptitle("Second order, parameterized: a Dirichlet over the simplex")
fig.tight_layout()
plt.show()

# %%
# Both inputs share a mean prediction, so both collapse to the same first-order
# distribution. The resolution of the second-order object is fixed by the concentration
# parameters rather than bought with a sample count, which is what makes this encoding
# cheap at inference and expensive at training.
