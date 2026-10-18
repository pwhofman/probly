"""======================================
SNGP Selective Prediction on Two Moons
======================================

Calibrate a selector on unlabeled data to a target coverage, then see what it does far from the data.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.datasets import make_moons
import torch
from torch import nn

from probly.method.sngp import reset_precision_matrix, sngp
from probly.selective_prediction import SelectivePredictor, selective_predictor, selector

from examples.utils.model import ResFFN

BLUE, RED, GRAY = "#1e88e5", "#ff0d57", "#7f8c8d"

# %%
# Data
# ----
#
# Out-of-distribution points lie on a ring around the two moons. A validation split from the training distribution
# is held out to calibrate the selector. Its labels are not used.

X_train, y_train = make_moons(n_samples=500, noise=0.25, random_state=0)
X_val, _ = make_moons(n_samples=1000, noise=0.25, random_state=2)
X_test, y_test = make_moons(n_samples=1000, noise=0.25, random_state=1)

angles = np.random.default_rng(0).uniform(0, 2 * np.pi, 500)
X_ood = np.c_[0.5 + 4 * np.cos(angles), 0.25 + 3 * np.sin(angles)]

# %%
# Transform, then train
# ---------------------

torch.manual_seed(0)
model = sngp(ResFFN(), num_random_features=128, ridge_penalty=0.01, norm_multiplier=0.9, n_power_iterations=1)

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
X_train_t = torch.from_numpy(X_train).float()
y_train_t = torch.from_numpy(y_train).long()
model.train()
for _ in range(300):
    reset_precision_matrix(model)
    logits, _ = model(X_train_t)
    opt.zero_grad()
    nn.functional.cross_entropy(logits, y_train_t).backward()
    opt.step()
reset_precision_matrix(model)
with torch.no_grad():
    model(X_train_t)
model.eval()

# %%
# Calibrate to a coverage
# -----------------------
#
# A :func:`~probly.selective_prediction.selector` with ``coverage=0.8`` sets its threshold on the criterion of
# unlabeled validation data, so that about 80 percent of new instances are accepted. The guarantee assumes that new
# instances are exchangeable with the validation data. The test set meets this assumption, but the ring does not, and
# SNGP is uncertain far from the data, so almost none of its points are accepted.


def run(predictor: SelectivePredictor, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with torch.no_grad():
        result = predictor.predict(torch.from_numpy(x).float())
    return result.decision.probabilities.argmax(dim=-1).numpy(), result.uncertainty.numpy(), result.accepted.numpy()


predictor = selective_predictor(model, selector=selector(coverage=0.8), representer_kwargs={"num_samples": 100})
with torch.no_grad():
    predictor.calibrate(None, torch.from_numpy(X_val).float())

prediction, _, accepted = run(predictor, X_test)
correct = prediction == y_test
ood_accepted = run(predictor, X_ood)[2]
print(f"test  coverage {accepted.mean():.2f}  accuracy {correct[accepted].mean():.3f}  (target coverage 0.80)")
print(f"OOD   coverage {ood_accepted.mean():.2f}")

# %%
# Decision regions
# ----------------
#
# Gray regions are abstained on. Circles mark OOD points that are accepted.

xx, yy = np.meshgrid(np.linspace(-4.5, 5.5, 300), np.linspace(-3.5, 4.0, 300))
grid = np.c_[xx.ravel(), yy.ravel()]

grid_prediction, _, grid_accepted = run(predictor, grid)
panels = [
    ("Normal prediction", grid_prediction, np.ones_like(ood_accepted)),
    ("Selective prediction", np.where(grid_accepted, grid_prediction, 2), ood_accepted),
]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
for ax, (title, regions, ring_accepted) in zip(axes, panels, strict=True):
    ax.contourf(xx, yy, regions.reshape(xx.shape), levels=[-0.5, 0.5, 1.5, 2.5], colors=[BLUE, RED, GRAY], alpha=0.3)
    ax.scatter(*X_test.T, c=y_test, cmap=ListedColormap([BLUE, RED]), s=6, alpha=0.6)
    ax.scatter(*X_ood.T, facecolors="none", edgecolors=GRAY, s=10, linewidths=0.5)
    ax.scatter(
        *X_ood[ring_accepted].T,
        facecolors="none",
        edgecolors="black",
        s=10,
        linewidths=0.8,
        label=f"{ring_accepted.sum()} accepted OOD",
    )
    ax.set_title(title)
    ax.set_xlabel("$x_1$")
    ax.legend(loc="lower right")
axes[0].set_ylabel("$x_2$")
plt.tight_layout()
plt.show()
