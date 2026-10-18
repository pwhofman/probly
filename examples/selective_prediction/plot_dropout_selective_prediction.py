"""============================================
MC Dropout Selective Prediction on Two Moons
============================================

Abstain wherever an MC-dropout model is too uncertain, with a threshold that is the cost of abstaining.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.datasets import make_moons
import torch
from torch import nn

from probly.selective_prediction import selective_predictor, selector
from probly.transformation import dropout

from examples.utils.model import MLPClassifier

BLUE, RED, GRAY = "#1e88e5", "#ff0d57", "#7f8c8d"

# %%
# Data
# ----

X_train, y_train = make_moons(n_samples=500, noise=0.25, random_state=0)
X_test, y_test = make_moons(n_samples=1000, noise=0.25, random_state=1)

# %%
# Transform, then train
# ---------------------

torch.manual_seed(0)
model = dropout(MLPClassifier(), p=0.2, predictor_type="logit_classifier", shared_mask=True)

opt = torch.optim.Adam(model.parameters(), lr=1e-2)
X_train_t = torch.from_numpy(X_train).float()
y_train_t = torch.from_numpy(y_train).long()
model.train()
for _ in range(500):
    opt.zero_grad()
    nn.functional.cross_entropy(model(X_train_t), y_train_t).backward()
    opt.step()
model.eval()

# %%
# Selective prediction
# --------------------
#
# The default criterion is the total uncertainty under the zero-one loss. It is one minus the largest mean
# probability over the dropout samples, the model's own probability that its prediction is wrong. A
# :func:`~probly.selective_prediction.selector` with ``threshold=0.1`` accepts a prediction if that probability is at
# most 0.1. This is Chow's rule for an abstention that costs a tenth of a wrong prediction, so the threshold has a
# meaning and needs no calibration data.

predictor = selective_predictor(model, selector=selector(threshold=0.1), representer_kwargs={"num_samples": 100})


def run(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        result = predictor.predict(torch.from_numpy(x).float())
    return result.decision.probabilities.argmax(dim=-1).numpy(), result.accepted.numpy()


prediction, accepted = run(X_test)
correct = prediction == y_test
print(f"normal prediction  coverage 1.00  accuracy {correct.mean():.3f}")
print(f"selective          coverage {accepted.mean():.2f}  accuracy {correct[accepted].mean():.3f}")

# %%
# Decision regions
# ----------------
#
# Gray regions are abstained on. Crosses mark wrong predictions that are not abstained on.

xx, yy = np.meshgrid(np.linspace(-3.0, 4.0, 300), np.linspace(-2.5, 3.0, 300))
grid_prediction, grid_accepted = run(np.c_[xx.ravel(), yy.ravel()])

panels = [
    ("Normal prediction", grid_prediction, np.ones_like(correct)),
    ("Selective prediction", np.where(grid_accepted, grid_prediction, 2), accepted),
]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
for ax, (title, regions, shown) in zip(axes, panels, strict=True):
    ax.contourf(xx, yy, regions.reshape(xx.shape), levels=[-0.5, 0.5, 1.5, 2.5], colors=[BLUE, RED, GRAY], alpha=0.3)
    ax.scatter(*X_test.T, c=y_test, cmap=ListedColormap([BLUE, RED]), s=6, alpha=0.6)
    errors = shown & ~correct
    ax.scatter(*X_test[errors].T, marker="x", color="black", s=20, label=f"{errors.sum()} accepted errors")
    ax.set_title(title)
    ax.set_xlabel("$x_1$")
    ax.legend(loc="lower right")
axes[0].set_ylabel("$x_2$")
plt.tight_layout()
plt.show()
