"""======================================
SNGP Selective Prediction on Two Moons
======================================

Abstain wherever an SNGP model is too uncertain, both near the class boundary and far from the data.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.datasets import make_moons
import torch
from torch import nn

from probly.method.sngp import reset_precision_matrix, sngp
from probly.quantification import LogLoss
from probly.selective_prediction import SelectivePredictor, ThresholdSelector

from examples.utils.model import ResFFN

BLUE, RED, GRAY, PURPLE, TEAL = "#1e88e5", "#ff0d57", "#7f8c8d", "#9b59b6", "#16a085"

# %%
# Data
# ----
#
# Out-of-distribution points lie on a ring around the two moons.

X_train, y_train = make_moons(n_samples=500, noise=0.25, random_state=0)
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
# Selective prediction
# --------------------
#
# The recommended criterion is the total uncertainty under the zero-one loss, the default. It is one minus the
# largest mean probability, the model's own probability that its prediction is wrong, so a threshold of 0.1 is
# Chow's rule for an abstention that costs a tenth of an error. The epistemic uncertainty, in contrast, ignores the
# noise in the labels and measures only the model's lack of knowledge. It is better suited to rejecting
# out-of-distribution instances, and for that purpose, the log loss, whose epistemic part is the mutual
# information, works better than the zero-one loss.

sampling = {"num_samples": 100}
thresholds = {"total": 0.1, "epistemic": 0.03}
predictors = {
    "total": SelectivePredictor(model, ThresholdSelector(thresholds["total"]), representer_kwargs=sampling),
    "epistemic": SelectivePredictor(
        model,
        ThresholdSelector(thresholds["epistemic"]),
        notion="epistemic",
        loss=LogLoss(),
        representer_kwargs=sampling,
    ),
}


def run(predictor: SelectivePredictor, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with torch.no_grad():
        result = predictor(torch.from_numpy(x).float())
    prediction = result.decision.probabilities.argmax(dim=-1).numpy()
    return prediction, result.uncertainty.numpy(), result.accepted.numpy()


test = {notion: run(predictor, X_test) for notion, predictor in predictors.items()}
ood = {notion: run(predictor, X_ood) for notion, predictor in predictors.items()}
prediction = test["total"][0]
correct = prediction == y_test

print(f"normal prediction    coverage 1.00  accuracy {correct.mean():.3f}  OOD coverage 1.00")
for notion, (pred, _, accepted) in test.items():
    print(
        f"selective ({notion:9s}) coverage {accepted.mean():.2f}  accuracy {(pred == y_test)[accepted].mean():.3f}"
        f"  OOD coverage {ood[notion][2].mean():.2f}"
    )

# %%
# Decision regions
# ----------------
#
# Crosses mark wrong predictions that are not abstained on, circles mark accepted OOD points.

xx, yy = np.meshgrid(np.linspace(-4.5, 5.5, 300), np.linspace(-3.5, 4.0, 300))
grid = np.c_[xx.ravel(), yy.ravel()]
on_grid = {notion: run(predictor, grid) for notion, predictor in predictors.items()}

panels = [("Normal prediction", on_grid["total"][0], prediction, np.ones_like(correct), np.ones(len(X_ood), bool))]
for notion, (pred, _, accepted) in on_grid.items():
    panels.append(
        (f"Selective ({notion})", np.where(accepted, pred, 2), test[notion][0], test[notion][2], ood[notion][2])
    )

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)
for ax, (title, regions, pred, accepted, ood_accepted) in zip(axes, panels, strict=True):
    ax.contourf(xx, yy, regions.reshape(xx.shape), levels=[-0.5, 0.5, 1.5, 2.5], colors=[BLUE, RED, GRAY], alpha=0.3)
    ax.scatter(*X_test.T, c=y_test, cmap=ListedColormap([BLUE, RED]), s=6, alpha=0.6)
    errors = accepted & (pred != y_test)
    ax.scatter(*X_test[errors].T, marker="x", color="black", s=20, label=f"{errors.sum()} accepted errors")
    ax.scatter(*X_ood.T, facecolors="none", edgecolors=GRAY, s=10, linewidths=0.5)
    ax.scatter(
        *X_ood[ood_accepted].T,
        facecolors="none",
        edgecolors="black",
        s=10,
        linewidths=0.8,
        label=f"{ood_accepted.sum()} accepted OOD",
    )
    ax.set_title(title)
    ax.set_xlabel("$x_1$")
    ax.legend(loc="lower right")
axes[0].set_ylabel("$x_2$")
plt.tight_layout()
plt.show()

# %%
# Uncertainty of correct, wrong, and OOD predictions
# --------------------------------------------------

scales = {"total": "1 - max probability", "epistemic": "mutual information"}
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, (notion, (pred, uncertainty, _)) in zip(axes, test.items(), strict=True):
    hit = pred == y_test
    ood_uncertainty = ood[notion][1]
    bins = np.linspace(0, max(uncertainty.max(), ood_uncertainty.max()), 30)
    ax.hist(uncertainty[hit], bins=bins, color=BLUE, alpha=0.6, label="correct")
    ax.hist(uncertainty[~hit], bins=bins, color=RED, alpha=0.6, label="wrong")
    ax.hist(ood_uncertainty, bins=bins, color=GRAY, alpha=0.6, label="OOD")
    ax.axvline(thresholds[notion], color="black", linestyle="--", label="threshold")
    ax.set_yscale("log")
    ax.set_xlabel(f"{notion} uncertainty ({scales[notion]})")
    ax.set_ylabel("count")
    ax.legend()
plt.tight_layout()
plt.show()

# %%
# Accuracy vs. coverage
# ---------------------
#
# Dots mark the thresholds used above.

fig, ax = plt.subplots(figsize=(7, 4.5))
for (notion, (pred, uncertainty, accepted)), color in zip(test.items(), [PURPLE, TEAL], strict=True):
    hit = (pred == y_test)[np.argsort(uncertainty, kind="stable")]
    n = np.arange(1, len(hit) + 1)
    ax.plot(n / len(hit), np.cumsum(hit) / n, color=color, label=f"selective ({notion})")
    ax.scatter(accepted.mean(), (pred == y_test)[accepted].mean(), color=color, zorder=3)
ax.axhline(correct.mean(), color="black", linestyle="--", label="normal prediction")
ax.set_xlabel("coverage")
ax.set_ylabel("accuracy on accepted")
ax.set_xlim(0, 1)
ax.legend(loc="lower left")
plt.tight_layout()
plt.show()

# %%
# Coverage in and out of distribution
# -----------------------------------

labels = ["normal", *(f"selective\n({notion})" for notion in predictors)]
in_coverage = [1.0, *(test[notion][2].mean() for notion in predictors)]
out_coverage = [1.0, *(ood[notion][2].mean() for notion in predictors)]

fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(labels))
ax.bar(x - 0.2, in_coverage, width=0.4, color=BLUE, label="test")
ax.bar(x + 0.2, out_coverage, width=0.4, color=GRAY, label="OOD")
ax.set_xticks(x, labels)
ax.set_ylabel("coverage")
ax.set_ylim(0, 1.05)
ax.legend()
plt.tight_layout()
plt.show()
