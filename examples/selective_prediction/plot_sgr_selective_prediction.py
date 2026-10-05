"""======================================
Selective Prediction with Risk Control
======================================

Choose the threshold on labeled calibration data so that the risk of the accepted predictions stays below a target
(Geifman and El-Yaniv, 2017).
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
import torch
from torch import nn

from probly.selective_prediction import SelectivePredictor, SGRSelector
from probly.transformation import dropout

from examples.utils.model import MLPClassifier

BLUE, RED = "#1e88e5", "#ff0d57"

# %%
# Data and model
# --------------

X, y = make_moons(n_samples=10500, noise=0.25, random_state=0)
X_train, y_train = X[:500], y[:500]
X_pool, y_pool = torch.from_numpy(X[500:]).float(), y[500:]

torch.manual_seed(0)
model = dropout(MLPClassifier(), p=0.2, predictor_type="logit_classifier")

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
# Calibrate and predict
# ---------------------
#
# SGR needs the labels of the calibration instances, passed first. It returns the largest threshold it can
# certify, such that with probability at least ``1 - delta`` over the calibration set, the risk of the accepted
# predictions is at most ``risk``. The guarantee says nothing about the coverage. As in the paper, half of the pool
# calibrates and half tests, and ``delta = 0.001``.

risk, delta = 0.02, 0.001
X_cal, y_cal = X_pool[:5000], y_pool[:5000]
X_test, y_test = X_pool[5000:], y_pool[5000:]

predictor = SelectivePredictor(model, SGRSelector(risk, delta), representer_kwargs={"num_samples": 100})
with torch.no_grad():
    predictor.calibrate(torch.from_numpy(y_cal), X_cal)
    result = predictor.predict(X_test)

accepted = result.accepted.numpy()
wrong = result.decision.probabilities.argmax(dim=-1).numpy() != y_test
print(f"target risk {risk:.2f}  threshold {predictor.selector.threshold:.3f}  bound {predictor.selector.bound:.3f}")
print(f"test risk {wrong[accepted].mean():.3f}  test coverage {result.coverage:.3f}")

# %%
# Risk control for several targets
# --------------------------------
#
# The table follows the paper's Table 1: for each target risk ``r*``, the risk and coverage on the calibration
# ("train") and test halves, and the certified bound ``b*``. The figure follows its Fig. 2: the test risk-coverage
# curve of the criterion, with the SGR operating points on it. The uncertainties and errors are computed once, and
# the selectors are calibrated on the arrays.

with torch.no_grad():
    pooled = predictor.predict(X_pool)
kappa = pooled.uncertainty.numpy()
errors = (pooled.decision.probabilities.argmax(dim=-1).numpy() != y_pool).astype(float)
cal, test = np.arange(5000), np.arange(5000, len(kappa))

print(f"{'r*':>6} {'cal risk':>9} {'cal cov':>8} {'test risk':>9} {'test cov':>8} {'b*':>6}")
points = []
target_risks = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035]
for r in target_risks:
    selector = SGRSelector(r, delta)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        selector.calibrate(kappa[cal], errors[cal])
    rows = []
    for split in (cal, test):
        mask = selector.select(kappa[split])
        rows += [errors[split][mask].mean() if mask.any() else np.nan, mask.mean()]
    points.append((rows[3], rows[2]))
    print(f"{r:6.3f} {rows[0]:9.3f} {rows[1]:8.3f} {rows[2]:9.3f} {rows[3]:8.3f} {selector.bound:6.3f}")
print(f"full-coverage test risk {errors[test].mean():.3f}")

order = np.argsort(kappa[test], kind="stable")
curve_risk = np.cumsum(errors[test][order]) / np.arange(1, len(order) + 1)
curve_coverage = np.arange(1, len(order) + 1) / len(order)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(curve_coverage, curve_risk, color=BLUE, label="test risk-coverage curve")
coverages, risks = zip(*points, strict=True)
ax.scatter(coverages, risks, color=RED, zorder=3, label="SGR operating points")
ax.scatter(coverages, target_risks, color=RED, marker="_", s=200, label="target risk")
ax.set_xlabel("Coverage")
ax.set_ylabel("Risk")
ax.set_ylim(0, 1.1 * errors[test].mean())
ax.legend(loc="upper left")
plt.tight_layout()
plt.show()
