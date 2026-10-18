"""======================================
Selective Prediction with Risk Control
======================================

Choose the threshold of a plain neural network on labeled calibration data so that the risk of the accepted
predictions stays below a target (Geifman and El-Yaniv, 2017).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
import torch
from torch import nn

from probly.metrics.selective_prediction import risk_coverage_curve
from probly.selective_prediction import selective_predictor, selector

from examples.utils.model import MLPClassifier

BLUE, RED = "#1e88e5", "#ff0d57"

# %%
# Data and model
# --------------
#
# The network is an ordinary torch model. It is trained as usual, without a probly transformation.

X, y = make_moons(n_samples=10500, noise=0.25, random_state=0)
X_train, y_train = X[:500], y[:500]
X_pool, y_pool = torch.from_numpy(X[500:]).float(), y[500:]

torch.manual_seed(0)
model = MLPClassifier()

opt = torch.optim.Adam(model.parameters(), lr=1e-2)
X_train_t = torch.from_numpy(X_train).float()
y_train_t = torch.from_numpy(y_train).long()
for _ in range(500):
    opt.zero_grad()
    nn.functional.cross_entropy(model(X_train_t), y_train_t).backward()
    opt.step()
model.eval()

# %%
# Calibrate and predict
# ---------------------
#
# ``predictor_type="logit_classifier"`` tells :func:`~probly.selective_prediction.selective_predictor` to treat the
# network as a classifier that outputs logits. Its criterion is then one minus the softmax probability of the
# prediction, the softmax response of the paper.
#
# SGR needs the labels of the calibration instances, passed first. It returns the largest threshold it can certify,
# such that with probability at least ``1 - delta`` over the calibration set, the risk of the accepted predictions is
# at most ``risk``. The guarantee says nothing about the coverage. As in the paper, half of the pool calibrates and
# half tests, and ``delta = 0.001``.

risk, delta = 0.02, 0.001
X_cal, y_cal = X_pool[:5000], y_pool[:5000]
X_test, y_test = X_pool[5000:], y_pool[5000:]

predictor = selective_predictor(model, selector=selector(risk=risk, delta=delta), predictor_type="logit_classifier")
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
# For each target risk, the rule is fitted on the calibration half and applied to the test half. The figure shows
# the test risk-coverage curve of the criterion, from :func:`~probly.metrics.selective_prediction.risk_coverage_curve`,
# with the SGR operating points on it. They stay below their targets, and the gap to the curve is the price of the
# guarantee.

with torch.no_grad():
    pooled = predictor.predict(X_pool)
kappa = pooled.uncertainty.numpy()
errors = (pooled.decision.probabilities.argmax(dim=-1).numpy() != y_pool).astype(float)
cal, test = np.arange(5000), np.arange(5000, len(kappa))

target_risks = [0.02, 0.025, 0.03, 0.04, 0.05]
points = []
print(f"{'target':>7} {'test risk':>9} {'test cov':>8}")
for r in target_risks:
    rule = selector(risk=r, delta=delta).calibrate(kappa[cal], errors[cal])
    mask = rule.select(kappa[test])
    points.append((mask.mean(), errors[test][mask].mean()))
    print(f"{r:7.3f} {points[-1][1]:9.3f} {points[-1][0]:8.3f}")

curve_coverage, curve_risk, _ = risk_coverage_curve(kappa[test], errors[test])

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
