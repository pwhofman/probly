"""=========================================
Selective Prediction at a Target Coverage
=========================================

Choose the threshold on unlabeled calibration data so that a target fraction of predictions is accepted.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import betabinom
from sklearn.datasets import make_moons
import torch
from torch import nn

from probly.selective_prediction import CoverageSelector, SelectivePredictor
from probly.transformation import dropout

from examples.utils.model import MLPClassifier

BLUE, RED = "#1e88e5", "#ff0d57"

# %%
# Data and model
# --------------

X, y = make_moons(n_samples=2000, noise=0.25, random_state=0)
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
# The threshold is the ``ceil((n + 1) c)``-th smallest uncertainty on the calibration inputs, so the labels are
# ``None``. For exchangeable data, a new instance is then accepted with probability at least ``c``. The dropout masks
# are drawn per instance, so the uncertainties of calibration and test instances are exchangeable as well.

target = 0.9
X_cal, X_test, y_test = X_pool[:500], X_pool[500:], y_pool[500:]

predictor = SelectivePredictor(model, CoverageSelector(target), representer_kwargs={"num_samples": 100})
with torch.no_grad():
    predictor.calibrate(None, X_cal)
    result = predictor.predict(X_test)

accepted = result.accepted.numpy()
wrong = result.decision.probabilities.argmax(dim=-1).numpy() != y_test
print(f"target coverage {target:.2f}  threshold {predictor.selector.threshold:.3f}")
print(f"realized coverage {result.coverage:.3f}  selective risk {wrong[accepted].mean():.3f}  (all: {wrong.mean():.3f})")

# %%
# Coverage and risk for several targets
# -------------------------------------
#
# The table follows SelectiveNet's Tables 1 and 2 (Geifman and El-Yaniv, 2019): the realized test coverage (mean and
# standard deviation over 1000 random splits of the pool into 500 calibration and 1000 test instances), the selective
# risk at it, and the average violation of the target. The figure follows its Fig. 2: the test risk-coverage curve of
# the criterion on the first split, with markers at the calibrated coverages.

with torch.no_grad():
    pooled = predictor.predict(X_pool)
kappa = pooled.uncertainty.numpy()
errors = (pooled.decision.probabilities.argmax(dim=-1).numpy() != y_pool).astype(float)

targets = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
splits = [np.arange(len(kappa))] + [np.random.default_rng(i).permutation(len(kappa)) for i in range(999)]
coverage = np.zeros((len(splits), len(targets)))
risk = np.zeros((len(splits), len(targets)))
for i, order in enumerate(splits):
    cal, test = order[:500], order[500:]
    for j, c in enumerate(targets):
        mask = CoverageSelector(c).calibrate(kappa[cal]).select(kappa[test])
        coverage[i, j], risk[i, j] = mask.mean(), errors[test][mask].mean()

print(f"{'target':>6} {'coverage':>15} {'risk':>6}")
for j, c in enumerate(targets):
    print(f"{c:6.2f} {coverage[:, j].mean():7.3f} +- {coverage[:, j].std():.3f} {risk[:, j].mean():6.3f}")
print(f"average violation {np.mean(np.abs(coverage.mean(axis=0) - targets)):.4f}")

test = splits[0][500:]
order = np.argsort(kappa[test], kind="stable")
curve_risk = np.cumsum(errors[test][order]) / np.arange(1, len(order) + 1)
curve_coverage = np.arange(1, len(order) + 1) / len(order)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(curve_coverage, curve_risk, color=BLUE, label="test risk-coverage curve")
ax.scatter(coverage[0], risk[0], color=RED, zorder=3, label="calibrated coverages")
ax.set_xlim(0.65, 1.01)
ax.set_ylim(0, 1.2 * errors[test].mean())
ax.set_xlabel("Coverage")
ax.set_ylabel("Selective risk")
ax.legend(loc="upper left")
plt.tight_layout()
plt.show()

# %%
# The guarantee over calibration sets
# -----------------------------------
#
# The guarantee is marginal: for a single calibration set, the realized coverage varies around the target. Over random
# splits it approximately follows ``BetaBinomial(n_test, k, n + 1 - k) / n_test`` with ``k = ceil((n + 1) c)``
# (Angelopoulos and Bates, 2021, Section 3.3). The histogram shows the realized coverage at the target 0.9, with this
# law overlaid.

n, n_test = 500, len(kappa) - 500
k = math.ceil((n + 1) * target)
support = np.arange(n_test + 1)
law = betabinom(n_test, k, n + 1 - k)

fig, ax = plt.subplots(figsize=(6, 5))
ax.hist(coverage[:, targets.index(target)], bins=15, density=True, color=BLUE, alpha=0.5, label="realized coverage")
ax.plot(support / n_test, law.pmf(support) * n_test, color=RED, label="BetaBinomial law")
ax.axvline(target, color="black", linestyle="--", label="target")
ax.set_xlim(law.ppf(0.0005) / n_test, law.ppf(0.9995) / n_test)
ax.set_xlabel("Coverage")
ax.set_ylabel("Density")
ax.legend(loc="upper left")
plt.tight_layout()
plt.show()
