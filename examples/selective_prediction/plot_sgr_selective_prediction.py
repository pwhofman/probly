"""=====================================
Selective Prediction with Risk Control
=====================================

Choose the threshold on labeled calibration data so that the risk of the accepted predictions stays below a target.
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

X, y = make_moons(n_samples=3000, noise=0.25, random_state=0)
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
# SGR needs the labels of the calibration instances, passed as ``targets``. It returns the largest threshold it can
# certify, such that with probability at least ``1 - delta`` over the calibration set, the risk of the accepted
# predictions is at most ``risk``. The guarantee says nothing about the coverage.

risk, delta = 0.05, 0.1
X_cal, y_cal = X_pool[:1000], y_pool[:1000]
X_test, y_test = X_pool[1000:], y_pool[1000:]

predictor = SelectivePredictor(model, SGRSelector(risk, delta), representer_kwargs={"num_samples": 100})
with torch.no_grad():
    predictor.calibrate(X_cal, targets=torch.from_numpy(y_cal))
    result = predictor.predict(X_test)

accepted = result.accepted.numpy()
wrong = result.decision.probabilities.argmax(dim=-1).numpy() != y_test
print(f"target risk {risk:.2f}  delta {delta:.2f}  threshold {predictor.selector.threshold:.3f}")
print(f"bound {predictor.selector.bound:.3f}  realized risk {wrong[accepted].mean():.3f}  coverage {result.coverage:.3f}")

# %%
# The guarantee over calibration sets
# -----------------------------------
#
# The boxes show the realized risk over 200 random splits of the pooled instances into 1000 calibration and 1500 test
# instances. The risk exceeds the target in at most a fraction ``delta`` of the splits.

with torch.no_grad():
    pooled = predictor.predict(X_pool)
kappa = pooled.uncertainty.numpy()
errors = (pooled.decision.probabilities.argmax(dim=-1).numpy() != y_pool).astype(float)

risks = [0.03, 0.05, 0.07, 0.09]
rng = np.random.default_rng(0)
realized = {r: [] for r in risks}
for _ in range(200):
    order = rng.permutation(len(kappa))
    cal, test = order[:1000], order[1000:]
    for r in risks:
        selector = SGRSelector(r, delta)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            threshold = selector.calibrate(kappa[cal], errors[cal]).threshold
        mask = kappa[test] <= threshold
        if mask.any():
            realized[r].append(errors[test][mask].mean())

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot([0.02, 0.1], [0.02, 0.1], color="black", linestyle="--", label="target risk")
bp = ax.boxplot(
    [realized[r] for r in risks],
    positions=risks,
    widths=0.005,
    patch_artist=True,
    manage_ticks=False,
    medianprops={"color": RED},
)
for box in bp["boxes"]:
    box.set(facecolor=BLUE, alpha=0.5)
for r in risks:
    print(f"target risk {r:.2f}  share of splits above it {np.mean(np.array(realized[r]) > r):.3f}")
ax.set_xlabel("target risk")
ax.set_ylabel("realized risk of accepted predictions")
ax.legend(loc="upper left")
plt.tight_layout()
plt.show()
