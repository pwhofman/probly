"""=========================================
Selective Prediction at a Target Coverage
=========================================

Choose the threshold on unlabeled calibration data so that a target fraction of predictions is accepted.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
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
X_cal = torch.from_numpy(X[500:1000]).float()
X_test, y_test = torch.from_numpy(X[1000:]).float(), y[1000:]

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
# The threshold is the ``ceil((n + 1) c)``-th smallest uncertainty on the calibration inputs, no labels needed. For
# exchangeable data, a new instance is then accepted with probability at least ``c``. The dropout masks are drawn per
# instance, so the uncertainties of calibration and test instances are exchangeable as well.

target = 0.9
predictor = SelectivePredictor(model, CoverageSelector(target), representer_kwargs={"num_samples": 100})
with torch.no_grad():
    predictor.calibrate(X_cal)
    result = predictor.predict(X_test)

accepted = result.accepted.numpy()
correct = result.decision.probabilities.argmax(dim=-1).numpy() == y_test
print(f"target coverage {target:.2f}  threshold {predictor.selector.threshold:.3f}")
print(f"realized coverage {result.coverage:.3f}  accuracy {correct[accepted].mean():.3f}  (all: {correct.mean():.3f})")

# %%
# The guarantee on average
# ------------------------
#
# The guarantee is marginal: for a single calibration set, the realized coverage varies around the target. The
# boxes show it over 200 random splits of the pooled uncertainties into 500 calibration and 1000 test instances.

with torch.no_grad():
    kappa = predictor.predict(torch.from_numpy(X[500:]).float()).uncertainty.numpy()

targets = [0.5, 0.7, 0.8, 0.9, 0.95]
rng = np.random.default_rng(0)
realized = {c: [] for c in targets}
for _ in range(200):
    order = rng.permutation(len(kappa))
    cal, test = kappa[order[:500]], kappa[order[500:]]
    for c in targets:
        realized[c].append(CoverageSelector(c).calibrate(cal).select(test).mean())

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot([0.45, 1.0], [0.45, 1.0], color="black", linestyle="--", label="target")
bp = ax.boxplot(
    [realized[c] for c in targets],
    positions=targets,
    widths=0.03,
    patch_artist=True,
    manage_ticks=False,
    medianprops={"color": RED},
)
for box in bp["boxes"]:
    box.set(facecolor=BLUE, alpha=0.5)
ax.set_xlabel("target coverage")
ax.set_ylabel("realized coverage")
ax.legend(loc="upper left")
plt.tight_layout()
plt.show()
