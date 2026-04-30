r"""=============================================
Conformalized Credal Set Prediction — PyTorch.
=============================================

Conformalized credal set predictors combine conformal prediction with
imprecise probabilities.  Instead of a prediction set (a set of labels),
the method outputs a **credal set** -- a set of probability distributions
around the model's prediction.

The non-conformity score is the
:func:`Total Variation distance <probly.conformal_scores.tv_score_func>`
between predicted and true distributions.  After calibration the conformal
quantile defines the radius of a
:class:`~probly.representation.credal_set._common.DistanceBasedCredalSet`:

.. math::

    \mathcal{C}(x) = \{p : \mathrm{TV}(p, \hat{p}(x)) \leq q_{\alpha}\}

See *Javanmardi, Stutz & Hullermeier,
"Conformalized Credal Set Predictors", NeurIPS 2024*.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import torch
from torch import nn

from probly.calibrator import calibrate
from probly.method.conformal_credal_set_prediction import conformal_total_variation
from probly.predictor import predict
from probly.quantification.measure.credal_set import lower_entropy, upper_entropy

torch.manual_seed(42)

# %%
# Data preparation
# ----------------
ALPHA = 0.1
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_calib, y_train, y_calib = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42,
)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_calib_t = torch.tensor(X_calib, dtype=torch.float32)
y_calib_t = torch.tensor(y_calib, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)

# %%
# Define and train a classifier
# ------------------------------


class SimpleNet(nn.Module):
    """Two-layer softmax classifier."""

    def __init__(self, in_features: int, num_classes: int) -> None:
        """Initialize with input dimension and number of classes."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities."""
        return self.net(x).softmax(dim=-1)


model = SimpleNet(64, 10)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

model.train()
for _ in range(500):
    optimizer.zero_grad()
    loss_fn(model(X_train_t), y_train_t).backward()
    optimizer.step()
model.eval()

# %%
# Wrap with conformalized credal set prediction
# -----------------------------------------------
# :func:`~probly.method.conformal_credal_set_prediction.conformal_total_variation`
# wraps any predictor.  Calibration computes the TV-distance quantile.

ccp = conformal_total_variation(model)
calibrated = calibrate(ccp, ALPHA, y_calib_t, X_calib_t)
print(f"Conformal quantile (radius): {calibrated.conformal_quantile:.4f}")

# %%
# Predict credal sets
# --------------------
# Each prediction is a
# :class:`~probly.representation.credal_set.torch.TorchDistanceBasedCredalSet`
# with a nominal distribution and a TV-ball radius equal to the quantile.

credal_sets = predict(calibrated, X_test_t)
print(f"Type:       {type(credal_sets).__name__}")
print(f"Batch size: {credal_sets.nominal.shape[0]}")
print(f"Classes:    {credal_sets.num_classes}")
print(f"Radius:     {credal_sets.radius:.4f}")

# %%
# Inspect lower and upper probability envelopes
# -----------------------------------------------
# The TV ball implies per-class bounds:
# :math:`\\max(0,\\, p_i - r) \\leq q_i \\leq \\min(1,\\, p_i + r)`.

idx = 0
nominal = credal_sets.nominal[idx].probabilities.detach().numpy()
lower = credal_sets.lower()[idx].detach().numpy()
upper = credal_sets.upper()[idx].detach().numpy()

classes = np.arange(len(nominal))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(classes - width / 2, nominal, width, label="Nominal", color="steelblue")
ax.bar(classes + width / 2, upper - lower, width, bottom=lower,
       label="Credal interval", color="lightsalmon", edgecolor="tomato")
ax.set_xlabel("Class")
ax.set_ylabel("Probability")
ax.set_title("Conformalized credal set for a single test instance")
ax.set_xticks(classes)
ax.legend()
plt.tight_layout()
plt.show()

# %%
# Uncertainty quantification
# ---------------------------
# Upper and lower entropy bound the Shannon entropy over the credal set.

ue = upper_entropy(credal_sets).detach().numpy()
le = lower_entropy(credal_sets).detach().numpy()
print(f"Upper entropy (first 5): {ue[:5]}")
print(f"Lower entropy (first 5): {le[:5]}")

fig, ax = plt.subplots(figsize=(8, 4))
order = np.argsort(ue)
ax.fill_between(range(len(ue)), le[order], ue[order], alpha=0.3,
                color="steelblue", label="Entropy interval")
ax.plot(le[order], color="steelblue", linewidth=0.8)
ax.plot(ue[order], color="steelblue", linewidth=0.8)
ax.set_xlabel("Test instance (sorted by upper entropy)")
ax.set_ylabel("Shannon entropy (nats)")
ax.set_title("Entropy bounds from conformalized credal sets")
ax.legend()
plt.tight_layout()
plt.show()
