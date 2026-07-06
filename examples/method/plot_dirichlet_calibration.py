"""===================================
Dirichlet Calibration on Two Moons
===================================

Dirichlet calibration recalibrates a classifier by fitting a multinomial logistic
regression on the log-probabilities, ``q = softmax(W @ ln(p) + b)``, with a full
weight matrix ``W`` and bias ``b``.  It generalises temperature and vector scaling
and is regularised with Off-Diagonal and Intercept Regularisation (ODIR).  This
example over-trains a small MLP on the two-moons dataset until it is overconfident,
then fits Dirichlet calibration on a held-out split and compares the reliability of
the calibrated against the uncalibrated probabilities.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import torch
from torch import nn

from probly.calibrator import calibrate
from probly.method import cast
from probly.method.dirichlet_calibration import dirichlet_calibration
from probly.predictor import predict_raw
from probly.representer import representer

from examples.utils.model import MLPClassifier
from examples.utils.plotting import plot_example_uncertainty

NUM_CLASSES = 2
RELIABILITY_BINS = 10

# %%
# Setup
# -----
#
# Split the data into a training set, a held-out calibration set (used only to fit
# the calibrator) and a test set used for evaluation.

X, y = make_moons(n_samples=2000, noise=0.2, random_state=0)
X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.5, random_state=0)
X_calib, X_test, y_calib, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=0)


def to_x(array: np.ndarray) -> torch.Tensor:
    """Convert a feature array to a float tensor."""
    return torch.from_numpy(array).float()


def to_y(array: np.ndarray) -> torch.Tensor:
    """Convert a label array to a long tensor."""
    return torch.from_numpy(array).long()

# %%
# Model
# -----
#
# Train the MLP to convergence with a sharpening factor on the logits, which makes
# the softmax probabilities overconfident -- the regime where calibration helps.

torch.manual_seed(0)
model = MLPClassifier(in_features=2, hidden_features=64, out_features=NUM_CLASSES)
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
criterion = nn.CrossEntropyLoss()

model.train()
for _epoch in range(400):
    opt.zero_grad()
    logits = model(to_x(X_train))
    loss = criterion(logits, to_y(y_train))
    loss.backward()
    opt.step()
model.eval()

# %%
# Calibrate
# ---------
#
# Wrap the trained model with Dirichlet calibration and fit the map on the
# calibration split.  ``calibrate`` runs L-BFGS under the hood.

calibrated_model = dirichlet_calibration(
    model, num_classes=NUM_CLASSES, predictor_type="logit_classifier"
)
calibrate(calibrated_model, to_y(y_calib), to_x(X_calib))

# %%
# Evaluation
# ----------
#
# Compare negative log-likelihood (NLL) and expected calibration error (ECE) of the
# uncalibrated and Dirichlet-calibrated probabilities on the test set.


def _probs(logits: torch.Tensor) -> np.ndarray:
    return logits.softmax(-1).detach().numpy()


def _nll(probs: np.ndarray, labels: np.ndarray) -> float:
    clipped = np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)
    return float(-np.mean(np.log(clipped)))


def _brier(probs: np.ndarray, labels: np.ndarray) -> float:
    one_hot = np.eye(probs.shape[-1])[labels]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=-1)))


def _ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = RELIABILITY_BINS) -> tuple[float, np.ndarray, np.ndarray]:
    confidence = probs.max(-1)
    predictions = probs.argmax(-1)
    correct = (predictions == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, bin_conf, bin_acc = 0.0, np.full(n_bins, np.nan), np.full(n_bins, np.nan)
    for b in range(n_bins):
        mask = (confidence > edges[b]) & (confidence <= edges[b + 1])
        if mask.any():
            bin_conf[b] = confidence[mask].mean()
            bin_acc[b] = correct[mask].mean()
            ece += abs(bin_acc[b] - bin_conf[b]) * mask.mean()
    return float(ece), bin_conf, bin_acc


with torch.no_grad():
    uncal_probs = _probs(model(to_x(X_test)))
    cal_probs = _probs(predict_raw(calibrated_model, to_x(X_test)))

uncal_ece, uncal_conf, uncal_acc = _ece(uncal_probs, y_test)
cal_ece, cal_conf, cal_acc = _ece(cal_probs, y_test)

print(f"Uncalibrated:  NLL={_nll(uncal_probs, y_test):.4f}  Brier={_brier(uncal_probs, y_test):.4f}  ECE={uncal_ece:.4f}")
print(f"Dirichlet:     NLL={_nll(cal_probs, y_test):.4f}  Brier={_brier(cal_probs, y_test):.4f}  ECE={cal_ece:.4f}")

# %%
# Reliability Diagram
# -------------------

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
ax.plot(uncal_conf, uncal_acc, "o-", label=f"Uncalibrated (ECE={uncal_ece:.3f})")
ax.plot(cal_conf, cal_acc, "s-", label=f"Dirichlet (ECE={cal_ece:.3f})")
ax.set_xlabel("Confidence")
ax.set_ylabel("Accuracy")
ax.set_title("Reliability Diagram - Two Moons")
ax.legend(loc="upper left")
fig.tight_layout()

plt.show()

# %%
# Predictive Uncertainty Maps
# ---------------------------
#
# Comparing the uncalibrated model against the Dirichlet-calibrated one shows how
# calibration reshapes confidence: the overconfident model is near-certain almost everywhere,
# while calibration widens the uncertain band along the decision boundary.

uncalibrated_rep = representer(cast(model, predictor_type="logit_classifier"))
calibrated_rep = representer(calibrated_model)

plot_example_uncertainty(
    X_test, y_test, uncalibrated_rep, title="Predictive Entropy - Uncalibrated", notion="total"
).show()
plot_example_uncertainty(
    X_test, y_test, calibrated_rep, title="Predictive Entropy - Dirichlet Calibrated", notion="total"
).show()
