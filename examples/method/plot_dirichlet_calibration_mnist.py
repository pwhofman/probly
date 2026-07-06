"""===============================
Dirichlet Calibration on MNIST
===============================

Dirichlet calibration fits a multinomial logistic regression on the
log-probabilities of a classifier, ``q = softmax(W @ ln(p) + b)``.  On a
ten-class problem the full ``10 x 10`` weight matrix ``W`` has enough capacity to
correct class-specific miscalibration, while Off-Diagonal and Intercept
Regularisation (ODIR) keeps it from overfitting the calibration split.  This
example trains a small MLP on MNIST, fits Dirichlet calibration on a held-out
split, compares the negative log-likelihood and expected calibration error before
and after calibration, and visualises the learned weight matrix as a heatmap.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from probly.calibrator import calibrate
from probly.method.dirichlet_calibration import dirichlet_calibration
from probly.predictor import predict_raw
from probly_benchmark.data import load_mnist

from examples.utils.model import ResFFN
from examples.utils.plotting import plot_mnist_uncertainty

NUM_CLASSES = 10
RELIABILITY_BINS = 15

# %%
# Setup
# -----
#
# Use the training set to fit the network, the first half of the test set as the
# calibration split, and the second half as the evaluation set.

train_loader, test_loader = load_mnist(batch_size=256)

X_train_batches, y_train_batches = zip(*train_loader)
X_train = torch.cat([x.view(-1, 28 * 28) for x in X_train_batches])
y_train = torch.cat(list(y_train_batches))

X_test_batches, y_test_batches = zip(*test_loader)
X_all = torch.cat([x.view(-1, 28 * 28) for x in X_test_batches])
y_all = torch.cat(list(y_test_batches))

half = len(X_all) // 2
X_calib, y_calib = X_all[:half], y_all[:half]
X_test, y_test = X_all[half:], y_all[half:]

# %%
# Model
# -----

torch.manual_seed(0)
model = ResFFN(in_features=28 * 28, hidden_features=256, out_features=NUM_CLASSES)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

model.train()
for _epoch in range(5):
    correct, total = 0, 0
    for X_batch, y_batch in train_loader:
        X_flat = X_batch.view(-1, 28 * 28)
        opt.zero_grad()
        logits = model(X_flat)
        loss = criterion(logits, y_batch)
        loss.backward()
        opt.step()
        correct += (logits.detach().argmax(-1) == y_batch).sum().item()
        total += len(y_batch)
    if correct / total >= 0.97:
        break
model.eval()

# %%
# Calibrate
# ---------

calibrated_model = dirichlet_calibration(
    model, num_classes=NUM_CLASSES, predictor_type="logit_classifier"
)
calibrate(calibrated_model, y_calib, X_calib)

# %%
# Evaluation
# ----------


def _probs(logits: torch.Tensor) -> np.ndarray:
    return logits.softmax(-1).detach().numpy()


def _nll(probs: np.ndarray, labels: np.ndarray) -> float:
    clipped = np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)
    return float(-np.mean(np.log(clipped)))


def _brier(probs: np.ndarray, labels: np.ndarray) -> float:
    one_hot = np.eye(probs.shape[-1])[labels]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=-1)))


def _ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = RELIABILITY_BINS) -> float:
    confidence = probs.max(-1)
    correct = (probs.argmax(-1) == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        mask = (confidence > edges[b]) & (confidence <= edges[b + 1])
        if mask.any():
            ece += abs(correct[mask].mean() - confidence[mask].mean()) * mask.mean()
    return float(ece)


labels_test = y_test.numpy()
with torch.no_grad():
    uncal_probs = _probs(model(X_test))
    cal_probs = _probs(predict_raw(calibrated_model, X_test))

accuracy = (cal_probs.argmax(-1) == labels_test).mean() * 100
print(f"Test accuracy:         {accuracy:.1f}%")
print(f"Uncalibrated:  NLL={_nll(uncal_probs, labels_test):.4f}  Brier={_brier(uncal_probs, labels_test):.4f}  ECE={_ece(uncal_probs, labels_test):.4f}")
print(f"Dirichlet:     NLL={_nll(cal_probs, labels_test):.4f}  Brier={_brier(cal_probs, labels_test):.4f}  ECE={_ece(cal_probs, labels_test):.4f}")

# %%
# Learned Weight Matrix
# ---------------------
#
# The diagonal dominates (each class mostly maps to itself); the off-diagonal
# entries, shrunk by ODIR, capture cross-class corrections.

weight = calibrated_model.weight.numpy()

fig, ax = plt.subplots(figsize=(5.5, 4.5))
image = ax.imshow(weight, cmap="RdBu_r", vmin=-np.abs(weight).max(), vmax=np.abs(weight).max())
ax.set_xlabel("Input class (ln p)")
ax.set_ylabel("Output class")
ax.set_title("Dirichlet Calibration Weight Matrix W")
ax.set_xticks(range(NUM_CLASSES))
ax.set_yticks(range(NUM_CLASSES))
fig.colorbar(image, ax=ax)
fig.tight_layout()

plt.show()

# %%
# Most Uncertain Calibrated Predictions
# -------------------------------------

images_test = (X_test.view(-1, 28, 28) * 255).byte()
entropy_bits = -(cal_probs * np.log2(np.clip(cal_probs, 1e-12, 1.0))).sum(-1)

plot = plot_mnist_uncertainty(
    images_test,
    y_test,
    entropy_bits,
    cal_probs,
    title="Top-5 Most Uncertain Calibrated Predictions (Dirichlet)",
)
plot.show()
