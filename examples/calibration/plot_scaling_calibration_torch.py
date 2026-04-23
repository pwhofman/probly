"""==================================
Scaling Calibration Showcase - torch
==================================

Compare scaling-based calibration methods on a binary PyTorch classifier.

We train a small MLP with an explicit logit sharpening factor, which makes it
highly overconfident on noisy labels. We then calibrate the model outputs with
temperature, Platt, and vector scaling and compare reliability curves and
probability histograms against the uncalibrated baseline.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import torch
from torch import nn

from probly.calibrator import calibrate
from probly.method.calibration import (
    isotonic_regression,
    platt_scaling,
    temperature_scaling,
    torch_identity_logit_model,
    vector_scaling,
)
from probly.predictor import LogitClassifier, predict_raw

RANDOM_SEED = 13
RELIABILITY_BINS = 10
HIST_BINS = 10
LOGIT_SHARPENING = 8.0
TRAIN_EPOCHS = 320


def _binary_metrics(y_true: np.ndarray, probs: np.ndarray, n_bins: int) -> tuple[float, float, float]:
    probs_clipped = np.clip(np.asarray(probs, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    y = np.asarray(y_true, dtype=np.float64)
    nll = float(-np.mean(y * np.log(probs_clipped) + (1.0 - y) * np.log(1.0 - probs_clipped)))
    brier = float(np.mean((probs - y) ** 2))

    _, mean_pred, frac_pos, counts = _uniform_bin_stats(y, probs_clipped, n_bins)
    valid = counts > 0
    ece = float(np.sum(np.abs(mean_pred[valid] - frac_pos[valid]) * (counts[valid] / np.sum(counts[valid]))))

    return nll, brier, ece


def _uniform_bin_stats(y_true: np.ndarray, probs: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p = np.clip(np.asarray(probs, dtype=np.float64).reshape(-1), 1e-7, 1.0 - 1e-7)

    if y.shape[0] != p.shape[0]:
        msg = f"Mismatched y/probability lengths: {y.shape[0]} vs {p.shape[0]}"
        raise ValueError(msg)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    indices = np.digitize(p, edges[1:-1], right=True)

    mean_pred = np.full(n_bins, np.nan, dtype=np.float64)
    frac_pos = np.full(n_bins, np.nan, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)
    for idx in range(n_bins):
        mask = indices == idx
        counts[idx] = int(np.sum(mask))
        if counts[idx] > 0:
            mean_pred[idx] = float(np.mean(p[mask]))
            frac_pos[idx] = float(np.mean(y[mask]))

    return centers, mean_pred, frac_pos, counts


def _plot_method_panel(
    ax_curve: plt.Axes,
    ax_hist: plt.Axes,
    title: str,
    y_true: np.ndarray,
    uncalibrated_probs: np.ndarray,
    calibrated_probs: np.ndarray,
    uncalibrated_ece: float,
    calibrated_ece: float,
    reliability_bins: int,
    hist_bins: int,
) -> None:
    centers, _, frac_uncal, _ = _uniform_bin_stats(y_true, uncalibrated_probs, reliability_bins)
    _, _, frac_cal, _ = _uniform_bin_stats(y_true, calibrated_probs, reliability_bins)
    valid_uncal = np.isfinite(frac_uncal)
    valid_cal = np.isfinite(frac_cal)

    ax_curve.plot([0.0, 1.0], [0.0, 1.0], "--", color="0.5", label="Perfect calibration")
    ax_curve.vlines(
        centers[valid_uncal],
        np.minimum(centers[valid_uncal], frac_uncal[valid_uncal]),
        np.maximum(centers[valid_uncal], frac_uncal[valid_uncal]),
        color="tab:red",
        alpha=0.22,
    )
    ax_curve.vlines(
        centers[valid_cal],
        np.minimum(centers[valid_cal], frac_cal[valid_cal]),
        np.maximum(centers[valid_cal], frac_cal[valid_cal]),
        color="tab:blue",
        alpha=0.22,
    )
    ax_curve.plot(
        centers[valid_uncal],
        frac_uncal[valid_uncal],
        "o-",
        color="tab:red",
        lw=2,
        ms=6,
        label=f"Uncalibrated (ECE={uncalibrated_ece:.3f})",
    )
    ax_curve.plot(
        centers[valid_cal],
        frac_cal[valid_cal],
        "o-",
        color="tab:blue",
        lw=2,
        ms=6,
        label=f"Calibrated (ECE={calibrated_ece:.3f})",
    )
    ax_curve.set_title(title)
    ax_curve.set_xlim(0.0, 1.0)
    ax_curve.set_ylim(0.0, 1.0)
    ax_curve.grid(alpha=0.25)

    bins = np.linspace(0.0, 1.0, hist_bins + 1)
    ax_hist.hist(uncalibrated_probs, bins=bins, alpha=0.55, color="tab:red", label="Uncalibrated")
    ax_hist.hist(calibrated_probs, bins=bins, alpha=0.55, color="tab:blue", label="Calibrated")
    ax_hist.set_xlim(0.0, 1.0)
    ax_hist.grid(alpha=0.25)
    ax_hist.set_xlabel("Predicted positive probability")


def _to_two_class_logits(binary_logits: torch.Tensor) -> torch.Tensor:
    logits_1d = binary_logits.reshape(-1)
    return torch.stack([torch.zeros_like(logits_1d), logits_1d], dim=-1)


class OverconfidentBinaryMLP(nn.Module, LogitClassifier):
    """Small MLP with fixed output sharpening for overconfidence."""

    def __init__(self, in_features: int, sharpening: float) -> None:
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.output = nn.Linear(64, 1)
        self.sharpening = sharpening

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.output(self.hidden(x)).squeeze(-1)
        return self.sharpening * logits


# %%
# Build a noisy binary dataset and reserve a comparatively small train split.
#
# This setup intentionally encourages overconfident mistakes on the held-out
# set so that post-hoc calibration has a clear visual effect.

X, y = make_classification(
    n_samples=4000,
    n_features=14,
    n_informative=10,
    n_redundant=2,
    class_sep=1.1,
    flip_y=0.18,
    random_state=RANDOM_SEED,
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, stratify=y, random_state=RANDOM_SEED)
X_train, X_calib, y_train, y_calib = train_test_split(
    X_train,
    y_train,
    test_size=0.4,
    stratify=y_train,
    random_state=RANDOM_SEED,
)

train_mean = X_train.mean(axis=0, keepdims=True)
train_std = X_train.std(axis=0, keepdims=True) + 1e-6
X_train = (X_train - train_mean) / train_std
X_calib = (X_calib - train_mean) / train_std
X_test = (X_test - train_mean) / train_std

X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_calib_t = torch.tensor(X_calib, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)

y_train_t = torch.tensor(y_train, dtype=torch.float32)
y_calib_t = torch.tensor(y_calib, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

torch.manual_seed(RANDOM_SEED)

# %%
# Train an intentionally overconfident MLP.
#
# A fixed logit sharpening factor makes the trained predictor overconfident.

model = OverconfidentBinaryMLP(in_features=X_train.shape[1], sharpening=LOGIT_SHARPENING)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCEWithLogitsLoss()

model.train()
for _ in range(TRAIN_EPOCHS):
    optimizer.zero_grad()
    train_logits = model(X_train_t)
    loss = loss_fn(train_logits, y_train_t)
    loss.backward()
    optimizer.step()
model.eval()

with torch.no_grad():
    calib_logits = model(X_calib_t)
    test_logits = model(X_test_t)

uncalibrated_probs = torch.sigmoid(test_logits).cpu().numpy().reshape(-1)

# %%
# Calibrate with temperature, Platt, vector scaling, and isotonic regression.

temperature_model = temperature_scaling(torch_identity_logit_model())
platt_model = platt_scaling(torch_identity_logit_model())
vector_model = vector_scaling(torch_identity_logit_model(), num_classes=2)
isotonic_model = isotonic_regression(torch_identity_logit_model())

calibrate(temperature_model, y_calib_t, calib_logits)
calibrate(platt_model, y_calib_t, calib_logits)
calibrate(vector_model, y_calib_t.long(), _to_two_class_logits(calib_logits))
calibrate(isotonic_model, y_calib_t, calib_logits)

with torch.no_grad():
    temp_probs = torch.sigmoid(predict_raw(temperature_model, test_logits)).cpu().numpy().reshape(-1)
    platt_probs = torch.sigmoid(predict_raw(platt_model, test_logits)).cpu().numpy().reshape(-1)
    vector_logits_test = predict_raw(vector_model, _to_two_class_logits(test_logits))
    vector_probs = torch.softmax(vector_logits_test, dim=-1)[:, 1].cpu().numpy().reshape(-1)
    isotonic_probs = predict_raw(isotonic_model, test_logits).cpu().numpy().reshape(-1)

method_probs: dict[str, np.ndarray] = {
    "Temperature scaling": temp_probs,
    "Platt scaling": platt_probs,
    "Vector scaling": vector_probs,
    "Isotonic regression": isotonic_probs,
}

# %%
# Print quantitative metrics before plotting.

y_test_np = y_test_t.cpu().numpy().astype(int)
base_nll, base_brier, base_ece = _binary_metrics(y_test_np, uncalibrated_probs, RELIABILITY_BINS)
print("Model                  NLL    Brier    ECE")
print("---------------------------------------------")
print(f"Uncalibrated       {base_nll:7.4f} {base_brier:8.4f} {base_ece:7.4f}")
method_metrics: dict[str, tuple[np.ndarray, float, float, float]] = {}
for name, probs in method_probs.items():
    nll, brier, ece = _binary_metrics(y_test_np, probs, RELIABILITY_BINS)
    method_metrics[name] = (probs, nll, brier, ece)
    print(f"{name:<18} {nll:7.4f} {brier:8.4f} {ece:7.4f}")

# %%
# Reliability curves and per-method probability histograms.

fig, axes = plt.subplots(
    2,
    4,
    figsize=(18, 7),
    sharex="col",
    gridspec_kw={"height_ratios": [3, 2]},
)

for idx, (method_name, (probs_after, _, _, ece_after)) in enumerate(method_metrics.items()):
    _plot_method_panel(
        ax_curve=axes[0, idx],
        ax_hist=axes[1, idx],
        title=method_name,
        y_true=y_test_np,
        uncalibrated_probs=uncalibrated_probs,
        calibrated_probs=probs_after,
        uncalibrated_ece=base_ece,
        calibrated_ece=ece_after,
        reliability_bins=RELIABILITY_BINS,
        hist_bins=HIST_BINS,
    )

axes[0, 0].set_ylabel("Fraction of positives")
axes[1, 0].set_ylabel("Count")
axes[0, 0].legend(loc="upper left", fontsize=9)
axes[1, 0].legend(loc="upper center", fontsize=9)

fig.suptitle("Binary calibration: uncalibrated vs scaling methods (torch)", y=1.02)
plt.tight_layout()
plt.show()
