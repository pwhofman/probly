"""====================================
Scaling Calibration Showcase - sklearn
====================================

Compare scaling-based calibration methods on a binary task.

We sample labels from a latent binary logit model, then intentionally distort
the logits with an affine transformation so the predictor becomes strongly
miscalibrated. Temperature, Platt, and vector scaling are then calibrated on a
held-out calibration set and compared against the uncalibrated baseline.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from probly.calibrator import calibrate
from probly.method.calibration import (
    isotonic_regression,
    platt_scaling,
    sklearn_identity_logit_estimator,
    temperature_scaling,
    vector_scaling,
)
from probly.predictor import LogitClassifier

RANDOM_SEED = 7
RELIABILITY_BINS = 10
HIST_BINS = 10
DISTORTION_SCALE = 5.0
DISTORTION_SHIFT = -2.0
CALIBRATION_SAMPLES = 9000
TEST_SAMPLES = 7000


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _sample_binary_logits(seed: int, num_samples: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    true_logits = rng.normal(loc=0.0, scale=1.0, size=num_samples)
    y = rng.binomial(1, _sigmoid(true_logits), size=num_samples).astype(int)
    return true_logits, y


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


# %%
# Create a synthetic binary logit problem with known ground truth, then apply
# an affine distortion to produce a strongly miscalibrated predictor.

true_calib_logits, y_calib = _sample_binary_logits(seed=RANDOM_SEED + 100, num_samples=CALIBRATION_SAMPLES)
true_test_logits, y_test = _sample_binary_logits(seed=RANDOM_SEED + 200, num_samples=TEST_SAMPLES)

calib_logits_distorted = DISTORTION_SCALE * true_calib_logits + DISTORTION_SHIFT
test_logits_distorted = DISTORTION_SCALE * true_test_logits + DISTORTION_SHIFT

calib_logits_1c = calib_logits_distorted.reshape(-1, 1)
test_logits_1c = test_logits_distorted.reshape(-1, 1)

calib_logits_2c = np.stack([np.zeros_like(calib_logits_distorted), calib_logits_distorted], axis=-1)
test_logits_2c = np.stack([np.zeros_like(test_logits_distorted), test_logits_distorted], axis=-1)

uncalibrated_probs = _sigmoid(test_logits_distorted)

# %%
# Calibrate with temperature, Platt, and vector scaling.

temperature_model = temperature_scaling(sklearn_identity_logit_estimator())
platt_model = platt_scaling(sklearn_identity_logit_estimator())
vector_model = vector_scaling(sklearn_identity_logit_estimator(), num_classes=2)
isotonic_model = isotonic_regression(sklearn_identity_logit_estimator())

calibrate(temperature_model, y_calib, calib_logits_1c)
calibrate(platt_model, y_calib, calib_logits_1c)
calibrate(vector_model, y_calib, calib_logits_2c)
calibrate(isotonic_model, y_calib, calib_logits_1c)

method_probs: dict[str, np.ndarray] = {
    "Temperature scaling": np.asarray(temperature_model.predict_proba(test_logits_1c), dtype=float)[:, 1],
    "Platt scaling": np.asarray(platt_model.predict_proba(test_logits_1c), dtype=float)[:, 1],
    "Vector scaling": np.asarray(vector_model.predict_proba(test_logits_2c), dtype=float)[:, 1],
    "Isotonic regression": np.asarray(isotonic_model.predict_proba(test_logits_1c), dtype=float)[:, 1],
}

# %%
# Print quantitative metrics before plotting.

base_nll, base_brier, base_ece = _binary_metrics(y_test, uncalibrated_probs, RELIABILITY_BINS)
print("Model                  NLL    Brier    ECE")
print("---------------------------------------------")
print(f"Uncalibrated       {base_nll:7.4f} {base_brier:8.4f} {base_ece:7.4f}")
method_metrics: dict[str, tuple[np.ndarray, float, float, float]] = {}
for name, probs in method_probs.items():
    nll, brier, ece = _binary_metrics(y_test, probs, RELIABILITY_BINS)
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
        y_true=y_test,
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

fig.suptitle("Binary calibration: uncalibrated vs scaling methods (sklearn)", y=1.02)
plt.tight_layout()
plt.show()
