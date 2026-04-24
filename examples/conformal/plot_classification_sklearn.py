"""==============================================
Classification Conformal Prediction — sklearn
==============================================

Demonstrate all four classification non-conformity scores
(:func:`~probly.conformal.scores.lac_score`,
:class:`~probly.conformal.scores.APSScore`,
:class:`~probly.conformal.scores.RAPSScore`,
:class:`~probly.conformal.scores.SAPSScore`)
using a :class:`~sklearn.tree.DecisionTreeClassifier` on the Iris dataset.

The workflow is the same for every score:

1. Fit a base model.
2. Create a score-specific conformal wrapper.
3. Calibrate with :func:`~probly.calibrator.calibrate`.
4. Call :func:`~probly.representer.representer` to obtain typed conformal sets.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split

from probly.calibrator import calibrate
from probly.metrics._common import average_set_size, empirical_coverage_classification
from probly.method.conformal import conformal_aps, conformal_lac, conformal_raps, conformal_saps
from probly.representer import representer

# %%
# Data preparation
# ----------------
# Load the Digits dataset and split into 60 % train / 20 % calibration / 20 % test.
ALPHA = 0.05
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_calib, y_train, y_calib = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# %%
# Build and train the model
# -------------------------

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# %%
# LAC score
# ---------
# The Least Ambiguous set-valued Classifier score: ``1 - P(y | x)``.

calibrated_model = calibrate(conformal_lac(model), ALPHA, y_calib, X_calib)
output = representer(calibrated_model).predict(X_test)
lac_cov = empirical_coverage_classification(output, y_test)
lac_size = average_set_size(output)
print(f"LAC  — coverage: {lac_cov:.3f}, avg set size: {lac_size:.3f}")

# %%
# APS score
# ---------
# Adaptive Prediction Sets: cumulative sorted probabilities with randomisation.

calibrated_model = calibrate(conformal_aps(model, randomized=True), ALPHA, y_calib, X_calib)
output = representer(calibrated_model).predict(X_test)
aps_cov = empirical_coverage_classification(output, y_test)
aps_size = average_set_size(output)
print(f"APS  — coverage: {aps_cov:.3f}, avg set size: {aps_size:.3f}")

# %%
# RAPS score
# ----------
# Regularised APS: adds a size penalty to encourage smaller prediction sets.

calibrated_model = calibrate(conformal_raps(model, randomized=True, lambda_reg=0.1, k_reg=0), ALPHA, y_calib, X_calib)
output = representer(calibrated_model).predict(X_test)
raps_cov = empirical_coverage_classification(output, y_test)
raps_size = average_set_size(output)
print(f"RAPS — coverage: {raps_cov:.3f}, avg set size: {raps_size:.3f}")

# %%
# SAPS score
# ----------
# Sorted APS: penalises gaps between consecutive sorted probabilities.

calibrated_model = calibrate(conformal_saps(model, randomized=True, lambda_val=0.1), ALPHA, y_calib, X_calib)
output = representer(calibrated_model).predict(X_test)
saps_cov = empirical_coverage_classification(output, y_test)
saps_size = average_set_size(output)
print(f"SAPS — coverage: {saps_cov:.3f}, avg set size: {saps_size:.3f}")

# %%
# Summary (Averaged over multiple runs)
# --------------------------------------
res = {
    "LAC": [],
    "APS": [],
    "RAPS": [],
    "SAPS": [],
}
for fold, (train_idx, test_idx) in enumerate(KFold(n_splits=5, shuffle=True, random_state=42).split(X)):
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    X_train, X_calib, y_train, y_calib = train_test_split(X_train, y_train, test_size=0.25, random_state=fold)

    fold_model = RandomForestClassifier(random_state=fold)
    fold_model.fit(X_train, y_train)
    for name, conformal_func in [("LAC", conformal_lac), ("APS", conformal_aps), ("RAPS", conformal_raps), ("SAPS", conformal_saps)]:
        calibrated_model = calibrate(conformal_func(fold_model), ALPHA, y_calib, X_calib)
        output = representer(calibrated_model).predict(X_test)
        cov = empirical_coverage_classification(output, y_test)
        size = average_set_size(output)
        res[name].append((cov, size))

for name, vals in res.items():
    covs, sizes = zip(*vals)
    print(f"{name} — coverage: {np.mean(covs):.3f} ± {np.std(covs):.3f}, avg set size: {np.mean(sizes):.3f} ± {np.std(sizes):.3f}")
