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

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from probly.calibrator import calibrate
from probly.metrics._common import average_set_size, empirical_coverage_classification
from probly.method.conformal import conformal_aps, conformal_lac, conformal_raps, conformal_saps
from probly.representer import representer

# %%
# Data preparation
# ----------------
# Load the Iris dataset and split into 60 % train / 20 % calibration / 20 % test.

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_calib, y_train, y_calib = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# %%
# Build and train the model
# -------------------------

model = DecisionTreeClassifier(max_depth=2, random_state=42)
model.fit(X_train, y_train)

# %%
# LAC score
# ---------
# The Least Ambiguous set-valued Classifier score: ``1 - P(y | x)``.

calibrated_model = calibrate(conformal_lac(model), 0.05, y_calib, X_calib)
output = representer(calibrated_model).predict(X_test)
lac_cov = empirical_coverage_classification(output, y_test)
lac_size = average_set_size(output)
print(f"LAC  — coverage: {lac_cov:.3f}, avg set size: {lac_size:.3f}")

# %%
# APS score
# ---------
# Adaptive Prediction Sets: cumulative sorted probabilities with randomisation.

calibrated_model = calibrate(conformal_aps(model), 0.05, y_calib, X_calib)
output = representer(calibrated_model).predict(X_test)
aps_cov = empirical_coverage_classification(output, y_test)
aps_size = average_set_size(output)
print(f"APS  — coverage: {aps_cov:.3f}, avg set size: {aps_size:.3f}")

# %%
# RAPS score
# ----------
# Regularised APS: adds a size penalty to encourage smaller prediction sets.

calibrated_model = calibrate(conformal_raps(model), 0.05, y_calib, X_calib)
output = representer(calibrated_model).predict(X_test)
raps_cov = empirical_coverage_classification(output, y_test)
raps_size = average_set_size(output)
print(f"RAPS — coverage: {raps_cov:.3f}, avg set size: {raps_size:.3f}")

# %%
# SAPS score
# ----------
# Sorted APS: penalises gaps between consecutive sorted probabilities.

calibrated_model = calibrate(conformal_saps(model), 0.05, y_calib, X_calib)
output = representer(calibrated_model).predict(X_test)
saps_cov = empirical_coverage_classification(output, y_test)
saps_size = average_set_size(output)
print(f"SAPS — coverage: {saps_cov:.3f}, avg set size: {saps_size:.3f}")

# %%
# Summary
# -------

print("\n{:<6} {:>10} {:>14}".format("Score", "Coverage", "Avg set size"))
print("-" * 32)
for name, cov, sz in [
    ("LAC", lac_cov, lac_size),
    ("APS", aps_cov, aps_size),
    ("RAPS", raps_cov, raps_size),
    ("SAPS", saps_cov, saps_size),
]:
    print(f"{name:<6} {cov:>10.3f} {sz:>14.3f}")
