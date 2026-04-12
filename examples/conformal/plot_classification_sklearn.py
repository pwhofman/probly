"""==============================================
Classification Conformal Prediction — sklearn
==============================================

Demonstrate all four classification non-conformity scores
(:class:`~probly.conformal_prediction.scores_new.LACScore`,
:class:`~probly.conformal_prediction.scores_new.APSScore`,
:class:`~probly.conformal_prediction.scores_new.RAPSScore`,
:class:`~probly.conformal_prediction.scores_new.SAPSScore`)
using a :class:`~sklearn.tree.DecisionTreeClassifier` on the Iris dataset.

The three-step workflow is the same for every score:

1. Wrap the model once with :func:`~probly.conformal_prediction.methods.classification.clas_conformal`.
2. Re-calibrate with a different :class:`~probly.conformal_prediction.scores_new.NonConformityScore`.
3. Call :func:`~probly.representer.representer` to obtain typed conformal sets.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from probly.calibrator import calibrate
from probly.conformal.metrics import average_set_size, empirical_coverage_classification
from probly.conformal.methods.classification import conformalize_classifier
from probly.conformal.scores import APSScore, LACScore, RAPSScore, SAPSScore
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
# ``conformalize_classifier`` deep-copies the model and attaches conformal attributes.
# Training must happen **after** the wrap so that the copy's state is updated.

model = DecisionTreeClassifier(max_depth=2, random_state=42)
model = conformalize_classifier(model)
model.fit(X_train, y_train)

# %%
# LAC score
# ---------
# The Least Ambiguous set-valued Classifier score: ``1 - P(y | x)``.

calibrate(model, X_calib, y_calib, LACScore(), alpha=0.05)
output = representer(model).predict(X_test)
lac_cov = empirical_coverage_classification(output, y_test)
lac_size = average_set_size(output)
print(f"LAC  — coverage: {lac_cov:.3f}, avg set size: {lac_size:.3f}")

# %%
# APS score
# ---------
# Adaptive Prediction Sets: cumulative sorted probabilities with randomisation.

calibrate(model, X_calib, y_calib, APSScore(), alpha=0.05)
output = representer(model).predict(X_test)
aps_cov = empirical_coverage_classification(output, y_test)
aps_size = average_set_size(output)
print(f"APS  — coverage: {aps_cov:.3f}, avg set size: {aps_size:.3f}")

# %%
# RAPS score
# ----------
# Regularised APS: adds a size penalty to encourage smaller prediction sets.

calibrate(model, X_calib, y_calib, RAPSScore(), alpha=0.05)
output = representer(model).predict(X_test)
raps_cov = empirical_coverage_classification(output, y_test)
raps_size = average_set_size(output)
print(f"RAPS — coverage: {raps_cov:.3f}, avg set size: {raps_size:.3f}")

# %%
# SAPS score
# ----------
# Sorted APS: penalises gaps between consecutive sorted probabilities.

calibrate(model, X_calib, y_calib, SAPSScore(), alpha=0.05)
output = representer(model).predict(X_test)
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
