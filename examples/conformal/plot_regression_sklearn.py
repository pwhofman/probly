"""==========================================
Regression Conformal Prediction — sklearn
==========================================

Demonstrate :func:`~probly.conformal.scores.absolute_error_score`
using a :class:`~sklearn.tree.DecisionTreeRegressor` on the Diabetes dataset.

The conformal interval for a new point :math:`x` is

.. math::

    \\hat{f}(x) \\pm \\hat{q}_{1-\\alpha}

where :math:`\\hat{q}_{1-\\alpha}` is the empirical :math:`(1-\\alpha)`-quantile
of the absolute residuals on the calibration set.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from probly.calibrator import calibrate
from probly.metrics._common import average_interval_size, empirical_coverage_regression
from probly.method.conformal import conformal_absolute_error
from probly.representer import representer

# %%
# Data preparation
# ----------------

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_calib, y_train, y_calib = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# %%
# Build and train the model
# -------------------------

model = DecisionTreeRegressor(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# %%
# Absolute error score
# --------------------
calibrated_model = calibrate(conformal_absolute_error(model), 0.05, y_calib, X_calib)
output = representer(calibrated_model).predict(X_test)
coverage = empirical_coverage_regression(output, y_test)
avg_size = average_interval_size(output)
print(f"Absolute Error — coverage: {coverage:.3f}, avg interval size: {avg_size:.1f}")

# %%
# Visualise prediction intervals
# -------------------------------
# Sort by true label for a cleaner plot.

intervals = output.array  # (n_test, 2): [lower, upper]
order = np.argsort(y_test)

plt.figure(figsize=(9, 4))
plt.fill_between(
    range(len(y_test)),
    intervals[order, 0],
    intervals[order, 1],
    alpha=0.35,
    label="90% conformal interval",
)
plt.scatter(range(len(y_test)), y_test[order], s=15, color="tab:red", label="True value", zorder=3)
plt.xlabel("Test sample (sorted by true label)")
plt.ylabel("Target")
plt.title("Conformal Regression Intervals — Absolute Error Score (sklearn)")
plt.legend()
plt.tight_layout()
plt.show()
