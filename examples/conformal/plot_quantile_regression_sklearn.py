"""==================================================
Quantile Regression Conformal Prediction — sklearn
==================================================

Demonstrate :func:`~probly.conformal.scores.cqr_score` and
:func:`~probly.conformal.scores.cqr_r_score` using a custom
``DualQuantileRegressor`` wrapper on the Diabetes dataset.

sklearn's :class:`~sklearn.linear_model.QuantileRegressor` predicts a
single quantile. ``DualQuantileRegressor`` trains two instances — one for
the lower quantile :math:`\\alpha/2` and one for :math:`1 - \\alpha/2` —
and returns both as a ``(n_samples, 2)`` array, which is the format expected
by the quantile-regression conformal scores.

**CQR** (Conformalized Quantile Regression) adjusts the interval symmetrically::

    score = max(q_lo - y, y - q_hi)

**CQRr** normalises by the interval width, giving adaptive-width corrections.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.datasets import load_diabetes
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import train_test_split

from probly.calibrator import calibrate
from probly.metrics._common import average_interval_size, empirical_coverage_regression
from probly.method.conformal import conformal_cqr, conformal_cqr_r
from probly.representer import representer

# %%
# Data preparation
# ----------------

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_calib, y_train, y_calib = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# %%
# DualQuantileRegressor
# ---------------------
# A thin sklearn-compatible wrapper that trains one lower-quantile and one
# upper-quantile regressor and stacks their predictions into ``(n_samples, 2)``.


class DualQuantileRegressor(BaseEstimator, RegressorMixin):
    """Pair of QuantileRegressors producing ``[lower_q, upper_q]`` per sample.

    Args:
        alpha: Miscoverage level. The lower quantile is ``alpha / 2`` and the upper
            quantile is ``1 - alpha / 2``, targeting nominal coverage ``1 - alpha``
            before conformal calibration.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray) -> DualQuantileRegressor:
        self._lo = QuantileRegressor(quantile=self.alpha / 2, solver="highs")
        self._hi = QuantileRegressor(quantile=1.0 - self.alpha / 2, solver="highs")
        self._lo.fit(X, y)
        self._hi.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return shape ``(n_samples, 2)`` with columns ``[lower, upper]``."""
        return np.column_stack([self._lo.predict(X), self._hi.predict(X)])


# %%
# Build and train the model
# -------------------------
# Fit the base quantile regressor once, then wrap per score.

model = DualQuantileRegressor(alpha=0.1)
model.fit(X_train, y_train)

# %%
# CQR score
# ---------
# Symmetric correction: ``score = max(q_lo - y, y - q_hi)``.

calibrated_model = calibrate(conformal_cqr(model), 0.05, y_calib, X_calib)
output = representer(calibrated_model).predict(X_test)
cqr_cov = empirical_coverage_regression(output, y_test)
cqr_size = average_interval_size(output)
print(f"CQR  — coverage: {cqr_cov:.3f}, avg interval size: {cqr_size:.1f}")

# %%
# CQRr score
# ----------
# Width-normalised correction: adapts to heteroscedastic models.

calibrated_model = calibrate(conformal_cqr_r(model), 0.05, y_calib, X_calib)
output = representer(calibrated_model).predict(X_test)
cqrr_cov = empirical_coverage_regression(output, y_test)
cqrr_size = average_interval_size(output)
print(f"CQRr — coverage: {cqrr_cov:.3f}, avg interval size: {cqrr_size:.1f}")

# %%
# Comparison
# ----------

print("\n{:<5} {:>10} {:>18}".format("Score", "Coverage", "Avg interval size"))
print("-" * 35)
for name, cov, sz in [("CQR", cqr_cov, cqr_size), ("CQRr", cqrr_cov, cqrr_size)]:
    print(f"{name:<5} {cov:>10.3f} {sz:>18.1f}")
