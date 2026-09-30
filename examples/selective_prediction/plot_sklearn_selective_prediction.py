"""===================================================
Selective Prediction with a scikit-learn Classifier
===================================================

Abstain with a plain scikit-learn classifier, using an uncertainty score you compute yourself.

A :class:`~probly.selective_prediction.Selector` decides on the uncertainty criterion alone, so it works with any
model: compute the criterion per instance and apply the selector to it. This route also covers precomputed scores and
models that probly does not know. Here, the model is a random forest, the criterion is one minus its maximum predicted
probability, and the threshold is chosen on held-out data for a target coverage. The last section shows the shorter
route through :class:`~probly.selective_prediction.SelectivePredictor`, which accepts the forest once its output type
is declared with :func:`~probly.method.cast`.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier

from probly.method import cast
from probly.selective_prediction import SelectivePredictor, ThresholdSelector

BLUE, RED, GRAY, PURPLE = "#1e88e5", "#ff0d57", "#7f8c8d", "#9b59b6"

# %%
# Data
# ----
#
# A validation split is held out to choose the threshold.

X_train, y_train = make_moons(n_samples=1000, noise=0.25, random_state=0)
X_val, y_val = make_moons(n_samples=500, noise=0.25, random_state=2)
X_test, y_test = make_moons(n_samples=1000, noise=0.25, random_state=1)

# %%
# Train
# -----
#
# The model is used as is, without a probly transformation.

model = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=0).fit(X_train, y_train)

# %%
# Uncertainty criterion
# ---------------------
#
# Any score works as long as it has one value per instance and higher means more uncertain. A confidence score such
# as the maximum probability therefore has to be turned around.


def uncertainty(x: np.ndarray) -> np.ndarray:
    return 1.0 - model.predict_proba(x).max(axis=1)


# %%
# Threshold for a target coverage
# -------------------------------
#
# A threshold at the ``q``-quantile of the validation criterion accepts a fraction of about ``q`` of new instances
# from the same distribution. The ``inverted_cdf`` quantile is a value of the criterion itself, so ties at the
# threshold are accepted, as :class:`~probly.selective_prediction.ThresholdSelector` does.

val_uncertainty = uncertainty(X_val)
target_coverages = [0.95, 0.9, 0.8, 0.7, 0.5]
selectors = {
    coverage: ThresholdSelector(np.quantile(val_uncertainty, coverage, method="inverted_cdf"))
    for coverage in target_coverages
}

# %%
# Selective prediction
# --------------------
#
# The selector sees only the criterion; the prediction comes from the model as usual.

test_uncertainty = uncertainty(X_test)
prediction = model.predict(X_test)
correct = prediction == y_test

print(f"normal prediction          coverage 1.00  accuracy {correct.mean():.3f}")
for coverage, selector in selectors.items():
    accepted = selector.select(test_uncertainty)
    print(
        f"selective (target {coverage:.2f})    coverage {accepted.mean():.2f}  accuracy {correct[accepted].mean():.3f}"
        f"  threshold {selector.threshold:.3f}"
    )

# %%
# Decision regions
# ----------------
#
# Gray regions are abstained on at a target coverage of 0.8. Crosses mark wrong predictions that are not abstained on.
#
# Far from the data, the forest abstains in some directions and is confident in others. The maximum probability
# measures how ambiguous an instance is, not how little data the model has seen near it; for a model that also
# abstains far from the data, see :ref:`sphx_glr_auto_examples_selective_prediction_plot_sngp_selective_prediction.py`.

selector = selectors[0.8]
xx, yy = np.meshgrid(np.linspace(-3.0, 4.0, 300), np.linspace(-2.5, 3.0, 300))
grid = np.c_[xx.ravel(), yy.ravel()]
grid_prediction = model.predict(grid)
grid_accepted = selector.select(uncertainty(grid))

panels = [
    ("Normal prediction", grid_prediction, np.ones_like(correct)),
    ("Selective (target coverage 0.8)", np.where(grid_accepted, grid_prediction, 2), selector(test_uncertainty)),
]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
for ax, (title, regions, accepted) in zip(axes, panels, strict=True):
    ax.contourf(xx, yy, regions.reshape(xx.shape), levels=[-0.5, 0.5, 1.5, 2.5], colors=[BLUE, RED, GRAY], alpha=0.3)
    ax.scatter(*X_test.T, c=y_test, cmap=ListedColormap([BLUE, RED]), s=6, alpha=0.6)
    errors = accepted & ~correct
    ax.scatter(*X_test[errors].T, marker="x", color="black", s=20, label=f"{errors.sum()} accepted errors")
    ax.set_title(title)
    ax.set_xlabel("$x_1$")
    ax.legend(loc="lower right")
axes[0].set_ylabel("$x_2$")
plt.tight_layout()
plt.show()

# %%
# Accuracy vs. coverage
# ---------------------
#
# Dots mark the thresholds chosen on the validation split, at the coverage they reach on the test set.

fig, ax = plt.subplots(figsize=(7, 4.5))
hit = correct[np.argsort(test_uncertainty, kind="stable")]
n = np.arange(1, len(hit) + 1)
ax.plot(n / len(hit), np.cumsum(hit) / n, color=PURPLE, label="selective")
for coverage, selector in selectors.items():
    accepted = selector(test_uncertainty)
    point = (accepted.mean(), correct[accepted].mean())
    ax.scatter(*point, color=PURPLE, zorder=3)
    ax.annotate(f"{coverage:.2f}", point, textcoords="offset points", xytext=(0, 8), ha="center")
ax.axhline(correct.mean(), color="black", linestyle="--", label="normal prediction")
ax.set_xlabel("coverage")
ax.set_ylabel("accuracy on accepted")
ax.set_xlim(0, 1)
ax.legend(loc="lower left")
plt.tight_layout()
plt.show()

# %%
# Chow's rule
# -----------
#
# Instead of a target coverage, a cost ``c`` for abstaining can be given, where a correct prediction costs 0 and a
# wrong one costs 1. Chow's rule (Chow, 1970) accepts a prediction if and only if its
# maximum probability is at least ``1 - c``, which is the same as a criterion of at most ``c``. It is therefore a
# :class:`~probly.selective_prediction.ThresholdSelector` at ``c`` on the criterion above and needs no validation data.
# With ``K`` classes, only costs below ``1 - 1/K`` lead to any abstentions.
# For models transformed by probly, :class:`~probly.selective_prediction.SelectivePredictor` uses this criterion by
# default, so a :class:`~probly.selective_prediction.ThresholdSelector` at ``c`` applies Chow's rule there as well.
#
# The rule minimizes the expected cost if the predicted probabilities are the true class probabilities. The forest
# only estimates them, so the curves below show how close Chow's threshold comes to the lowest cost on the test set.
# The lowest cost lies at slightly larger thresholds: averaging over trees and leaves pulls the forest's probabilities
# toward 0.5, so it is underconfident, and Chow's rule abstains a little more often than the costs warrant.

costs = [0.05, 0.1, 0.2]
thresholds = np.linspace(0.0, 0.5, 201)


def empirical_cost(accepted: np.ndarray, cost: float) -> float:
    return float(np.mean(np.where(accepted, ~correct, cost)))


fig, ax = plt.subplots(figsize=(7, 4.5))
for cost, color in zip(costs, [BLUE, PURPLE, RED], strict=True):
    chow = ThresholdSelector(cost)
    accepted = chow.select(test_uncertainty)
    chow_cost = empirical_cost(accepted, cost)
    curve = [empirical_cost(ThresholdSelector(t).select(test_uncertainty), cost) for t in thresholds]
    best = thresholds[np.argmin(curve)]
    print(
        f"Chow (c = {cost:.2f})    coverage {accepted.mean():.2f}  accuracy {correct[accepted].mean():.3f}"
        f"  cost {chow_cost:.3f}  (lowest {min(curve):.3f} at threshold {best:.3f})"
    )
    ax.plot(thresholds, curve, color=color, label=f"c = {cost}")
    ax.scatter(cost, chow_cost, color=color, zorder=3)
ax.set_xlabel("threshold on 1 - max probability")
ax.set_ylabel("cost per instance")
ax.set_xlim(0, 0.5)
ax.legend(title="Chow's rule (dots)")
plt.tight_layout()
plt.show()

# %%
# The same selection with SelectivePredictor
# ------------------------------------------
#
# Declared as a probabilistic classifier with :func:`~probly.method.cast`, the forest can be wrapped by
# :class:`~probly.selective_prediction.SelectivePredictor`. Its representation is a single categorical distribution,
# so the default criterion is one minus the maximum probability, the criterion computed by hand above, and the
# predictor accepts exactly the same instances.

predictor = SelectivePredictor(cast(model, predictor_type="probabilistic_classifier"), selectors[0.8])
result = predictor.predict(X_test)
print(f"same criterion: {np.allclose(result.uncertainty, test_uncertainty)}")
print(f"same selection: {np.array_equal(result.accepted, selectors[0.8].select(test_uncertainty))}")
