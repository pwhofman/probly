"""===================================================
Selective Prediction with a scikit-learn Classifier
===================================================

Abstain with a plain scikit-learn classifier, without any probly transformation.

:class:`~probly.selective_prediction.SelectivePredictor` accepts every model with ``predict_proba``, and the default
criterion is one minus the maximum predicted probability. Here, the model is a random forest, and the threshold is
chosen on held-out data for a target coverage or given as the cost of abstaining. The last section shows how to use
a criterion that you compute yourself, which also covers models that probly does not know.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier

from probly.evaluation.selective_prediction import evaluate_selective_prediction
from probly.metrics.selective_prediction import risk_coverage_curve
from probly.selective_prediction import SelectivePredictor, selector

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
# Threshold for a target coverage
# -------------------------------
#
# The predictor calibrates a threshold on the unlabeled validation criterion, so that about the target fraction of new
# instances from the same distribution is accepted. The criterion is one value per instance, and higher means more
# uncertain.

target_coverages = [0.95, 0.9, 0.8, 0.7, 0.5]
predictors = {
    c: SelectivePredictor(model, selector=selector(coverage=c)).calibrate(None, X_val) for c in target_coverages
}

# %%
# Selective prediction
# --------------------
#
# The decision is the most probable class, as usual, and the uncertainty is one minus its probability.

results = {c: predictor.predict(X_test) for c, predictor in predictors.items()}
test_uncertainty = results[0.8].uncertainty
prediction = results[0.8].decision.probabilities.argmax(axis=-1)
correct = prediction == y_test

print(f"normal prediction          coverage 1.00  accuracy {correct.mean():.3f}")
for coverage, result in results.items():
    accepted = result.accepted
    print(
        f"selective (target {coverage:.2f})    coverage {accepted.mean():.2f}  accuracy {correct[accepted].mean():.3f}"
        f"  threshold {predictors[coverage].selector.threshold:.3f}"
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

predictor = predictors[0.8]
xx, yy = np.meshgrid(np.linspace(-3.0, 4.0, 300), np.linspace(-2.5, 3.0, 300))
grid = np.c_[xx.ravel(), yy.ravel()]
grid_result = predictor.predict(grid)
grid_prediction = grid_result.decision.probabilities.argmax(axis=-1)
grid_accepted = grid_result.accepted

panels = [
    ("Normal prediction", grid_prediction, np.ones_like(correct)),
    ("Selective (target coverage 0.8)", np.where(grid_accepted, grid_prediction, 2), results[0.8].accepted),
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
# The curve comes from :func:`~probly.metrics.selective_prediction.risk_coverage_curve`, and
# :func:`~probly.evaluation.selective_prediction.evaluate_selective_prediction` summarizes it by its areas (lower is
# better) and the risk at full coverage. Dots mark the thresholds chosen on the validation split, at the coverage they
# reach on the test set.

fig, ax = plt.subplots(figsize=(7, 4.5))
test_errors = (~correct).astype(float)
curve_coverage, curve_risk, _ = risk_coverage_curve(test_uncertainty, test_errors)
ax.plot(curve_coverage, 1 - curve_risk, color=PURPLE, label="selective")
m = evaluate_selective_prediction(test_uncertainty, test_errors, ["augrc", "aurc", "risk@1"])
print(f"AUGRC {m['augrc']:.4f}  AURC {m['aurc']:.4f}  risk at full coverage {m['risk@1']:.3f}")
for coverage, result in results.items():
    accepted = result.accepted
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
# threshold of ``c`` on the criterion above (``threshold=c``) and needs no validation data. With ``K`` classes, only
# costs below ``1 - 1/K`` lead to any abstentions.
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
    accepted = SelectivePredictor(model, selector=selector(threshold=cost)).predict(X_test).accepted
    chow_cost = empirical_cost(accepted, cost)
    curve = [empirical_cost(test_uncertainty <= t, cost) for t in thresholds]
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
# A criterion of your own
# -----------------------
#
# A selector decides on an array of criterion values alone, so it works with any model, also one that probly does not
# know. Here, the criterion is computed by hand from the forest's probabilities and reproduces the selection above.

rule = selector(coverage=0.8).calibrate(1.0 - model.predict_proba(X_val).max(axis=1))
own_uncertainty = 1.0 - model.predict_proba(X_test).max(axis=1)
print(f"same criterion: {np.allclose(own_uncertainty, test_uncertainty)}")
print(f"same selection: {np.array_equal(rule.select(own_uncertainty), results[0.8].accepted)}")
