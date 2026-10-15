"""=============================================
Selective Prediction with a Model of Your Own
=============================================

Abstain with a model that probly does not know, by computing the criterion, the decision and the losses yourself and
giving them to a selector.

A :class:`~probly.selective_prediction.Selector` decides on arrays of uncertainty values alone, so it works with any
model. Here, the model is a random forest regressor. The spread of the trees is the criterion, their mean is the
decision, and a prediction counts as an error if it is off by more than a tolerance. This turns a regression task into
one with binary losses, so :func:`~probly.selective_prediction.selector` with ``risk`` and ``delta`` applies.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from probly.evaluation.selective_prediction import evaluate_selective_prediction
from probly.metrics.selective_prediction import risk_coverage_curve
from probly.selective_prediction import selector

BLUE, RED, GRAY = "#1e88e5", "#ff0d57", "#7f8c8d"

# %%
# Data
# ----
#
# The target is a sine curve with noise that grows with ``|x|``. The training data have a gap around ``x = 2``, and
# the test inputs cover the gap and a range beyond the training data. A validation split, drawn like the training
# data, is held out to calibrate the rule.

rng = np.random.default_rng(0)


def sample(n: int, low: float = -4.0, high: float = 4.0, gap: bool = False) -> tuple[np.ndarray, np.ndarray]:
    x = rng.uniform(low, high, size=n)
    if gap:
        x = x[np.abs(x - 2.0) > 0.7]
    y = np.sin(x) + rng.normal(scale=0.05 + 0.1 * np.abs(x), size=x.shape)
    return x[:, None], y


X_train, y_train = sample(800, gap=True)
X_val, y_val = sample(600, gap=True)
X_test, y_test = sample(2000, low=-5.0, high=5.0)

# %%
# Criterion, decision and loss
# ----------------------------
#
# ``predict`` of the forest gives only the mean over the trees. The prediction of every tree has shape
# ``(n_instances, n_trees)``, from which three functions define the task:
#
# - the criterion is one uncertainty per instance, higher meaning more uncertain: the standard deviation of the trees.
# - the decision is the mean of the trees.
# - the loss is one value per instance. A prediction is wrong (loss 1) if it is off by more than ``tolerance``.

forest = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=0).fit(X_train, y_train)
tolerance = 0.3


def predict(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the criterion and the decision for the inputs."""
    output = np.stack([tree.predict(x) for tree in forest.estimators_], axis=1)
    return output.std(axis=1), output.mean(axis=1)


def loss(decision: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (np.abs(decision - y) > tolerance).astype(float)


# %%
# Selection rule
# --------------
#
# The risk rule picks the largest set of instances whose selective risk, the error rate among accepted instances, is
# below ``risk`` with probability at least ``1 - delta``. It is fitted on the criterion and the losses of the
# validation split.

uncertainty_val, decision_val = predict(X_val)
rule = selector(risk=0.15, delta=0.1).calibrate(uncertainty_val, loss(decision_val, y_val))
print(f"threshold on the tree spread: {rule.threshold:.3f}   certified bound on the risk: {rule.bound:.3f}")

# %%
# Prediction
# ----------
#
# The rule accepts the instances whose criterion is at most the threshold. The decision exists for rejected
# instances as well, so you can still inspect it.

uncertainty, decision = predict(X_test)
accepted = rule.select(uncertainty)
error = loss(decision, y_test).astype(bool)

print(f"all instances    coverage 1.00  error rate {error.mean():.3f}")
print(f"accepted only    coverage {accepted.mean():.2f}  error rate {error[accepted].mean():.3f}  (target 0.15)")

# %%
# Where the model abstains
# ------------------------
#
# The forest is accurate where the data are dense and the noise is small. It accepts almost only instances near the
# center and abstains where the trees disagree: in the gap, beyond the training range, and in the noisy tails.

order = np.argsort(X_test[:, 0])
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, height_ratios=[3, 1.5])
ax = axes[0]
ax.scatter(X_test[accepted, 0], y_test[accepted], s=6, color=BLUE, alpha=0.5, label="accepted")
ax.scatter(X_test[~accepted, 0], y_test[~accepted], s=6, color=GRAY, alpha=0.5, label="rejected")
ax.scatter(X_test[accepted & error, 0], y_test[accepted & error], marker="x", s=20, color=RED, label="accepted error")
ax.plot(X_test[order, 0], decision[order], color="black", linewidth=1.2, label="forest mean")
ax.axvspan(1.3, 2.7, color=GRAY, alpha=0.15)
ax.set_ylabel("$y$")
ax.legend(loc="upper left", ncol=2)

ax = axes[1]
ax.plot(X_test[order, 0], uncertainty[order], color=BLUE, linewidth=0.8)
ax.axhline(rule.threshold, color=RED, linestyle="--", label="threshold")
ax.axvspan(1.3, 2.7, color=GRAY, alpha=0.15)
ax.set_xlabel("$x$")
ax.set_ylabel("tree spread")
ax.legend(loc="upper center")
plt.tight_layout()
plt.show()

# %%
# Risk-coverage curve
# -------------------
#
# The same uncertainty and losses give the full risk-coverage curve, and
# :func:`~probly.evaluation.selective_prediction.evaluate_selective_prediction` summarizes it (lower is better). The dot
# marks the point reached by the calibrated threshold on the test set.

losses = loss(decision, y_test)
curve_coverage, curve_risk, _ = risk_coverage_curve(uncertainty, losses)
m = evaluate_selective_prediction(uncertainty, losses, ["augrc", "aurc", "risk@1"])
print(f"AUGRC {m['augrc']:.4f}  AURC {m['aurc']:.4f}  risk at full coverage {m['risk@1']:.3f}")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(curve_coverage, curve_risk, color=BLUE, label="forest, ranked by tree spread")
ax.scatter(accepted.mean(), error[accepted].mean(), color=RED, zorder=3, label="calibrated threshold")
ax.axhline(0.15, color="black", linestyle="--", label="target risk")
ax.set_xlabel("coverage")
ax.set_ylabel("selective risk")
ax.set_xlim(0, 1)
ax.legend(loc="upper left")
plt.tight_layout()
plt.show()
