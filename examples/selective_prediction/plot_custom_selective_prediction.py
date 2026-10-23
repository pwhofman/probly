"""=================================
Extending Selective Prediction
=================================

Plug your own loss and your own selection rule into selective prediction.

A :func:`~probly.selective_prediction.selective_predictor` takes a task loss, and the selection rule is a
:class:`~probly.selective_prediction.selectors.Selector`. Both can be written by you. Here, a random forest from
scikit-learn faces an asymmetric task: missing a positive costs three times as much as a false alarm, and abstaining
costs 0.3. The rule that fits this task is not among the built-in ones, so it is registered as a new one.
"""

from __future__ import annotations

from typing import Any, Self, override

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier

from probly.representer import representer
from probly.selective_prediction import selective_predictor, selector
from probly.selective_prediction.selectors import Selector, selector_registry

BLUE, RED, GRAY = "#1e88e5", "#ff0d57", "#7f8c8d"

# %%
# Data and model
# --------------
#
# The forest is used as is, without a probly transformation. A validation split is held out to fit the rule.

X_train, y_train = make_moons(n_samples=1000, noise=0.25, random_state=0)
X_val, y_val = make_moons(n_samples=1000, noise=0.25, random_state=2)
X_test, y_test = make_moons(n_samples=1000, noise=0.25, random_state=1)

model = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=0).fit(X_train, y_train)

# %%
# A loss of your own
# ------------------
#
# A function loss receives the decision, a categorical distribution, and the labels, and returns one loss per
# instance. Class 1 is the positive class.

MISS_COST, ABSTAIN_COST = 3.0, 0.3


def asymmetric_loss(decision: Any, y: np.ndarray) -> np.ndarray:  # noqa: ANN401
    predicted = decision.probabilities.argmax(axis=-1)
    return np.where(predicted == y, 0.0, np.where(y == 1, MISS_COST, 1.0))


# %%
# A selector of your own
# ----------------------
#
# A selector decides on an array of criterion values alone. This one is fitted on the criterion and the losses of
# calibration data: it accepts the instances below the threshold that has the lowest empirical cost, where an accepted
# instance costs its loss and a rejected one costs ``cost``. ``takes_losses`` tells the predictor to pass the losses
# to ``calibrate``, and ``is_calibrated`` makes it check the rule before the forward pass. Registering the class
# under a keyword makes :func:`~probly.selective_prediction.selector` build it.


@selector_registry.register("cost")  # ty: ignore[invalid-argument-type]
class CostSelector(Selector):
    takes_losses = True

    def __init__(self, cost: float) -> None:
        self.cost = float(cost)

    @override
    def __repr__(self) -> str:
        return f"selector(cost={self.cost})"

    @property
    def is_calibrated(self) -> bool:
        return self.threshold is not None

    @override
    def select(self, uncertainty: Any) -> Any:
        if not self.is_calibrated:
            msg = f"{self!r} is not calibrated. Call calibrate() first."
            raise ValueError(msg)
        return uncertainty <= self.threshold

    def calibrate(self, uncertainty: Any, losses: Any) -> Self:
        order = np.argsort(np.asarray(uncertainty, dtype=float), kind="stable")
        sorted_uncertainty = np.asarray(uncertainty, dtype=float)[order]
        sorted_losses = np.asarray(losses, dtype=float)[order]
        n = len(sorted_losses)
        # Accepting the k most certain instances costs their losses plus the abstention cost of the other n - k.
        total = np.concatenate([[0.0], np.cumsum(sorted_losses)]) + self.cost * (n - np.arange(n + 1))
        k = int(np.argmin(total))
        self.threshold = -np.inf if k == 0 else float(sorted_uncertainty[k - 1])
        return self


# %%
# Selective prediction
# --------------------
#
# The predictor takes the new rule and the loss like any other. The default criterion is one minus the largest
# predicted probability. The rule is fitted on the validation split, whose labels give the losses.

predictor = selective_predictor(representer(model), selector=selector(cost=ABSTAIN_COST), loss=asymmetric_loss)
predictor.calibrate(y_val, X_val)
result = predictor.predict(X_test)

losses = asymmetric_loss(result.decision, y_test)
accepted = result.accepted
print(f"threshold {predictor.selector.threshold:.3f}  coverage {accepted.mean():.2f}")
print(f"cost without abstaining  {losses.mean():.3f}")
print(f"cost with abstaining     {np.where(accepted, losses, ABSTAIN_COST).mean():.3f}")

# %%
# Selectors on raw arrays
# -----------------------
#
# The selector never sees the model, so it also works on a criterion and losses that you compute by hand, for a model
# that probly does not know. This reproduces the selection above.

probabilities_val, probabilities_test = model.predict_proba(X_val), model.predict_proba(X_test)
losses_val = np.where(probabilities_val.argmax(axis=1) == y_val, 0.0, np.where(y_val == 1, MISS_COST, 1.0))
rule = selector(cost=ABSTAIN_COST).calibrate(1.0 - probabilities_val.max(axis=1), losses_val)
by_hand = rule.select(1.0 - probabilities_test.max(axis=1))
print(f"same selection: {np.array_equal(by_hand, accepted)}")

# %%
# Decision regions
# ----------------
#
# Gray regions are abstained on. Crosses mark accepted instances that cost something.

xx, yy = np.meshgrid(np.linspace(-3.0, 4.0, 300), np.linspace(-2.5, 3.0, 300))
grid_result = predictor.predict(np.c_[xx.ravel(), yy.ravel()])
grid_prediction = grid_result.decision.probabilities.argmax(axis=-1)

panels = [
    ("Normal prediction", grid_prediction, np.ones_like(accepted)),
    ("Selective prediction", np.where(grid_result.accepted, grid_prediction, 2), accepted),
]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
for ax, (title, regions, shown) in zip(axes, panels, strict=True):
    ax.contourf(xx, yy, regions.reshape(xx.shape), levels=[-0.5, 0.5, 1.5, 2.5], colors=[BLUE, RED, GRAY], alpha=0.3)
    ax.scatter(*X_test.T, c=y_test, cmap=ListedColormap([BLUE, RED]), s=6, alpha=0.6)
    costly = shown & (losses > 0)
    ax.scatter(*X_test[costly].T, marker="x", color="black", s=20, label=f"{costly.sum()} costly predictions")
    ax.set_title(title)
    ax.set_xlabel("$x_1$")
    ax.legend(loc="lower right")
axes[0].set_ylabel("$x_2$")
plt.tight_layout()
plt.show()
