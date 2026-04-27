"""============================================
Active Learning with sklearn — Margin vs Random
============================================

Compare margin sampling against random query selection on the Digits
dataset using a :class:`~sklearn.linear_model.LogisticRegression` model.

The workflow follows four steps:

1. Create an active learning pool with :func:`~probly.evaluation.active_learning.from_dataset`.
2. Wrap an sklearn estimator so it conforms to the :class:`~probly.evaluation.active_learning.Estimator` protocol.
3. Iterate with :func:`~probly.evaluation.active_learning.active_learning_steps`.
4. Evaluate with :func:`~probly.evaluation.active_learning.compute_accuracy` and
   :func:`~probly.evaluation.active_learning.compute_nauc`.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from probly.evaluation.active_learning import (
    MarginSampling,
    RandomQuery,
    active_learning_steps,
    compute_accuracy,
    compute_nauc,
    from_dataset,
)

SEED = 42
INITIAL_SIZE = 50
QUERY_SIZE = 50
N_ITERATIONS = 10

# %%
# Data preparation
# ----------------
# Load the Digits dataset (1797 samples, 64 features, 10 classes) and split
# into 80 % train / 20 % test.

X, y = load_digits(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED,
)


# %%
# Estimator wrapper
# -----------------
# The active learning loop expects an object with ``fit``, ``predict``, and
# ``predict_proba`` methods that accept and return numpy arrays.


class SklearnEstimator:
    """Thin wrapper making an sklearn classifier compatible with the AL loop."""

    def __init__(self) -> None:
        self._model = LogisticRegression(max_iter=500, random_state=SEED)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._model.predict(x)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(x).astype(np.float32)


# %%
# Run active learning
# -------------------
# We run two strategies side by side: margin sampling (selects the most
# uncertain samples) and random selection (baseline).

strategies = {
    "Margin Sampling": MarginSampling(),
    "Random": RandomQuery(seed=SEED),
}

results: dict[str, dict[str, list[float]]] = {}

for name, strategy in strategies.items():
    pool = from_dataset(x_train, y_train, x_test, y_test, initial_size=INITIAL_SIZE, seed=SEED)
    estimator = SklearnEstimator()

    accuracies: list[float] = []
    labeled_sizes: list[int] = []

    for state in active_learning_steps(
        pool, estimator, strategy, query_size=QUERY_SIZE, n_iterations=N_ITERATIONS,
    ):
        acc = compute_accuracy(
            state.estimator.predict(state.pool.x_test), state.pool.y_test,
        )
        accuracies.append(acc)
        labeled_sizes.append(state.pool.n_labeled)

    nauc = compute_nauc(accuracies)
    results[name] = {"accuracies": accuracies, "labeled_sizes": labeled_sizes}
    print(f"{name:20s}  final acc: {accuracies[-1]:.3f}  NAUC: {nauc:.3f}")

# %%
# Plot accuracy curves
# --------------------
# Margin sampling reaches higher accuracy faster because it queries the
# samples the model is most confused about.

fig, ax = plt.subplots(figsize=(8, 5))
for name, data in results.items():
    ax.plot(data["labeled_sizes"], data["accuracies"], marker="o", label=name)
ax.set_xlabel("Labeled samples")
ax.set_ylabel("Test accuracy")
ax.set_title("Active learning on Digits (sklearn)")
ax.legend()
ax.grid(alpha=0.25)
plt.tight_layout()
plt.show()
