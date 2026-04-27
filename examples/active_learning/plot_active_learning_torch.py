"""==========================================
Active Learning with PyTorch — Margin vs Random
==========================================

Compare margin sampling against random query selection using a small
PyTorch neural network on the Digits dataset.

This example mirrors the sklearn version but demonstrates the torch-backed
active learning pipeline. The same strategies, pool, and metrics work
transparently with torch tensors.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
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
TRAIN_EPOCHS = 30
LEARNING_RATE = 1e-2


# %%
# Data preparation
# ----------------
# Load Digits, split 80/20, and convert to float32 tensors.

X, y = load_digits(return_X_y=True)
x_train_np, x_test_np, y_train_np, y_test_np = train_test_split(
    X, y, test_size=0.2, random_state=SEED,
)

x_train = torch.from_numpy(x_train_np).float()
y_train = torch.from_numpy(y_train_np).long()
x_test = torch.from_numpy(x_test_np).float()
y_test = torch.from_numpy(y_test_np).long()


# %%
# Estimator wrapper
# -----------------
# A small two-layer network wrapped to satisfy the
# :class:`~probly.evaluation.active_learning.Estimator` protocol.


class TorchEstimator:
    """Small MLP estimator for the active learning loop."""

    def __init__(self, n_features: int, n_classes: int) -> None:
        self._n_features = n_features
        self._n_classes = n_classes
        self._model: nn.Module | None = None

    def _build(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(self._n_features, 64),
            nn.ReLU(),
            nn.Linear(64, self._n_classes),
        )

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self._model = self._build()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=LEARNING_RATE)
        loss_fn = nn.CrossEntropyLoss()
        self._model.train()
        for _ in range(TRAIN_EPOCHS):
            optimizer.zero_grad()
            loss_fn(self._model(x), y).backward()
            optimizer.step()

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        assert self._model is not None
        self._model.eval()
        return self._model(x).argmax(dim=1)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        assert self._model is not None
        self._model.eval()
        return torch.softmax(self._model(x), dim=1)


# %%
# Run active learning
# -------------------
# Same two strategies, now operating on torch tensors end-to-end.
# The pool, strategies, and metrics all dispatch to their torch
# implementations automatically.

torch.manual_seed(SEED)

strategies = {
    "Margin Sampling": MarginSampling(),
    "Random": RandomQuery(seed=SEED),
}

results: dict[str, dict[str, list[float]]] = {}

for name, strategy in strategies.items():
    pool = from_dataset(x_train, y_train, x_test, y_test, initial_size=INITIAL_SIZE, seed=SEED)
    estimator = TorchEstimator(n_features=x_train.shape[1], n_classes=10)

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
# Both strategies start from the same initial labeled set. Margin sampling
# queries the most uncertain samples, reaching higher accuracy with fewer
# labels.

fig, ax = plt.subplots(figsize=(8, 5))
for name, data in results.items():
    ax.plot(data["labeled_sizes"], data["accuracies"], marker="o", label=name)
ax.set_xlabel("Labeled samples")
ax.set_ylabel("Test accuracy")
ax.set_title("Active learning on Digits (PyTorch)")
ax.legend()
ax.grid(alpha=0.25)
plt.tight_layout()
plt.show()
