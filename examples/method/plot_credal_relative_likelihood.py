"""========================================
Credal Relative Likelihood Visualization
========================================

This example uses ``credal_relative_likelihood`` to build a class-bias ensemble
and visualize the resulting probability interval credal sets.

The first ensemble member is trained to convergence on the full data. Each
subsequent member is trained only until its relative likelihood (the ratio of
its training likelihood to that of the first member) crosses a per-member
threshold, mirroring the benchmark training recipe.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from probly.method.credal_relative_likelihood import credal_relative_likelihood
from probly.plot.credal import plot_credal_set
from probly.representer import representer

from examples.utils.model import MLPClassifier

np.random.seed(42)
torch.manual_seed(42)


def _train_member(member: torch.nn.Module, loader: DataLoader, epochs: int, lr: float = 1e-2) -> None:
    member.train()
    opt = torch.optim.Adam(member.parameters(), lr=lr)
    for _epoch in range(epochs):
        for inputs, targets in loader:
            opt.zero_grad()
            loss = F.cross_entropy(member(inputs), targets)
            loss.backward()
            opt.step()


@torch.no_grad()
def _log_likelihood(member: torch.nn.Module, loader: DataLoader) -> float:
    member.eval()
    total, count = 0.0, 0
    for inputs, targets in loader:
        log_probs = F.log_softmax(member(inputs), dim=-1)
        total += log_probs.gather(1, targets.unsqueeze(1)).sum().item()
        count += targets.numel()
    return total / count


# %%
# Setup
# -----

centers = [[-7.0, -4.0], [0.0, 8.0], [7.0, -4.0]]
X, y = make_blobs(n_samples=300, centers=centers, cluster_std=2.0, random_state=42)
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).long()

dataset = TensorDataset(X_train_tensor, y_train_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# %%
# Model
# -----

base_model = MLPClassifier(in_features=2, hidden_features=64, out_features=3)
num_members = 5
credal_model = credal_relative_likelihood(
    base_model,
    predictor_type="logit_classifier",
    num_members=num_members,
)
members = list(credal_model)

# %%
# Training
# --------
#
# Train the first member fully (its relative likelihood is 1.0 by definition),
# compute the reference log-likelihood, then train the remaining members until
# each reaches its assigned relative-likelihood threshold.

alpha = 0.5
_train_member(members[0], dataloader, epochs=50)
max_ll = _log_likelihood(members[0], dataloader)

thresholds = torch.linspace(alpha, 1.0, num_members)[:-1].tolist()
max_epochs_per_step = 5

for member, threshold in zip(members[1:], thresholds, strict=True):
    for _ in range(50):
        _train_member(member, dataloader, epochs=max_epochs_per_step)
        rel_lik = float(np.exp(_log_likelihood(member, dataloader) - max_ll))
        if rel_lik >= threshold:
            break

for member in members:
    member.eval()

# %%
# Credal Set Visualization
# ------------------------

rep = representer(credal_model)
X_test = torch.tensor([
    [-7.0, -4.0],
    [0.0, 0.0],
    [0.0, 8.0],
])
credal_sets = rep.predict(X_test)

plot_credal_set(
    credal_sets,
    title="Credal Relative Likelihood",
    labels=["Class 0", "Class 1", "Class 2"],
    series_labels=["Near Class 0", "OOD", "Near Class 1"],
    show=True,
)
