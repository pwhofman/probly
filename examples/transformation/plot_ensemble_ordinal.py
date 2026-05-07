"""==========================================
Ensemble Ordinal Classification Uncertainty
==========================================

Demonstrate :func:`~probly.method.ensemble.ensemble`
using a PyTorch multi-layer perceptron classifier on a dummy ordinal dataset.

This example computes epistemic, aleatoric, and total variance uncertainty
of the expected predictive distribution using ordinal variance measures.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from probly.method.ensemble import ensemble
from probly.representer import representer
from probly.quantification.decomposition.ordinal import (
    OrdinalVarianceDecomposition,
    OrdinalEntropyDecomposition,
)

# %%
# Data preparation
# ----------------
# We create a simple noisy sine wave dataset and discretize it into ordered classes.

X_np = np.linspace(-3, 3, 100).reshape(-1, 1).astype(np.float32)
y_continuous = np.sin(X_np) + np.random.normal(0, 0.1, X_np.shape)

# Create 5 ordinal classes
bins = np.linspace(-0.8, 0.8, 4)
y_np = np.digitize(y_continuous, bins).astype(np.int64).flatten()

# Split
idx = np.random.permutation(len(X_np))
X_train, X_test = torch.from_numpy(X_np[idx[:80]]), torch.from_numpy(X_np[idx[80:]])
y_train, y_test = torch.from_numpy(y_np[idx[:80]]), torch.from_numpy(y_np[idx[80:]])

# %%
# Build and train the ensemble
# -------------------------
# We use a simple PyTorch MLP and build an ensemble of 5.

class MLPClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 5) # 5 classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# reset_params=True will re-initialise the network weights per member to ensure diversity.
# We specify predictor_type="logit_classifier" to get probabilistic distributions out.
ensemble_predictor = ensemble(MLPClassifier(), num_members=5, reset_params=True, predictor_type="logit_classifier")


# Train each member on the dataset for 10 epochs.
loss_fn = nn.CrossEntropyLoss()
optimizers = []
member_indices = []

for member in ensemble_predictor:
    optimizer = torch.optim.Adam(member.parameters(), lr=0.01)
    optimizers.append(optimizer)
    member_idx = torch.randint(0, len(X_train), (len(X_train),)) # Bagging
    member_indices.append(member_idx)
    X_bag, y_bag = X_train[member_idx], y_train[member_idx]

    for _ in range(10):
        optimizer.zero_grad()
        out = member(X_bag)
        loss = loss_fn(out, y_bag)
        loss.backward()
        optimizer.step()

# Evaluate the state after 10 epochs
output_epoch_10 = representer(ensemble_predictor).predict(X_test)
variance_epoch_10 = OrdinalVarianceDecomposition(output_epoch_10)
entropy_epoch_10 = OrdinalEntropyDecomposition(output_epoch_10)

total_unc_epoch_10 = variance_epoch_10.total
aleatoric_unc_epoch_10 = variance_epoch_10.aleatoric
epistemic_unc_epoch_10 = variance_epoch_10.epistemic

total_entropy_epoch_10 = entropy_epoch_10.total
aleatoric_entropy_epoch_10 = entropy_epoch_10.aleatoric
epistemic_entropy_epoch_10 = entropy_epoch_10.epistemic

# Calculate the expected class dynamically (BMA probability mean)
bma_probs_epoch_10 = torch.mean(output_epoch_10.tensor.probabilities, dim=output_epoch_10.sample_axis)
classes = torch.arange(5, dtype=torch.float32)
expected_class_epoch_10 = torch.sum(bma_probs_epoch_10 * classes, dim=-1).detach().numpy()


# Train each member on the dataset for another 90 epochs (to reach 100 epochs total)
for i, member in enumerate(ensemble_predictor):
    optimizer = optimizers[i]
    X_bag, y_bag = X_train[member_indices[i]], y_train[member_indices[i]]

    for _ in range(90):
        optimizer.zero_grad()
        out = member(X_bag)
        loss = loss_fn(out, y_bag)
        loss.backward()
        optimizer.step()

# %%
# Represent and Quantify
# ----------------------
# We wrap the ensemble predictor to generate our second-order representation.
output_epoch_100 = representer(ensemble_predictor).predict(X_test)

# Quantify total, aleatoric, and epistemic uncertainty via OrdinalVarianceDecomposition
variance_epoch_100 = OrdinalVarianceDecomposition(output_epoch_100)
entropy_epoch_100 = OrdinalEntropyDecomposition(output_epoch_100)

total_unc_epoch_100 = variance_epoch_100.total
aleatoric_unc_epoch_100 = variance_epoch_100.aleatoric
epistemic_unc_epoch_100 = variance_epoch_100.epistemic

total_entropy_epoch_100 = entropy_epoch_100.total
aleatoric_entropy_epoch_100 = entropy_epoch_100.aleatoric
epistemic_entropy_epoch_100 = entropy_epoch_100.epistemic

# %%
# Results
# -------

# Calculate predictive mean (expected class)
bma_probs_epoch_100 = torch.mean(output_epoch_100.tensor.probabilities, dim=output_epoch_100.sample_axis)
expected_class_epoch_100 = torch.sum(bma_probs_epoch_100 * classes, dim=-1).detach().numpy()

# Sort for plotting
order = np.argsort(X_test.numpy().ravel())
X_plot = X_test.numpy().ravel()[order]

# Plots for 10 epochs
mean_plot_epoch_10 = expected_class_epoch_10[order]
tot_plot_epoch_10 = total_unc_epoch_10.squeeze().detach().numpy()[order]
ale_plot_epoch_10 = aleatoric_unc_epoch_10.squeeze().detach().numpy()[order]
epi_plot_epoch_10 = epistemic_unc_epoch_10.squeeze().detach().numpy()[order]
std_total_epoch_10 = np.sqrt(tot_plot_epoch_10)

tot_ent_epoch_10 = total_entropy_epoch_10.squeeze().detach().numpy()[order]
ale_ent_epoch_10 = aleatoric_entropy_epoch_10.squeeze().detach().numpy()[order]
epi_ent_epoch_10 = epistemic_entropy_epoch_10.squeeze().detach().numpy()[order]


# Plots for 100 epochs
mean_plot_epoch_100 = expected_class_epoch_100[order]
tot_plot_epoch_100 = total_unc_epoch_100.squeeze().detach().numpy()[order]
ale_plot_epoch_100 = aleatoric_unc_epoch_100.squeeze().detach().numpy()[order]
epi_plot_epoch_100 = epistemic_unc_epoch_100.squeeze().detach().numpy()[order]
std_total_epoch_100 = np.sqrt(tot_plot_epoch_100)

tot_ent_epoch_100 = total_entropy_epoch_100.squeeze().detach().numpy()[order]
ale_ent_epoch_100 = aleatoric_entropy_epoch_100.squeeze().detach().numpy()[order]
epi_ent_epoch_100 = epistemic_entropy_epoch_100.squeeze().detach().numpy()[order]

fig, axs = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
ax1_epoch_10, ax1_epoch_100 = axs[0, 0], axs[0, 1]
ax2_epoch_10, ax2_epoch_100 = axs[1, 0], axs[1, 1]
ax3_epoch_10, ax3_epoch_100 = axs[2, 0], axs[2, 1]

# Compute y-limits for the decompositions (shared across measures, separate for states)
max_decomp_y_epoch_10 = max(
    tot_plot_epoch_10.max(), tot_ent_epoch_10.max()
) * 1.05
max_decomp_y_epoch_100 = max(
    tot_plot_epoch_100.max(), tot_ent_epoch_100.max()
) * 1.05

min_class_y = min(
    (mean_plot_epoch_10 - 2 * std_total_epoch_10).min(),
    (mean_plot_epoch_100 - 2 * std_total_epoch_100).min(),
    y_train.numpy().min(), y_test.numpy().min()
) - 0.5
max_class_y = max(
    (mean_plot_epoch_10 + 2 * std_total_epoch_10).max(),
    (mean_plot_epoch_100 + 2 * std_total_epoch_100).max(),
    y_train.numpy().max(), y_test.numpy().max()
) + 0.5

def plot_expected_class(ax, mean_plot, std_total, title_prefix):
    ax.scatter(X_train.numpy().ravel(), y_train.numpy(), color="gray", alpha=0.5, label="Train data")
    ax.scatter(X_plot, y_test.numpy()[order], color="tab:red", label="Test data")
    ax.plot(X_plot, mean_plot, color="tab:blue", linewidth=2, label="Expected Class")
    ax.fill_between(X_plot, mean_plot - 2*std_total, mean_plot + 2*std_total, color="tab:blue", alpha=0.2, label="Total Uncertainty (scaled)")
    ax.set_ylabel("Class")
    ax.set_ylim(min_class_y, max_class_y)
    ax.legend(loc="upper left")
    ax.set_title(f"{title_prefix}: Expected Class and Uncertainty")

def plot_decomposition(ax, tot_plot, epi_plot, ale_plot, ylabel, title_prefix, max_y):
    ax.plot(X_plot, tot_plot, color="tab:purple", linewidth=2, label="Total")
    ax.plot(X_plot, epi_plot, color="tab:green", linewidth=2, linestyle="--", label="Epistemic")
    ax.plot(X_plot, ale_plot, color="tab:orange", linewidth=2, linestyle=":", label="Aleatoric")
    ax.fill_between(X_plot, 0, epi_plot, color="tab:green", alpha=0.2)
    ax.fill_between(X_plot, epi_plot, tot_plot, color="tab:orange", alpha=0.2)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max_y)
    ax.legend(loc="upper left")
    ax.set_title(f"{title_prefix}: {ylabel} Decomposition")

plot_expected_class(ax1_epoch_10, mean_plot_epoch_10, std_total_epoch_10, "10 Epochs")
plot_expected_class(ax1_epoch_100, mean_plot_epoch_100, std_total_epoch_100, "100 Epochs")

plot_decomposition(ax2_epoch_10, tot_plot_epoch_10, epi_plot_epoch_10, ale_plot_epoch_10, "Ordinal Variance", "10 Epochs", max_decomp_y_epoch_10)
plot_decomposition(ax2_epoch_100, tot_plot_epoch_100, epi_plot_epoch_100, ale_plot_epoch_100, "Ordinal Variance", "100 Epochs", max_decomp_y_epoch_100)

plot_decomposition(ax3_epoch_10, tot_ent_epoch_10, epi_ent_epoch_10, ale_ent_epoch_10, "Ordinal Entropy", "10 Epochs", max_decomp_y_epoch_10)
plot_decomposition(ax3_epoch_100, tot_ent_epoch_100, epi_ent_epoch_100, ale_ent_epoch_100, "Ordinal Entropy", "100 Epochs", max_decomp_y_epoch_100)

ax3_epoch_10.set_xlabel("X")
ax3_epoch_100.set_xlabel("X")

fig.tight_layout()
plt.show()

# Print specific point values
print("--- After 10 Epochs ---")
print("Variance Decomp:")
print(f"  Total Range:     [{total_unc_epoch_10.min().item():.4f}, {total_unc_epoch_10.max().item():.4f}]")
print(f"  Epistemic Range: [{epistemic_unc_epoch_10.min().item():.4f}, {epistemic_unc_epoch_10.max().item():.4f}]")
print(f"  Aleatoric Range: [{aleatoric_unc_epoch_10.min().item():.4f}, {aleatoric_unc_epoch_10.max().item():.4f}]")
print("Entropy Decomp:")
print(f"  Total Range:     [{total_entropy_epoch_10.min().item():.4f}, {total_entropy_epoch_10.max().item():.4f}]")
print(f"  Epistemic Range: [{epistemic_entropy_epoch_10.min().item():.4f}, {epistemic_entropy_epoch_10.max().item():.4f}]")
print(f"  Aleatoric Range: [{aleatoric_entropy_epoch_10.min().item():.4f}, {aleatoric_entropy_epoch_10.max().item():.4f}]")
print("\n--- After 100 Epochs ---")
print("Variance Decomp:")
print(f"  Total Range:     [{total_unc_epoch_100.min().item():.4f}, {total_unc_epoch_100.max().item():.4f}]")
print(f"  Epistemic Range: [{epistemic_unc_epoch_100.min().item():.4f}, {epistemic_unc_epoch_100.max().item():.4f}]")
print(f"  Aleatoric Range: [{aleatoric_unc_epoch_100.min().item():.4f}, {aleatoric_unc_epoch_100.max().item():.4f}]")
print("Entropy Decomp:")
print(f"  Total Range:     [{total_entropy_epoch_100.min().item():.4f}, {total_entropy_epoch_100.max().item():.4f}]")
print(f"  Epistemic Range: [{epistemic_entropy_epoch_100.min().item():.4f}, {epistemic_entropy_epoch_100.max().item():.4f}]")
print(f"  Aleatoric Range: [{aleatoric_entropy_epoch_100.min().item():.4f}, {aleatoric_entropy_epoch_100.max().item():.4f}]")
