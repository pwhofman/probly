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

# Evaluate the untrained state for comparison
output_pre = representer(ensemble_predictor).predict(X_test)
variance_pre = OrdinalVarianceDecomposition(output_pre)
entropy_pre = OrdinalEntropyDecomposition(output_pre)


total_unc_pre = variance_pre.total
aleatoric_unc_pre = variance_pre.aleatoric
epistemic_unc_pre = variance_pre.epistemic

total_entropy_pre = entropy_pre.total
aleatoric_entropy_pre = entropy_pre.aleatoric
epistemic_entropy_pre = entropy_pre.epistemic


# Calculate the expected class dynamically (BMA probability mean)
bma_probs_pre = torch.mean(output_pre.tensor.probabilities, dim=output_pre.sample_axis)
classes = torch.arange(5, dtype=torch.float32)
expected_class_pre = torch.sum(bma_probs_pre * classes, dim=-1).detach().numpy()


# Train each member on the dataset.
loss_fn = nn.CrossEntropyLoss()
for member in ensemble_predictor:
    optimizer = torch.optim.Adam(member.parameters(), lr=0.01)
    member_idx = torch.randint(0, len(X_train), (len(X_train),)) # Bagging
    X_bag, y_bag = X_train[member_idx], y_train[member_idx]

    for _ in range(500):
        optimizer.zero_grad()
        out = member(X_bag)
        loss = loss_fn(out, y_bag)
        loss.backward()
        optimizer.step()

# %%
# Represent and Quantify
# ----------------------
# We wrap the ensemble predictor to generate our second-order representation.
output_post = representer(ensemble_predictor).predict(X_test)

# Quantify total, aleatoric, and epistemic uncertainty via OrdinalVarianceDecomposition
variance_post = OrdinalVarianceDecomposition(output_post)
entropy_post = OrdinalEntropyDecomposition(output_post)

total_unc_post = variance_post.total
aleatoric_unc_post = variance_post.aleatoric
epistemic_unc_post = variance_post.epistemic

total_entropy_post = entropy_post.total
aleatoric_entropy_post = entropy_post.aleatoric
epistemic_entropy_post = entropy_post.epistemic

# %%
# Results
# -------

# Calculate predictive mean (expected class)
bma_probs_post = torch.mean(output_post.tensor.probabilities, dim=output_post.sample_axis)
expected_class_post = torch.sum(bma_probs_post * classes, dim=-1).detach().numpy()

# Sort for plotting
order = np.argsort(X_test.numpy().ravel())
X_plot = X_test.numpy().ravel()[order]

# Pre-train plots
mean_plot_pre = expected_class_pre[order]
tot_plot_pre = total_unc_pre.squeeze().detach().numpy()[order]
ale_plot_pre = aleatoric_unc_pre.squeeze().detach().numpy()[order]
epi_plot_pre = epistemic_unc_pre.squeeze().detach().numpy()[order]
std_total_pre = np.sqrt(tot_plot_pre)

tot_ent_pre = total_entropy_pre.squeeze().detach().numpy()[order]
ale_ent_pre = aleatoric_entropy_pre.squeeze().detach().numpy()[order]
epi_ent_pre = epistemic_entropy_pre.squeeze().detach().numpy()[order]



# Post-train plots
mean_plot_post = expected_class_post[order]
tot_plot_post = total_unc_post.squeeze().detach().numpy()[order]
ale_plot_post = aleatoric_unc_post.squeeze().detach().numpy()[order]
epi_plot_post = epistemic_unc_post.squeeze().detach().numpy()[order]
std_total_post = np.sqrt(tot_plot_post)

tot_ent_post = total_entropy_post.squeeze().detach().numpy()[order]
ale_ent_post = aleatoric_entropy_post.squeeze().detach().numpy()[order]
epi_ent_post = epistemic_entropy_post.squeeze().detach().numpy()[order]

fig, axs = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
ax1_pre, ax1_post = axs[0, 0], axs[0, 1]
ax2_pre, ax2_post = axs[1, 0], axs[1, 1]
ax3_pre, ax3_post = axs[2, 0], axs[2, 1]

# Compute y-limits for the decompositions (shared across measures, separate for states)
max_decomp_y_pre = max(
    tot_plot_pre.max(), tot_ent_pre.max()
) * 1.05
max_decomp_y_post = max(
    tot_plot_post.max(), tot_ent_post.max()
) * 1.05

min_class_y = min(
    (mean_plot_pre - 2 * std_total_pre).min(),
    (mean_plot_post - 2 * std_total_post).min(),
    y_train.numpy().min(), y_test.numpy().min()
) - 0.5
max_class_y = max(
    (mean_plot_pre + 2 * std_total_pre).max(),
    (mean_plot_post + 2 * std_total_post).max(),
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

plot_expected_class(ax1_pre, mean_plot_pre, std_total_pre, "Untrained")
plot_expected_class(ax1_post, mean_plot_post, std_total_post, "Trained")

plot_decomposition(ax2_pre, tot_plot_pre, epi_plot_pre, ale_plot_pre, "Ordinal Variance", "Untrained", max_decomp_y_pre)
plot_decomposition(ax2_post, tot_plot_post, epi_plot_post, ale_plot_post, "Ordinal Variance", "Trained", max_decomp_y_post)

plot_decomposition(ax3_pre, tot_ent_pre, epi_ent_pre, ale_ent_pre, "Ordinal Entropy", "Untrained", max_decomp_y_pre)
plot_decomposition(ax3_post, tot_ent_post, epi_ent_post, ale_ent_post, "Ordinal Entropy", "Trained", max_decomp_y_post)

ax3_pre.set_xlabel("X")
ax3_post.set_xlabel("X")

fig.tight_layout()
plt.show()

# Print specific point values
print("--- Before Training ---")
print("Variance Decomp:")
print(f"  Total Range:     [{total_unc_pre.min().item():.4f}, {total_unc_pre.max().item():.4f}]")
print(f"  Epistemic Range: [{epistemic_unc_pre.min().item():.4f}, {epistemic_unc_pre.max().item():.4f}]")
print(f"  Aleatoric Range: [{aleatoric_unc_pre.min().item():.4f}, {aleatoric_unc_pre.max().item():.4f}]")
print("Entropy Decomp:")
print(f"  Total Range:     [{total_entropy_pre.min().item():.4f}, {total_entropy_pre.max().item():.4f}]")
print(f"  Epistemic Range: [{epistemic_entropy_pre.min().item():.4f}, {epistemic_entropy_pre.max().item():.4f}]")
print(f"  Aleatoric Range: [{aleatoric_entropy_pre.min().item():.4f}, {aleatoric_entropy_pre.max().item():.4f}]")
print("\n--- After Training ---")
print("Variance Decomp:")
print(f"  Total Range:     [{total_unc_post.min().item():.4f}, {total_unc_post.max().item():.4f}]")
print(f"  Epistemic Range: [{epistemic_unc_post.min().item():.4f}, {epistemic_unc_post.max().item():.4f}]")
print(f"  Aleatoric Range: [{aleatoric_unc_post.min().item():.4f}, {aleatoric_unc_post.max().item():.4f}]")
print("Entropy Decomp:")
print(f"  Total Range:     [{total_entropy_post.min().item():.4f}, {total_entropy_post.max().item():.4f}]")
print(f"  Epistemic Range: [{epistemic_entropy_post.min().item():.4f}, {epistemic_entropy_post.max().item():.4f}]")
print(f"  Aleatoric Range: [{aleatoric_entropy_post.min().item():.4f}, {aleatoric_entropy_post.max().item():.4f}]")
