"""==========================================
Ensemble Regression Uncertainty
==========================================

Demonstrate :func:`~probly.method.ensemble.ensemble`
using a PyTorch multi-layer perceptron regressor on a dummy dataset.

This example computes epistemic, aleatoric, and total variance uncertainty
of the expected predictive distribution.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from probly.method.ensemble import ensemble
from probly.representer import representer
from probly.quantification import quantify
from probly.quantification.decomposition.variance import SecondOrderVarianceDecomposition

# %%
# Data preparation
# ----------------
# We create a simple noisy sine wave dataset.

X_np = np.linspace(-3, 3, 100).reshape(-1, 1).astype(np.float32)
y_np = (np.sin(X_np) + np.random.normal(0, 0.1, X_np.shape)).astype(np.float32)

# Split
idx = np.random.permutation(len(X_np))
X_train, X_test = torch.from_numpy(X_np[idx[:80]]), torch.from_numpy(X_np[idx[80:]])
y_train, y_test = torch.from_numpy(y_np[idx[:80]]), torch.from_numpy(y_np[idx[80:]])

# %%
# Build and train the ensemble
# -------------------------
# We use a simple PyTorch MLP and build an ensemble of 5.

class MLPRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# reset_params=True will re-initialise the network weights per member to ensure diversity.
ensemble_predictor = ensemble(MLPRegressor(), num_members=5, reset_params=True)

# Evaluate the untrained state for comparison
output_pre = representer(ensemble_predictor).predict(X_test)
unc_pre = SecondOrderVarianceDecomposition(output_pre)

total_unc_pre = unc_pre.total
aleatoric_unc_pre = unc_pre.aleatoric
epistemic_unc_pre = unc_pre.epistemic
bma_mean_pre = torch.mean(output_pre.tensor, dim=output_pre.sample_axis).squeeze().detach().numpy()

# Train each member on the dataset.
loss_fn = nn.MSELoss()
for member in ensemble_predictor:
    optimizer = torch.optim.Adam(member.parameters(), lr=0.01)
    member_idx = torch.randint(0, len(X_train), (len(X_train),)) # Bagging
    X_bag, y_bag = X_train[member_idx], y_train[member_idx]

    for _ in range(200):
        optimizer.zero_grad()
        out = member(X_bag)
        loss = loss_fn(out, y_bag)
        loss.backward()
        optimizer.step()

# %%
# Represent and Quantify
# ----------------------
# We wrap the ensemble predictor to generate our second-order representation.
# For PyTorch point regressors, the representer gives empirical distribution samples (Dirac deltas),
# which act as degenerate Gaussians with 0 variance.
output_post = representer(ensemble_predictor).predict(X_test)

# Quantify total, aleatoric, and epistemic uncertainty via VarianceDecomposition
unc_post = SecondOrderVarianceDecomposition(output_post)

total_unc_post = unc_post.total
aleatoric_unc_post = unc_post.aleatoric
epistemic_unc_post = unc_post.epistemic

# %%
# Results
# -------

# Calculate predictive mean
bma_mean_post = torch.mean(output_post.tensor, dim=output_post.sample_axis).squeeze().detach().numpy()

# Sort for plotting
order = np.argsort(X_test.numpy().ravel())
X_plot = X_test.numpy().ravel()[order]

# Pre-train plots
mean_plot_pre = bma_mean_pre[order]
tot_plot_pre = total_unc_pre.squeeze().detach().numpy()[order]
ale_plot_pre = aleatoric_unc_pre.squeeze().detach().numpy()[order]
epi_plot_pre = epistemic_unc_pre.squeeze().detach().numpy()[order]
std_total_pre = np.sqrt(tot_plot_pre)

# Post-train plots
mean_plot_post = bma_mean_post[order]
tot_plot_post = total_unc_post.squeeze().detach().numpy()[order]
ale_plot_post = aleatoric_unc_post.squeeze().detach().numpy()[order]
epi_plot_post = epistemic_unc_post.squeeze().detach().numpy()[order]
std_total_post = np.sqrt(tot_plot_post)

fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
ax1_pre, ax1_post = axs[0, 0], axs[0, 1]
ax2_pre, ax2_post = axs[1, 0], axs[1, 1]

# Compute shared y-limits to directly see the magnitude difference
max_decomp_y = max(tot_plot_pre.max(), tot_plot_post.max()) * 1.05

min_y = min(
    (mean_plot_pre - 2 * std_total_pre).min(),
    (mean_plot_post - 2 * std_total_post).min(),
    y_train.numpy().min(), y_test.numpy().min()
) - 0.5
max_y = max(
    (mean_plot_pre + 2 * std_total_pre).max(),
    (mean_plot_post + 2 * std_total_post).max(),
    y_train.numpy().max(), y_test.numpy().max()
) + 0.5

# Top-Left plot: Untrained Mean and CI
ax1_pre.scatter(X_train.numpy().ravel(), y_train.numpy(), color="gray", alpha=0.5, label="Train data")
ax1_pre.scatter(X_plot, y_test.numpy()[order], color="tab:red", label="Test data")
ax1_pre.plot(X_plot, mean_plot_pre, color="tab:blue", linewidth=2, label="Ensemble Mean")
ax1_pre.fill_between(X_plot, mean_plot_pre - 2*std_total_pre, mean_plot_pre + 2*std_total_pre, color="tab:blue", alpha=0.2, label="Total Uncertainty (2 std)")
ax1_pre.set_ylabel("y")
ax1_pre.set_ylim(min_y, max_y)
ax1_pre.legend(loc="upper left")
ax1_pre.set_title("Untrained: Mean and Confidence Interval")

# Bottom-Left plot: Untrained Variance Decomposition
ax2_pre.plot(X_plot, tot_plot_pre, color="tab:purple", linewidth=2, label="Total Variance")
ax2_pre.plot(X_plot, epi_plot_pre, color="tab:green", linewidth=2, linestyle="--", label="Epistemic Variance")
ax2_pre.plot(X_plot, ale_plot_pre, color="tab:orange", linewidth=2, linestyle=":", label="Aleatoric Variance")
ax2_pre.fill_between(X_plot, 0, epi_plot_pre, color="tab:green", alpha=0.2)
ax2_pre.fill_between(X_plot, epi_plot_pre, tot_plot_pre, color="tab:orange", alpha=0.2)
ax2_pre.set_xlabel("X")
ax2_pre.set_ylabel("Variance")
ax2_pre.set_ylim(0, max_decomp_y)
ax2_pre.legend(loc="upper left")
ax2_pre.set_title("Untrained: Variance Decomposition")

# Top-Right plot: Trained Mean and CI
ax1_post.scatter(X_train.numpy().ravel(), y_train.numpy(), color="gray", alpha=0.5, label="Train data")
ax1_post.scatter(X_plot, y_test.numpy()[order], color="tab:red", label="Test data")
ax1_post.plot(X_plot, mean_plot_post, color="tab:blue", linewidth=2, label="Ensemble Mean")
ax1_post.fill_between(X_plot, mean_plot_post - 2*std_total_post, mean_plot_post + 2*std_total_post, color="tab:blue", alpha=0.2, label="Total Uncertainty (2 std)")
ax1_post.set_ylabel("y")
ax1_post.set_ylim(min_y, max_y)
ax1_post.legend(loc="upper left")
ax1_post.set_title("Trained: Mean and Confidence Interval")

# Bottom-Right plot: Trained Variance Decomposition
ax2_post.plot(X_plot, tot_plot_post, color="tab:purple", linewidth=2, label="Total Variance")
ax2_post.plot(X_plot, epi_plot_post, color="tab:green", linewidth=2, linestyle="--", label="Epistemic Variance")
ax2_post.plot(X_plot, ale_plot_post, color="tab:orange", linewidth=2, linestyle=":", label="Aleatoric Variance")
ax2_post.fill_between(X_plot, 0, epi_plot_post, color="tab:green", alpha=0.2)
ax2_post.fill_between(X_plot, epi_plot_post, tot_plot_post, color="tab:orange", alpha=0.2)
ax2_post.set_xlabel("X")
ax2_post.set_ylabel("Variance")
ax2_post.set_ylim(0, max_decomp_y)
ax2_post.legend(loc="upper left")
ax2_post.set_title("Trained: Variance Decomposition")

fig.tight_layout()
plt.show()

# Print specific point values
print("--- Before Training ---")
print(f"Total Variance Range: [{total_unc_pre.min().item():.4f}, {total_unc_pre.max().item():.4f}]")
print(f"Epistemic Variance Range: [{epistemic_unc_pre.min().item():.4f}, {epistemic_unc_pre.max().item():.4f}]")
print(f"Aleatoric Variance Range: [{aleatoric_unc_pre.min().item():.4f}, {aleatoric_unc_pre.max().item():.4f}]")
print("\n--- After Training ---")
print(f"Total Variance Range: [{total_unc_post.min().item():.4f}, {total_unc_post.max().item():.4f}]")
print(f"Epistemic Variance Range: [{epistemic_unc_post.min().item():.4f}, {epistemic_unc_post.max().item():.4f}]")
print(f"Aleatoric Variance Range: [{aleatoric_unc_post.min().item():.4f}, {aleatoric_unc_post.max().item():.4f}]")
