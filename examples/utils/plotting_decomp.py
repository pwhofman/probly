import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from probly.quantification import quantify


def plot_example_uncertainty_decomp(
    X,
    X_tensor,
    y,
    rep,
    title: str = "Predictive Uncertainty",
) -> None:
    # 1. INCREASE RESOLUTION: 200x200 for smooth boundaries
    grid_res = 200
    xx, yy = np.meshgrid(np.linspace(-3.0, 3.0, grid_res), np.linspace(-3.0, 3.0, grid_res))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.from_numpy(grid).float()

    with torch.no_grad():
        # Convert from nats (default) to bits to get a [0, 1] scale like the paper
        decomp_grid = quantify(rep.represent(grid_tensor))

        # Manually calculate total uncertainty: Aleatoric + Epistemic
        total_unc = (decomp_grid._aleatoric + decomp_grid._epistemic).numpy() / np.log(2)
        if total_unc.ndim > 1:
            total_unc = total_unc.sum(-1)

    test_unc = total_unc.reshape(xx.shape)
    u_min = np.min(test_unc)
    u_max = np.max(test_unc)

    if u_max - u_min < 1e-6:
        test_unc_norm = np.zeros_like(test_unc)
    else:
        test_unc_norm = (test_unc - u_min) / (u_max - u_min)

    # %%
    # 5. Plot the epistemic uncertainty surface
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)

    levels = np.linspace(0.0, 1.0, 100)
    contour = ax.contourf(xx, yy, test_unc_norm, levels=levels, cmap="viridis", antialiased=True)

    cbar = plt.colorbar(contour, ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_label("Predictive Uncertainty (bits)", fontsize=12, fontweight='bold')

    ax.scatter(X[y == 0, 0], X[y == 0, 1], color="#ff7f0e", edgecolor="white", linewidths=0.5, s=25, zorder=3,
               label="Class 0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], color="#1f77b4", edgecolor="white", linewidths=0.5, s=25, zorder=3,
               label="Class 1")

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)

    # %%
    # 6. Quantitative OOD Evaluation
    rng = np.random.default_rng(42)
    angles = rng.uniform(0, 2 * np.pi, 150)
    radii = np.sqrt(rng.uniform(0, 1, 150))
    X_ood = np.column_stack([
        1.5 + radii * 0.8 * np.cos(angles),
        -2.2 + radii * 0.3 * np.sin(angles)
    ])

    with torch.no_grad():
        decomp_id = quantify(rep.represent(X_tensor))
        unc_id = (decomp_id._aleatoric + decomp_id._epistemic).numpy() / np.log(2)
        unc_id = unc_id.sum(-1) if unc_id.ndim > 1 else unc_id

        decomp_ood = quantify(rep.represent(torch.from_numpy(X_ood).float()))
        unc_ood = (decomp_ood._aleatoric + decomp_ood._epistemic).numpy() / np.log(2)
        unc_ood = unc_ood.sum(-1) if unc_ood.ndim > 1 else unc_ood

    labels = np.concatenate([np.zeros(len(X)), np.ones(len(X_ood))])
    scores = np.concatenate([unc_id, unc_ood])
    auroc = roc_auc_score(labels, scores)

    ax.scatter(X_ood[:, 0], X_ood[:, 1], color="#d62728", marker="x", s=30, alpha=1.0, linewidths=1.5, zorder=4,
               label="OOD Data")

    ax.set_title(f"{title}\n(OOD AUROC: {auroc:.3f})", fontsize=14, fontweight='bold')
    ax.legend(loc="upper right", framealpha=0.95, edgecolor="black")

    fig.tight_layout()

    return plt
