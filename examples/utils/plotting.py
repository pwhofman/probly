from types import ModuleType

import torch
import numpy as np
import matplotlib.pyplot as plt

from probly.quantification import quantify


def plot_example_uncertainty(
    X,
    y,
    rep,
    title: str = "Predictive Uncertainty",
    vmin: float | None = 0.0,
    vmax: float | None = 1.0,
    xlim: tuple[float, float] = (-3.0, 3.0),
    ylim: tuple[float, float] = (-3.0, 3.0),
) -> ModuleType:
    grid_res = 200
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], grid_res), np.linspace(ylim[0], ylim[1], grid_res))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.from_numpy(grid).float()

    try:
        device = next(rep.predictor.parameters()).device
        grid_tensor = grid_tensor.to(device)
    except (AttributeError, StopIteration):
        pass

    with torch.no_grad():
        decomp = quantify(rep.represent(grid_tensor))
        if hasattr(decomp, "total"):
            unc = decomp.total
        elif hasattr(decomp, "epistemic"):
            unc = decomp.epistemic
        else:
            unc = decomp.aleatoric
        test_unc = unc.cpu().numpy() / np.log(2)
        if test_unc.ndim > 1:
            test_unc = test_unc.sum(-1)

    test_unc = test_unc.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)

    vmin = test_unc.min() if vmin is None else vmin
    vmax = test_unc.max() if vmax is None else vmax
    levels = np.linspace(vmin, vmax, 100)
    contour = ax.contourf(xx, yy, test_unc, levels=levels, cmap="viridis", antialiased=True, extend="both")

    ticks = np.linspace(vmin, vmax, 5)
    cbar = plt.colorbar(contour, ticks=ticks)
    cbar.set_label("Predictive Uncertainty (bits)", fontsize=12, fontweight="bold")

    ax.scatter(X[y == 0, 0], X[y == 0, 1], color="#ff7f0e", edgecolor="white", linewidths=0.5, s=25, zorder=3, label="Class 0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], color="#1f77b4", edgecolor="white", linewidths=0.5, s=25, zorder=3, label="Class 1")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.95, edgecolor="black")

    fig.tight_layout()
    return plt
