from __future__ import annotations

from types import ModuleType
from typing import Literal

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FormatStrFormatter

from probly.quantification import quantify


def plot_mnist_uncertainty(
    images_test,
    y_test,
    uncertainty,
    mean_probs,
    title: str = "Most Uncertain Test Predictions",
    n_top: int = 5,
    unit: str = "bits",
) -> ModuleType:
    num_classes = mean_probs.shape[1]
    top_idx = uncertainty.argsort()[-n_top:][::-1]
    preds = mean_probs.argmax(axis=-1)

    tab_colors = plt.cm.tab10.colors

    fig, axes = plt.subplots(1, n_top, figsize=(n_top * 2.4, 5))
    fig.suptitle(title)

    for col, idx in enumerate(top_idx):
        img = images_test[idx]
        if hasattr(img, "numpy"):
            img = img.numpy()

        axes[col].imshow(img, cmap="gray")
        axes[col].set_title(
            f"True: {int(y_test[idx])} | Pred: {int(preds[idx])}\n"
            f"U = {uncertainty[idx]:.2f} {unit}"
        )
        axes[col].axis("off")

    return plt


_GRID_RES = 200


_UncertaintyKind = Literal["total", "epistemic", "aleatoric"]


def _select_uncertainty(decomp, notion: _UncertaintyKind | None) -> torch.Tensor:
    candidates = (notion,) if notion is not None else ("total", "epistemic", "aleatoric")
    for name in candidates:
        value = getattr(decomp, name, None)
        if value is not None:
            return value
    raise AttributeError(f"Decomposition exposes none of {candidates}.")


def plot_example_uncertainty(
    X,
    y,
    rep,
    title: str = "Predictive Uncertainty",
    notion: _UncertaintyKind | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    xlim: tuple[float, float] = (-3.0, 3.0),
    ylim: tuple[float, float] = (-3.0, 3.0),
    log_scale: bool = False,
) -> ModuleType:
    xx, yy = np.meshgrid(
        np.linspace(xlim[0], xlim[1], _GRID_RES),
        np.linspace(ylim[0], ylim[1], _GRID_RES),
    )
    grid = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()

    with torch.no_grad():
        unc = _select_uncertainty(quantify(rep.represent(grid)), notion)
    test_unc = unc.detach().cpu().numpy().reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    if log_scale:
        positive = test_unc[test_unc > 0]
        lo = float(positive.min()) if vmin is None else vmin
        hi = float(test_unc.max()) if vmax is None else vmax
        norm = LogNorm(vmin=lo, vmax=hi)
        levels = np.geomspace(lo, hi, 100)
        contour = ax.contourf(
            xx, yy, np.clip(test_unc, lo, None),
            levels=levels, cmap="viridis", antialiased=True, extend="both", norm=norm,
        )
        cbar = plt.colorbar(contour)
    else:
        lo = float(test_unc.min()) if vmin is None else vmin
        hi = float(test_unc.max()) if vmax is None else vmax
        contour = ax.contourf(
            xx, yy, test_unc,
            levels=np.linspace(lo, hi, 100),
            cmap="viridis", antialiased=True, extend="both",
        )
        cbar = plt.colorbar(contour, ticks=np.linspace(lo, hi, 5))

    notion_label = (notion or "predictive").capitalize()
    cbar.set_label(f"{notion_label} Uncertainty", fontsize=12, fontweight="bold")
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    for label, mask, color in (("Class 0", y == 0, "#ff7f0e"), ("Class 1", y == 1, "#1f77b4")):
        ax.scatter(X[mask, 0], X[mask, 1], color=color, edgecolor="white",
                   linewidths=0.5, s=25, zorder=3, label=label)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.95, edgecolor="black")

    fig.tight_layout()
    return plt
