from __future__ import annotations

from types import ModuleType

import torch
import numpy as np
import matplotlib.pyplot as plt

from probly.quantification import quantify


def plot_mnist_uncertainty(
    images_test,
    y_test,
    uncertainty,
    mean_probs,
    member_probs=None,
    is_ensemble: bool = False,
    title: str = "Most Uncertain Test Predictions",
    n_top: int = 5,
    unit: str = "bits",
) -> ModuleType:
    num_classes = mean_probs.shape[1]
    top_idx = uncertainty.argsort()[-n_top:][::-1]
    preds = mean_probs.argmax(axis=-1)

    tab_colors = plt.cm.tab10.colors

    fig, axes = plt.subplots(2, n_top, figsize=(n_top * 2.4, 5))
    fig.suptitle(title)

    for col, idx in enumerate(top_idx):
        img = images_test[idx]
        if hasattr(img, "numpy"):
            img = img.numpy()

        axes[0, col].imshow(img, cmap="gray")
        axes[0, col].set_title(
            f"True: {int(y_test[idx])} | Pred: {int(preds[idx])}\n"
            f"U = {uncertainty[idx]:.2f} {unit}"
        )
        axes[0, col].axis("off")

        ax = axes[1, col]
        if is_ensemble and member_probs is not None:
            num_members = member_probs.shape[0]
            for m in range(num_members):
                ax.plot(
                    range(num_classes),
                    member_probs[m, idx],
                    color=tab_colors[m % len(tab_colors)],
                    marker=".",
                    label=f"Member {m + 1}" if col == 0 else None,
                )
            if col == 0:
                ax.legend(loc="upper right", fontsize=7)
        else:
            ax.bar(range(num_classes), mean_probs[idx], color="steelblue")

        ax.set_xticks(range(num_classes))
        ax.set_xlabel("class")
        if col == 0:
            ax.set_ylabel("p")

    plt.tight_layout(h_pad=3.5)

    if is_ensemble:
        y_row1 = axes[1, 0].get_position().y1
        fig.text(
            0.5, y_row1 + 0.01, "Member Agreement",
            ha="center", va="bottom", fontsize="large",
        )

    return plt


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

        test_unc = np.clip(test_unc, 0, 1)

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
