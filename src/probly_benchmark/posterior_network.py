"""Benchmarking a posterior network implementation on a simple model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn

from probly.evaluation.tasks import selective_prediction
from probly.method.posterior_network import posterior_network
from probly.quantification.classification import evidential_uncertainty
from probly.train.evidential.torch import postnet_loss
from probly_benchmark import data, utils
from probly_benchmark.models import LeNet

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BATCH_SIZE = 128
NUM_EPOCHS = 5
LR = 1e-3
N_BINS = 50
DEVICE = utils.get_device()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(model: nn.Module, loader: DataLoader, epochs: int = NUM_EPOCHS, lr: float = LR) -> None:
    """Train model with ... loss."""
    model.to(DEVICE).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = postnet_loss

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for x, y in loader:
            optimizer.zero_grad()
            loss = criterion(model(x.to(DEVICE)).log(), y.to(DEVICE))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y.to(DEVICE))
        n = len(loader.dataset)  # ty:ignore[invalid-argument-type]
        print(f"Epoch {epoch}/{epochs}  loss={total_loss / n:.4f}")


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def plot_arc(
    accs: np.ndarray,
    auroc: float,
    random_accs: np.ndarray,
    random_auroc: float,
    n_bins: int = N_BINS,
) -> None:
    """Plot accuracy-rejection curve."""
    rejection_rates = np.linspace(0, 1, n_bins)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        rejection_rates, accs, marker="o", markersize=3, linewidth=1.5, label=f"Posterior Network (AUROC={auroc:.3f})"
    )
    ax.plot(
        rejection_rates,
        random_accs,
        linestyle="--",
        linewidth=1,
        color="grey",
        label=f"Random rejection (AUROC={random_auroc:.3f})",
    )
    ax.set_xlabel("Rejection rate")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy-rejection curve (MNIST, Posterior Network)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(seed: int = 0) -> None:
    """Run the full benchmark pipeline."""
    utils.set_seed(seed)

    print("Loading MNIST...")
    train_loader, test_loader = data.load_mnist(BATCH_SIZE)

    print("Building model...")
    base_model = LeNet().to(DEVICE)

    # compute the counts per class in the MNIST training set
    class_counts = np.unique_counts(train_loader.dataset.targets.tolist()).counts.tolist()  # ty: ignore

    print("Building Dropout...")
    postnet_model = posterior_network(base_model, dim=16, num_classes=10, class_counts=class_counts, num_flows=2).to(  # ty: ignore
        DEVICE
    )

    print("Training...")
    train(postnet_model, train_loader)

    print("Predicting...")
    all_alphas: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    with torch.no_grad():
        for x, y in test_loader:
            alpha = postnet_model(x.to(DEVICE))
            all_alphas.append(alpha.tensor.cpu().numpy())
            all_labels.append(y.numpy())
    alphas = np.array(all_alphas)
    labels = np.array(all_labels)
    print(f"Sample tensor shape: {alphas.shape}")

    print("Evaluating selective prediction...")
    criterion = evidential_uncertainty(alphas)
    accs = (alphas.argmax(axis=1) == labels).astype(float)
    auroc, bin_losses = selective_prediction(criterion, accs, n_bins=N_BINS)
    random_criterion = np.random.default_rng(seed).permutation(len(accs)).astype(float)
    random_auroc, random_bin_losses = selective_prediction(random_criterion, accs, n_bins=N_BINS)
    baseline_acc = accs.mean()
    print(f"Baseline accuracy : {baseline_acc:.4f}")
    print(f"Selective pred AUROC: {auroc:.4f}")
    print(f"Random rejection AUROC: {random_auroc:.4f}")

    plot_arc(bin_losses, auroc, random_bin_losses, random_auroc)


if __name__ == "__main__":
    main()
