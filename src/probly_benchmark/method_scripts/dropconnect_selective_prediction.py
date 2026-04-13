"""Dropconnect benchmark on MNIST using selective prediction."""

from __future__ import annotations

from typing import TYPE_CHECKING

from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import entropy
import torch
from torch import nn

from probly.evaluation.tasks import selective_prediction
from probly.method.dropconnect import dropconnect
from probly.representer.sampler import Sampler
from probly_benchmark import data, utils
from probly_benchmark.models import LeNet

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def total_entropy(probs: np.ndarray, base: float = 2) -> np.ndarray:
    """Compute the total entropy as the total uncertainty.

    The computation is based on samples from a second-order distribution.
    Based on :cite:`depewegDecompositionUncertainty2018`.

    Args:
        probs: Probability distributions of shape (n_instances, n_samples, n_classes).
        base: Base of the logarithm.

    Returns:
        Total entropy values of shape (n_instances,).

    """
    te = entropy(probs.mean(axis=1), axis=1, base=base)
    return te


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BATCH_SIZE = 128
NUM_EPOCHS = 5
LR = 1e-3
DROPCONNECT_P = 0.25
NUM_SAMPLES = 50
N_BINS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(model: nn.Module, loader: DataLoader, epochs: int = NUM_EPOCHS, lr: float = LR) -> None:
    """Train model with NLL loss (works with Softmax output)."""
    model.to(DEVICE).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.NLLLoss()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for x, y in loader:
            optimizer.zero_grad()
            # log of softmax output for NLLLoss
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
    ax.plot(rejection_rates, accs, marker="o", markersize=3, linewidth=1.5, label=f"DropConnect (AUROC={auroc:.3f})")
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
    ax.set_title("Accuracy-rejection curve (MNIST, DropConnect)")
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

    print("Building DropConnect...")
    dropconnect_model = dropconnect(base_model, p=DROPCONNECT_P)

    print("Training...")
    train(dropconnect_model, train_loader)

    print("Predicting...")
    sampler = Sampler(dropconnect_model, num_samples=NUM_SAMPLES)
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    with torch.no_grad():
        for x, y in test_loader:
            sample = sampler.predict(x.to(DEVICE))
            all_probs.append(sample.tensor.cpu().numpy())
            all_labels.append(y.numpy())
    probs = np.concatenate(all_probs).swapaxes(1, 2)
    labels = np.concatenate(all_labels)
    print(f"Sample tensor shape: {probs.shape}")

    print("Evaluating selective prediction...")
    criterion = total_entropy(probs)
    mean_probs = probs.mean(axis=1)
    accs = (mean_probs.argmax(axis=1) == labels).astype(float)
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
