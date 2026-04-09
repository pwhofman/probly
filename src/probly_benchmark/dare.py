"""DARE benchmark on MNIST using selective prediction."""

from __future__ import annotations

from typing import TYPE_CHECKING

from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import entropy
import torch
from torch import nn

from probly.evaluation.tasks import selective_prediction
from probly.layers.torch import DAREWrapper  # Import DAREWrapper to check for its instances
from probly.method.dare import dare
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
DELTA = 0.05
NUM_MEMBERS = 5
NUM_SAMPLES = 1
N_BINS = 50
ANTI_REG_WEIGHT = 1e-4  # weight for the anti-regularization loss
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
            output = model(x.to(DEVICE))

            # expand batch to (m * b)
            y_expanded = y.to(DEVICE).repeat(NUM_MEMBERS)

            nll_loss = criterion(output.log(), y_expanded)

            # calculate anti-regularization loss
            anti_reg_loss = torch.tensor(0.0, device=DEVICE)
            for module in model.modules():
                if isinstance(module, DAREWrapper):
                    anti_reg_loss += module.get_anti_reg_loss()

            loss = nll_loss - ANTI_REG_WEIGHT * anti_reg_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y_expanded)
        n = len(loader.dataset) * NUM_MEMBERS
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
    ax.plot(rejection_rates, accs, marker="o", markersize=3, linewidth=1.5, label=f"DARE (AUROC={auroc:.3f})")
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
    ax.set_title("Accuracy-rejection curve (MNIST, DARE)")
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
    base_model = LeNet()

    print("Building DARE...")
    dare_model = dare(base_model, delta=DELTA, num_members=NUM_MEMBERS)

    print("Training...")
    train(dare_model, train_loader)

    print("Predicting...")
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    dare_model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            output = dare_model(x.to(DEVICE))  # (m * b, c)

            m = NUM_MEMBERS
            b = x.shape[0]
            c = output.shape[-1]

            # align ensemble members and samples
            # reshape to (member, batch, channel) then transpose to (batch, member, channel)
            output = output.view(m, b, c).transpose(0, 1)

            all_probs.append(output.cpu().numpy())
            all_labels.append(y.numpy())

    probs = np.concatenate(all_probs, axis=0)  # (total_samples, m, c)
    labels = np.concatenate(all_labels, axis=0)
    print(f"Probs tensor shape: {probs.shape}")

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
