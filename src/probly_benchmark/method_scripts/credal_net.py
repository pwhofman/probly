"""Credal Net benchmark on MNIST using selective prediction."""

from __future__ import annotations

from typing import TYPE_CHECKING

from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn

from probly.evaluation.tasks import selective_prediction
from probly.method.credal_net import credal_net
from probly.quantification.classification import upper_entropy
from probly.representation.credal_set.torch import TorchProbabilityIntervalsCredalSet
from probly_benchmark import utils
from probly_benchmark.method_scripts import data
from probly_benchmark.models import LeNet

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BATCH_SIZE = 128
NUM_EPOCHS = 5
LR = 1e-3
NUM_CLASSES = 10
N_BINS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(model: nn.Module, loader: DataLoader, epochs: int = NUM_EPOCHS, lr: float = LR, delta: float = 0.5) -> None:
    """Train model using NLLLoss on the upper credal probabilities."""
    model.to(DEVICE).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    no_red_criterion = nn.NLLLoss(reduction="none")
    criterion = nn.NLLLoss()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for x, y in loader:
            optimizer.zero_grad()
            output = model(x.to(DEVICE))
            hi = output[:, NUM_CLASSES:]
            lo = output[:, :NUM_CLASSES]
            loss_up = criterion(hi.log(), y.to(DEVICE))
            loss_lo = no_red_criterion(lo.log(), y.to(DEVICE))

            # Select top delta * batch_size samples with highest loss for backward
            loss_lo_sort, _ = torch.sort(loss_lo, descending=True, dim=-1)

            bound_index = int(np.floor(delta * y.shape[0])) - 1
            bound_value = loss_lo_sort[bound_index]

            choose_index = torch.greater_equal(loss_lo, bound_value)
            choose_outputs_lo = lo[choose_index]
            choose_targets = y[choose_index].to(DEVICE)

            loss_lo_mod = criterion(choose_outputs_lo.log(), choose_targets)
            loss = loss_lo_mod + loss_up
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
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
    ax.plot(rejection_rates, accs, marker="o", markersize=3, linewidth=1.5, label=f"Credal Net (AUROC={auroc:.3f})")
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
    ax.set_title("Accuracy-rejection curve (MNIST, Credal Net)")
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

    print("Building Credal Net...")
    credal_model = credal_net(base_model)

    print("Training...")
    train(credal_model, train_loader)

    print("Predicting...")
    credal_model.eval()
    all_lo: list[torch.Tensor] = []
    all_hi: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    with torch.no_grad():
        for x, y in test_loader:
            output = credal_model(x.to(DEVICE))
            all_lo.append(output[:, :NUM_CLASSES])
            all_hi.append(output[:, NUM_CLASSES:])
            all_labels.append(y)
    lo = torch.cat(all_lo)
    hi = torch.cat(all_hi)
    labels = torch.cat(all_labels)
    cset = TorchProbabilityIntervalsCredalSet(lower_bounds=lo, upper_bounds=hi)

    print("Evaluating selective prediction...")
    preds = ((lo + hi) / 2).argmax(dim=1)
    accs = (preds == labels).numpy().astype(float)
    # upper entropy as criterion
    probs = np.array(cset)
    criterion_vals = upper_entropy(probs, n_jobs=-1)
    auroc, bin_losses = selective_prediction(criterion_vals, accs, n_bins=N_BINS)
    random_criterion = np.random.default_rng(seed).permutation(len(accs)).astype(float)
    random_auroc, random_bin_losses = selective_prediction(random_criterion, accs, n_bins=N_BINS)
    baseline_acc = accs.mean()
    print(f"Baseline accuracy : {baseline_acc:.4f}")
    print(f"Selective pred AUROC: {auroc:.4f}")
    print(f"Random rejection AUROC: {random_auroc:.4f}")

    plot_arc(bin_losses, auroc, random_bin_losses, random_auroc)


if __name__ == "__main__":
    main()
