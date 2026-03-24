"""Evidential Deep Learning benchmark on MNIST using selective prediction."""

from __future__ import annotations

from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from probly.evaluation.tasks import selective_prediction
from probly.methods.evidential import EvidentialClassification
from probly.quantification.classification import evidential_uncertainty
from probly.train.evidential.torch import evidential_kl_divergence, evidential_mse_loss
from probly_benchmark import data, utils
from probly_benchmark.models import LeNetEncoder

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BATCH_SIZE = 128
NUM_EPOCHS = 10
LR = 1e-3
N_BINS = 50
ANNEALING_EPOCHS = NUM_EPOCHS // 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
class EDLLoss:
    """Combined MSE + annealed KL loss (Sensoy et al. 2018)."""

    def __init__(self, annealing_epochs: int) -> None:
        """Initialize EDLLoss.

        Args:
            annealing_epochs: Number of epochs over which to anneal the KL term.
        """
        self.annealing_epochs = annealing_epochs
        self._epoch = 0

    def __call__(self, evidence: Tensor, targets: Tensor) -> Tensor:
        """Compute the combined EDL loss."""
        alphas = evidence + 1.0
        lam = min(1.0, self._epoch / max(self.annealing_epochs, 1))
        return evidential_mse_loss(alphas, targets) + lam * evidential_kl_divergence(alphas, targets)

    def step_epoch(self) -> None:
        """Increment epoch counter for annealing schedule."""
        self._epoch += 1


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(model: nn.Module, loader: DataLoader, loss_fn: EDLLoss, epochs: int = NUM_EPOCHS, lr: float = LR) -> None:
    """Train model with EDL loss."""
    model.to(DEVICE).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for x, y in loader:
            optimizer.zero_grad()
            evidence = model(x.to(DEVICE))
            loss = loss_fn(evidence, y.to(DEVICE))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
        n = len(loader.dataset)  # ty:ignore[invalid-argument-type]
        lam = min(1.0, loss_fn._epoch / max(loss_fn.annealing_epochs, 1))  # noqa: SLF001
        print(f"Epoch {epoch}/{epochs}  loss={total_loss / n:.4f}  lambda={lam:.3f}")
        loss_fn.step_epoch()


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
    ax.plot(rejection_rates, accs, marker="o", markersize=3, linewidth=1.5, label=f"Evidential DL (AUROC={auroc:.3f})")
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
    ax.set_title("Accuracy-rejection curve (MNIST, Evidential DL)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(seed: int = 0) -> None:
    """Run the full benchmark pipeline."""
    utils.set_random_seed(seed)

    print("Loading MNIST...")
    train_loader, test_loader = data.load_mnist(batch_size=BATCH_SIZE)

    print("Building model...")
    base_model = LeNetEncoder().to(DEVICE)

    print("Building EvidentialClassification...")
    edl_model = EvidentialClassification(base_model)

    print("Training...")
    loss_fn = EDLLoss(annealing_epochs=ANNEALING_EPOCHS)
    train(edl_model.predictor, train_loader, loss_fn)  # ty:ignore[invalid-argument-type]

    print("Predicting...")
    all_evidence: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    with torch.no_grad():
        for x, y in test_loader:
            evidence = edl_model.predict(x.to(DEVICE))
            all_evidence.append(evidence.cpu().numpy())
            all_labels.append(y.numpy())
    evidence_arr = np.concatenate(all_evidence)
    labels = np.concatenate(all_labels)
    print(f"Evidence shape: {evidence_arr.shape}")

    print("Evaluating selective prediction...")
    criterion = evidential_uncertainty(evidence_arr)
    predicted = evidence_arr.argmax(axis=1)
    accs = (predicted == labels).astype(float)
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
