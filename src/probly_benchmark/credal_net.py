"""Credal Net benchmark on MNIST using selective prediction."""

from __future__ import annotations

import random
import ssl

import joblib
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import entropy
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import tqdm

from probly.evaluation.tasks import selective_prediction
from probly.method.credal_net import credal_net
from probly.representation.credal_set.torch import TorchProbabilityIntervalsCredalSet
from probly_benchmark.models import LeNet

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def upper_entropy(probs: np.ndarray, base: float = 2, n_jobs: int | None = None) -> np.ndarray:
    """Compute the upper entropy of a credal set.

    Given the probs array the lower and upper probabilities are computed and the credal set is
    assumed to be a convex set including all probability distributions in the interval [lower, upper]
    for all classes. The upper entropy of this set is computed.

    Args:
        probs: Probability distributions of shape (n_instances, n_samples, n_classes).
        base: Base of the logarithm. Defaults to 2.
        n_jobs: Number of jobs for joblib.Parallel. Defaults to None. If -1, all available cores are used.

    Returns:
        ue: Upper entropy values of shape (n_instances,).
    """
    x0 = probs.mean(axis=1)
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}

    def compute_upper_entropy(i: int) -> float:
        def fun(x: np.ndarray) -> np.ndarray:
            return -entropy(x, base=base)

        bounds = list(zip(np.min(probs[i], axis=0), np.max(probs[i], axis=0), strict=False))
        res = minimize(fun=fun, x0=x0[i], bounds=bounds, constraints=constraints)
        return float(-res.fun)

    if n_jobs:
        ue = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(compute_upper_entropy)(i)
            for i in tqdm(range(probs.shape[0]), desc="Instances")  # ty:ignore[call-non-callable]
        )
        ue = np.array(ue)
    else:
        ue = np.empty(probs.shape[0])
        for i in tqdm(range(probs.shape[0]), desc="Instances"):  # ty:ignore[call-non-callable]
            ue[i] = compute_upper_entropy(i)
    return ue


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
def train(model: nn.Module, loader: DataLoader, epochs: int = NUM_EPOCHS, lr: float = LR) -> None:
    """Train model using NLLLoss on the upper credal probabilities."""
    model.to(DEVICE).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.NLLLoss()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for x, y in loader:
            optimizer.zero_grad()
            output = model(x.to(DEVICE))
            hi = output[:, NUM_CLASSES:]
            loss = criterion(hi.log(), y.to(DEVICE))
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
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print("Loading MNIST...")
    ssl._create_default_https_context = ssl._create_unverified_context  # ty:ignore[invalid-assignment]  # noqa: SLF001
    tf = transforms.ToTensor()
    train_data = datasets.MNIST("~/.cache/mnist", train=True, download=True, transform=tf)
    test_data = datasets.MNIST("~/.cache/mnist", train=False, download=True, transform=tf)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

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
