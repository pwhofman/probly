"""Efficient Credal Prediction (EffCRE) benchmark on MNIST using selective prediction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import joblib
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import entropy
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from probly.evaluation.tasks import selective_prediction
from probly.method.efficient_credal_prediction import efficient_credal_prediction
from probly_benchmark import data, utils
from probly_benchmark.models import LeNet

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


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
            joblib.delayed(compute_upper_entropy)(i) for i in tqdm(range(probs.shape[0]), desc="Instances")
        )
        ue = np.array(ue)
    else:
        ue = np.empty(probs.shape[0])
        for i in tqdm(range(probs.shape[0]), desc="Instances"):
            ue[i] = compute_upper_entropy(i)
    return ue


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BATCH_SIZE = 128
NUM_EPOCHS = 1
LR = 1e-3
NUM_CLASSES = 10
N_BINS = 50
ALPHAS = [0.9, 0.7, 0.5, 0.3, 0.1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# EffCRE core
# ---------------------------------------------------------------------------
def log_likelihood(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute the average log-likelihood given model outputs and targets.

    Args:
        outputs: The model outputs (logits) of shape (N, C) with N samples and C classes.
        targets: The target labels of shape (N,).

    Returns:
        A scalar tensor representing the average log-likelihood.
    """
    outputs = F.log_softmax(outputs, dim=1)
    ll = torch.mean(outputs[torch.arange(outputs.shape[0]), targets])
    return ll


def effcre(
    logits_train: torch.Tensor,
    targets_train: torch.Tensor,
    logits_test: torch.Tensor,
    n_classes: int,
    alphas: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute credal sets for test samples based on decalibration (EffCRE)."""
    csets = []
    rls = []
    mll = log_likelihood(logits_train, targets_train).cpu().detach().item()
    for alpha in tqdm(alphas, desc="Alphas"):
        bounds = []
        for k in range(n_classes):
            # 1 is finding minimum, -1 is finding maximum
            bound = []
            for direction in [1, -1]:

                def fun(x: np.ndarray) -> float:
                    return direction * x[k]  # noqa: B023

                def const(x: np.ndarray) -> float:
                    c = torch.tensor(x, device=logits_train.device)
                    logits_train_t = logits_train + c
                    lik = log_likelihood(logits_train_t, targets_train).cpu().detach().item()
                    rel_lik = np.exp(lik - mll)
                    return rel_lik

                x0 = np.zeros(n_classes)
                optim_bounds: list[tuple[float | None, float | None]] = [(0.0, 0.0) for _ in range(n_classes)]
                optim_bounds[k] = (None, None)
                constraints = {"type": "ineq", "fun": lambda x: const(x) - alpha}  # noqa: B023
                res = minimize(fun, x0, constraints=constraints, bounds=optim_bounds)
                bound.append(res.x)
            bounds.append(bound)

        # add the bounds to the logits_test to make predictions
        for k in range(n_classes):
            # both ``directions''
            for d in range(2):
                logits_test_t = logits_test + torch.tensor(bounds[k][d], device=logits_test.device)
                csets.append(F.softmax(logits_test_t, dim=1).cpu().detach().numpy())
        rls.append([alpha] * (2 * n_classes))
    csets = np.array(csets)
    rls = np.array(rls).flatten()
    return csets, rls


# ---------------------------------------------------------------------------
# Logit extraction
# ---------------------------------------------------------------------------
def get_logits(model: LeNet, x: torch.Tensor) -> torch.Tensor:
    """Extract pre-softmax logits from a LeNet by skipping its final Softmax layer."""
    out = model.features(x)
    for layer in list(model.classifier)[:-1]:
        out = layer(out)
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(model: nn.Module, loader: DataLoader, epochs: int = NUM_EPOCHS, lr: float = LR) -> None:
    """Train model with NLL loss (works with Softmax output)."""
    print(model)
    model.to(DEVICE).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for x, y in loader:
            optimizer.zero_grad()
            loss = criterion(model(x.to(DEVICE)), y.to(DEVICE))
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
    ax.plot(rejection_rates, accs, marker="o", markersize=3, linewidth=1.5, label=f"EffCRE (AUROC={auroc:.3f})")
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
    ax.set_title("Accuracy-rejection curve (MNIST, EffCRE)")
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
    model = LeNet().to(DEVICE)
    print(model)
    model = efficient_credal_prediction(model)
    print(model)

    print("Training...")
    train(model, train_loader)

    print("Collecting logits...")
    model.eval()
    all_logits_train: list[torch.Tensor] = []
    all_targets_train: list[torch.Tensor] = []
    all_logits_test: list[torch.Tensor] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for x, y in train_loader:
            all_logits_train.append(model(x.to(DEVICE)))
            all_targets_train.append(y.to(DEVICE))
        for x, y in test_loader:
            all_logits_test.append(model(x.to(DEVICE)))
            all_labels.append(y.numpy())

    logits_train = torch.cat(all_logits_train)
    targets_train = torch.cat(all_targets_train)
    logits_test = torch.cat(all_logits_test)
    labels = np.concatenate(all_labels)

    print("Computing credal sets with EffCRE...")
    csets, _ = effcre(logits_train, targets_train, logits_test, NUM_CLASSES, ALPHAS)
    # csets shape: (n_vertices, N, C) -> transpose to (N, n_vertices, C)
    csets = csets.transpose(1, 0, 2)

    print("Evaluating selective prediction...")
    mean_probs = csets.mean(axis=1)
    accs = (mean_probs.argmax(axis=1) == labels).astype(float)
    criterion_vals = upper_entropy(csets, n_jobs=-1)
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
