"""Training functionality for probly benchmark methods."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn, optim
from torch.amp import GradScaler, autocast
import torch.nn.functional as F

from flextype import flexdispatch
from probly.method.bayesian import BayesianPredictor
from probly.method.credal_ensembling import CredalEnsemblingPredictor
from probly.method.credal_relative_likelihood import CredalRelativeLikelihoodPredictor
from probly.method.credal_wrapper import CredalWrapperPredictor
from probly.method.ddu import DDUPredictor
from probly.method.dropconnect import DropConnectPredictor
from probly.method.dropout import DropoutPredictor
from probly.method.ensemble import EnsemblePredictor
from probly.method.posterior_network import PosteriorNetworkPredictor
from probly.method.subensemble import SubensemblePredictor
from probly.train.bayesian.torch import ELBOLoss, collect_kl_divergence
from probly.train.calibration.torch import ExpectedCalibrationError
from probly.train.evidential.torch import postnet_loss

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from probly.predictor import Predictor


@flexdispatch
def train_epoch(
    model: Predictor,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    optimizer: optim.Optimizer,
    **kwargs: Any,  # noqa: ANN401
) -> torch.Tensor | float:
    """Train for one epoch."""
    msg = f"No training function for {type(model)}"
    raise NotImplementedError(msg)


@train_epoch.register(BayesianPredictor)
def _(
    model: BayesianPredictor,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    optimizer: optim.Optimizer,
    grad_clip_norm: float | None = None,
    amp_enabled: bool = False,
    scaler: GradScaler | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> torch.Tensor | float:
    """Train a Bayesian predictor for one epoch."""
    criterion = ELBOLoss(kl_penalty=kwargs.get("kl_penalty", 1e-5))
    optimizer.zero_grad()
    with autocast(inputs.device.type, enabled=amp_enabled):
        outputs = model(inputs)
        kl = collect_kl_divergence(model)
        loss = criterion(outputs, targets, kl)
    if scaler is not None:
        scaler.scale(loss).backward()
        if grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        if grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
    return loss.item()


@train_epoch.register((DropConnectPredictor, DropoutPredictor))
def train_epoch_cross_entropy(
    model: Predictor,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    optimizer: optim.Optimizer,
    grad_clip_norm: float | None = None,
    amp_enabled: bool = False,
    scaler: GradScaler | None = None,
    **kwargs: Any,  # noqa: ANN401, ARG001
) -> torch.Tensor | float:
    """Train a stochastic NN (dropout/dropconnect) for one epoch with cross-entropy."""
    criterion = nn.CrossEntropyLoss()
    optimizer.zero_grad()
    with autocast(inputs.device.type, enabled=amp_enabled):
        outputs = model(inputs)  # ty: ignore[call-non-callable]
        loss = criterion(outputs, targets)
    if scaler is not None:
        scaler.scale(loss).backward()
        if grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)  # ty: ignore[unresolved-attribute]
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        if grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)  # ty: ignore[unresolved-attribute]
        optimizer.step()
    return loss.item()


@train_epoch.register(PosteriorNetworkPredictor)
def _(
    model: PosteriorNetworkPredictor,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    optimizer: optim.Optimizer,
    grad_clip_norm: float | None = None,
    amp_enabled: bool = False,
    scaler: GradScaler | None = None,
    entropy_weight: float = 1e-5,
    **kwargs: Any,  # noqa: ANN401, ARG001
) -> torch.Tensor | float:
    """Train a posterior network for one epoch with the PostNet loss."""
    optimizer.zero_grad()
    with autocast(inputs.device.type, enabled=amp_enabled):
        alpha = model(inputs)
        loss = postnet_loss(alpha, targets, entropy_weight=entropy_weight)
    if scaler is not None:
        scaler.scale(loss).backward()
        if grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        if grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
    return loss.item()


@train_epoch.register(DDUPredictor)
def train_epoch_ddu(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    optimizer: optim.Optimizer,
    grad_clip_norm: float | None = None,
    amp_enabled: bool = False,
    scaler: GradScaler | None = None,
    **kwargs: Any,  # noqa: ANN401, ARG001
) -> torch.Tensor | float:
    """Train a DDU predictor for one step with cross-entropy on the classification logits.

    The density head is not used during training; it is fitted on the full
    training set after the supervised phase completes.
    Mukhoti et al., "Deep Deterministic Uncertainty", CVPR 2023
    (https://arxiv.org/abs/2102.11582), Section 3.
    """
    model_ = cast("Any", model)
    criterion = nn.CrossEntropyLoss()
    optimizer.zero_grad()
    with autocast(inputs.device.type, enabled=amp_enabled):
        features = model_.encoder(inputs)
        logits = model_.classification_head(features)
        loss = criterion(logits, targets)
    if scaler is not None:
        scaler.scale(loss).backward()
        if grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        if grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
    return loss.item()


@flexdispatch
def validate(
    model: Predictor,
    val_loader: DataLoader,
    device: torch.device,
    amp_enabled: bool = False,
    **kwargs: Any,  # noqa: ANN401
) -> tuple[float, float]:
    """Validate a model."""
    msg = f"No validation function for {type(model)}"
    raise NotImplementedError(msg)


@validate.register(BayesianPredictor)
@torch.no_grad()
def _(
    model: BayesianPredictor,
    val_loader: DataLoader,
    device: torch.device,
    amp_enabled: bool = False,
    **kwargs: Any,  # noqa: ANN401, ARG001
) -> tuple[float, float]:
    """Validate a Bayesian predictor."""
    criterion = ELBOLoss()
    model.eval()
    val_loss, val_acc = 0.0, 0.0
    num_instances = 0
    for inputs_, targets_ in val_loader:
        inputs, targets = inputs_.to(device), targets_.to(device)
        with autocast(device.type, enabled=amp_enabled):
            outputs = model(inputs)
            kl = collect_kl_divergence(model)
            val_loss += criterion(outputs, targets, kl).item()
            val_acc += _accuracy(outputs, targets) * inputs.shape[0]
            num_instances += inputs.shape[0]
    val_loss /= len(val_loader)
    val_acc /= num_instances
    return val_loss, val_acc


@validate.register((DropConnectPredictor, DropoutPredictor))
@torch.no_grad()
def validate_cross_entropy(
    model: Predictor,
    val_loader: DataLoader,
    device: torch.device,
    amp_enabled: bool = False,
    **kwargs: Any,  # noqa: ANN401, ARG001
) -> tuple[float, float]:
    """Validate a dropout/dropconnect predictor with cross-entropy loss."""
    criterion = nn.CrossEntropyLoss()
    model.eval()  # ty: ignore[unresolved-attribute]
    val_loss = 0.0
    val_acc = 0.0
    num_instances = 0
    for inputs_, targets_ in val_loader:
        inputs, targets = inputs_.to(device), targets_.to(device)
        with autocast(device.type, enabled=amp_enabled):
            outputs = model(inputs)  # ty: ignore[call-non-callable]
            val_loss += criterion(outputs, targets).item()
            val_acc += _accuracy(outputs, targets) * inputs.shape[0]
            num_instances += inputs.shape[0]
    val_loss /= len(val_loader)
    val_acc /= num_instances
    return val_loss, val_acc


@validate.register(PosteriorNetworkPredictor)
@torch.no_grad()
def _(
    model: PosteriorNetworkPredictor,
    val_loader: DataLoader,
    device: torch.device,
    amp_enabled: bool = False,
    entropy_weight: float = 1e-5,
    **kwargs: Any,  # noqa: ANN401, ARG001
) -> tuple[float, float]:
    """Validate a posterior network with the PostNet loss."""
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    num_instances = 0
    for inputs_, targets_ in val_loader:
        inputs, targets = inputs_.to(device), targets_.to(device)
        with autocast(device.type, enabled=amp_enabled):
            alpha = model(inputs)
            val_loss += postnet_loss(alpha, targets, entropy_weight=entropy_weight).item()
            val_acc += _accuracy(alpha, targets) * inputs.shape[0]
            num_instances += inputs.shape[0]
    val_loss /= len(val_loader)
    val_acc /= num_instances
    return val_loss, val_acc


@validate.register(DDUPredictor)
@torch.no_grad()
def validate_ddu(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    amp_enabled: bool = False,
    **kwargs: Any,  # noqa: ANN401, ARG001
) -> tuple[float, float]:
    """Validate a DDU predictor with cross-entropy loss on the classification logits."""
    model_ = cast("Any", model)
    criterion = nn.CrossEntropyLoss()
    model_.eval()
    val_loss = 0.0
    val_acc = 0.0
    num_instances = 0
    for inputs_, targets_ in val_loader:
        inputs, targets = inputs_.to(device), targets_.to(device)
        with autocast(device.type, enabled=amp_enabled):
            features = model_.encoder(inputs)
            logits = model_.classification_head(features)
            val_loss += criterion(logits, targets).item()
            val_acc += _accuracy(logits, targets) * inputs.shape[0]
            num_instances += inputs.shape[0]
    val_loss /= len(val_loader)
    val_acc /= num_instances
    return val_loss, val_acc


@flexdispatch
def evaluate(
    model: Predictor,
    test_loader: DataLoader,
    device: torch.device,
    amp_enabled: bool = False,
    n_bins: int = 10,
    **kwargs: Any,  # noqa: ANN401
) -> dict[str, float]:
    """Evaluate model on test set, computing accuracy, NLL, and ECE."""
    msg = f"No evaluate function for {type(model)}"
    raise NotImplementedError(msg)


@evaluate.register(BayesianPredictor)
@torch.no_grad()
def _(
    model: BayesianPredictor,
    test_loader: DataLoader,
    device: torch.device,
    amp_enabled: bool = False,
    n_bins: int = 10,
    samples: int = 1,
) -> dict[str, float]:
    """Evaluate a Bayesian predictor on test set.

    Averages softmax probabilities over samples stochastic forward passes.
    """
    model.eval()
    all_probs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    for inputs_, targets_ in test_loader:
        inputs, targets = inputs_.to(device), targets_.to(device)
        with autocast(device.type, enabled=amp_enabled):
            sample_probs = torch.stack([F.softmax(model(inputs), dim=1) for _ in range(samples)])
        all_probs.append(sample_probs.mean(dim=0))
        all_labels.append(targets)

    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels)

    return _compute_metrics(probs, labels, n_bins)


@evaluate.register((DropConnectPredictor, DropoutPredictor))
@torch.no_grad()
def _(
    model: Predictor,
    test_loader: DataLoader,
    device: torch.device,
    amp_enabled: bool = False,
    n_bins: int = 10,
    samples: int = 1,
    **kwargs: Any,  # noqa: ANN401, ARG001
) -> dict[str, float]:
    """Evaluate a dropout/dropconnect predictor on test set.

    Keeps the model in train mode so dropout/dropconnect layers stay active, then
    averages softmax probabilities over ``samples`` stochastic forward passes.
    """
    model.train()  # ty: ignore[unresolved-attribute]
    all_probs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    for inputs_, targets_ in test_loader:
        inputs, targets = inputs_.to(device), targets_.to(device)
        with autocast(device.type, enabled=amp_enabled):
            sample_probs = torch.stack([F.softmax(model(inputs), dim=1) for _ in range(samples)])  # ty: ignore
        all_probs.append(sample_probs.mean(dim=0))
        all_labels.append(targets)

    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels)

    return _compute_metrics(probs, labels, n_bins)


@evaluate.register(PosteriorNetworkPredictor)
@torch.no_grad()
def _(
    model: PosteriorNetworkPredictor,
    test_loader: DataLoader,
    device: torch.device,
    amp_enabled: bool = False,
    n_bins: int = 10,
    **kwargs: Any,  # noqa: ANN401, ARG001
) -> dict[str, float]:
    """Evaluate a posterior network on test set.

    Uses the mean of the predicted Dirichlet (alpha / alpha.sum) as the class probabilities.
    """
    model.eval()
    all_probs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    for inputs_, targets_ in test_loader:
        inputs, targets = inputs_.to(device), targets_.to(device)
        with autocast(device.type, enabled=amp_enabled):
            alpha = model(inputs)
            probs_ = alpha / alpha.sum(dim=1, keepdim=True)
        all_probs.append(probs_)
        all_labels.append(targets)

    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels)

    return _compute_metrics(probs, labels, n_bins)


@evaluate.register(DDUPredictor)
@torch.no_grad()
def evaluate_ddu(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    amp_enabled: bool = False,
    n_bins: int = 10,
    **kwargs: Any,  # noqa: ANN401, ARG001
) -> dict[str, float]:
    """Evaluate a DDU predictor on the test set.

    Uses softmax of the classification logits as class probabilities.
    """
    model_ = cast("Any", model)
    model_.eval()
    all_probs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    for inputs_, targets_ in test_loader:
        inputs, targets = inputs_.to(device), targets_.to(device)
        with autocast(device.type, enabled=amp_enabled):
            features = model_.encoder(inputs)
            logits = model_.classification_head(features)
        all_probs.append(F.softmax(logits, dim=1))
        all_labels.append(targets)
    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels)
    return _compute_metrics(probs, labels, n_bins)


@evaluate.register(
    (
        EnsemblePredictor,
        CredalEnsemblingPredictor,
        CredalRelativeLikelihoodPredictor,
        CredalWrapperPredictor,
        SubensemblePredictor,
    )
)
@torch.no_grad()
def evaluate_ensemble(
    model: EnsemblePredictor,
    test_loader: DataLoader,
    device: torch.device,
    amp_enabled: bool = False,
    n_bins: int = 10,
    **kwargs: Any,  # noqa: ANN401, ARG001
) -> dict[str, float]:
    """Evaluate an ensemble by averaging member softmax outputs.

    Returns ensemble-level metrics and per-member metrics keyed as
    ``member_<i>/accuracy``, ``member_<i>/nll``, ``member_<i>/ece``.
    """
    members = list(model)
    for member in members:
        member.eval()

    all_member_probs: list[list[torch.Tensor]] = [[] for _ in members]
    all_labels: list[torch.Tensor] = []
    for inputs_, targets_ in test_loader:
        inputs, targets = inputs_.to(device), targets_.to(device)
        with autocast(device.type, enabled=amp_enabled):
            for j, member in enumerate(members):
                all_member_probs[j].append(F.softmax(member(inputs), dim=1))
        all_labels.append(targets)

    labels = torch.cat(all_labels)

    # Per-member metrics
    metrics: dict[str, float] = {}
    member_probs_cat: list[torch.Tensor] = []
    for j, member_batches in enumerate(all_member_probs):
        probs_j = torch.cat(member_batches)
        member_probs_cat.append(probs_j)
        for key, value in _compute_metrics(probs_j, labels, n_bins).items():
            metrics[f"member_{j}/{key}"] = value

    # Ensemble metrics (average of member probabilities)
    ensemble_probs = torch.stack(member_probs_cat).mean(dim=0)
    metrics.update(_compute_metrics(ensemble_probs, labels, n_bins))

    return metrics


def _compute_metrics(probs: torch.Tensor, labels: torch.Tensor, n_bins: int) -> dict[str, float]:
    """Compute accuracy, NLL, and ECE from probabilities and labels."""
    accuracy = _accuracy(probs, labels)
    # compute the epsilon to be used for clamping to prevent log(0) in NLL calculation
    eps = torch.finfo(probs.dtype).eps
    logprobs = torch.log(probs.clamp(min=eps))
    nll = F.nll_loss(logprobs, labels).item()
    ece = ExpectedCalibrationError(num_bins=n_bins)(probs, labels).item()

    return {"accuracy": accuracy, "nll": nll, "ece": ece}


def _accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute the accuracy given outputs (probabilities or logits) and targets (labels)."""
    return (outputs.argmax(1) == targets).float().mean().item()


def _is_improvement(new_loss: float, best_loss: float, min_delta: float = 0.0) -> bool:
    """Whether ``new_loss`` is a meaningful improvement over ``best_loss``."""
    return new_loss < best_loss - min_delta


class EarlyStopping:
    """Stop training when validation loss stops improving.

    Args:
        patience: Number of epochs to wait before stopping.
        min_delta: Minimum change to qualify as an improvement.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0) -> None:
        """Initialize early stopping."""
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")

    def should_stop(self, val_loss: float) -> bool:
        """Check if training should stop."""
        if _is_improvement(val_loss, self.best_loss, self.min_delta):
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


class BestModelTracker:
    """Track the state_dict of the model with the lowest validation loss seen.

    Weights are stored as a CPU-resident deep copy of ``model.state_dict()`` so that
    the running "best" weights do not share storage with the live model and do not
    occupy GPU memory.

    Args:
        min_delta: Minimum change to qualify as an improvement.
    """

    def __init__(self, min_delta: float = 0.0) -> None:
        """Initialize the best-model tracker."""
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.best_state_dict: dict[str, torch.Tensor] | None = None

    def update(self, val_loss: float, model: nn.Module) -> bool:
        """Store a CPU clone of ``model.state_dict()`` if ``val_loss`` improves.

        Returns whether the stored best was updated.
        """
        if _is_improvement(val_loss, self.best_loss, self.min_delta):
            self.best_loss = val_loss
            self.best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return True
        return False
