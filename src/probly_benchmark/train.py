"""Script to train models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from probly.method.efficient_credal_prediction import EfficientCredalPredictor

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from torch.utils.data import DataLoader

    from probly.method.ddu.torch import GaussianMixtureHead


import pathlib
import tempfile

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from pytorch_optimizer import Lamb
import scipy.optimize
import scipy.special
import torch
from torch import nn, optim
from torch.amp import GradScaler
from tqdm import tqdm
import wandb
import wandb.util

from flextype import flexdispatch
from probly.method.bayesian import BayesianPredictor
from probly.method.credal_ensembling import CredalEnsemblingPredictor
from probly.method.credal_relative_likelihood import CredalRelativeLikelihoodPredictor
from probly.method.credal_wrapper import CredalWrapperPredictor
from probly.method.ddu import DDUPredictor
from probly.method.ensemble import EnsemblePredictor
from probly.method.subensemble import SubensemblePredictor
from probly_benchmark import data, metadata, utils
from probly_benchmark.builders import BuildContext, build_model
from probly_benchmark.paths import CHECKPOINT_PATH
from probly_benchmark.train_funcs import (
    BestModelTracker,
    EarlyStopping,
    evaluate,
    train_epoch,
    train_epoch_cross_entropy,
    validate,
    validate_cross_entropy,
)

torch.set_float32_matmul_precision("high")


OPTIMIZERS = {
    "adam": optim.Adam,
    "sgd": optim.SGD,
    "lamb": Lamb,
}


def _get_state_dict(model: nn.Module | list[nn.Module]) -> dict | list[dict]:
    """Return state dict(s) for a model or list of models."""
    if isinstance(model, list):
        return [cast("nn.Module", m).state_dict() for m in model]
    return model.state_dict()


def get_optimizer(name: str, params: Iterable[nn.Parameter], **kwargs: Any) -> optim.Optimizer:  # noqa: ANN401
    """Get optimizer function."""
    name = name.lower()
    if name not in OPTIMIZERS:
        msg = f"Unknown optimizer: {name}"
        raise ValueError(msg)
    return OPTIMIZERS[name](params, **kwargs)


def _make_warmup_cosine(
    optimizer: optim.Optimizer,
    *,
    total_iters: int,
    warmup_iters: int,
    warmup_start_factor: float = 1e-2,
    min_lr: float = 0.0,
) -> optim.lr_scheduler.LRScheduler:
    """Build a linear-warmup + cosine-decay scheduler that steps per-iteration."""
    warmup = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=warmup_start_factor,
        end_factor=1.0,
        total_iters=warmup_iters,
    )
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_iters - warmup_iters),
        eta_min=min_lr,
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_iters],
    )
    scheduler._step_per_iter = True  # ty: ignore[unresolved-attribute]  # noqa: SLF001
    return scheduler


SCHEDULERS: dict[str, Callable[..., optim.lr_scheduler.LRScheduler] | None] = {
    "cosine": optim.lr_scheduler.CosineAnnealingLR,
    "multistep": optim.lr_scheduler.MultiStepLR,
    "step": optim.lr_scheduler.StepLR,
    "plateau": optim.lr_scheduler.ReduceLROnPlateau,
    "warmup_cosine": _make_warmup_cosine,
}


def get_scheduler(
    name: str,
    optimizer: optim.Optimizer,
    epochs: int,
    iters_per_epoch: int,
    **kwargs: Any,  # noqa: ANN401
) -> optim.lr_scheduler.LRScheduler | None:
    """Get learning rate scheduler.

    Args:
        name: Scheduler name.
        optimizer: Optimizer to schedule.
        epochs: Total number of training epochs. Used as default ``T_max`` for cosine
            and to derive ``total_iters`` for ``warmup_cosine``.
        iters_per_epoch: Number of optimizer steps per epoch. Used to convert the
            ``warmup_epochs`` config into per-iteration counts for ``warmup_cosine``.
        **kwargs: Additional scheduler-specific kwargs. An explicit ``T_max`` here
            overrides the ``epochs`` default for cosine.
    """
    name = name.lower()
    if name not in SCHEDULERS:
        msg = f"Unknown scheduler: {name}"
        raise ValueError(msg)
    factory = SCHEDULERS[name]
    if factory is None:
        return None
    if name == "cosine":
        kwargs.setdefault("T_max", epochs)
    elif name == "warmup_cosine":
        warmup_epochs = kwargs.pop("warmup_epochs", 0)
        kwargs["total_iters"] = epochs * iters_per_epoch
        kwargs["warmup_iters"] = warmup_epochs * iters_per_epoch
    return factory(optimizer, **kwargs)


def _training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    cfg: DictConfig,
    device: torch.device,
    run: Any,  # noqa: ANN401
    train_kwargs: dict[str, Any],
    train_fn: Callable[..., float],
    val_fn: Callable[..., tuple[float, float]],
    log_prefix: str = "",
) -> None:
    """Run the training loop for a single model.

    Args:
        model: The model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader, or ``None`` to skip validation.
        cfg: Hydra config with training hyperparameters.
        device: Device to train on.
        run: Weights & Biases run for logging.
        train_kwargs: Method-specific keyword arguments forwarded to ``train_fn``
            and ``val_fn``.
        train_fn: Per-batch training function.
        val_fn: Validation function.
        log_prefix: Prefix for W&B log keys (e.g. ``"member_0/"``).
    """
    model.forward = torch.compile(model.forward)

    optimizer = get_optimizer(
        cfg.optimizer.name,
        model.parameters(),
        **cfg.optimizer.get("params", {}),
    )
    scheduler = get_scheduler(
        cfg.scheduler.name,
        optimizer,
        cfg.epochs,
        len(train_loader),
        **cfg.scheduler.get("params", {}),
    )
    step_per_iter = getattr(scheduler, "_step_per_iter", False)
    min_delta = cfg.early_stopping.get("min_delta", 0.0)
    early_stopping = (
        EarlyStopping(patience=cfg.early_stopping.patience, min_delta=min_delta)
        if cfg.early_stopping.patience
        else None
    )
    best_tracker = BestModelTracker(min_delta=min_delta) if val_loader is not None else None

    grad_clip_norm = cfg.get("grad_clip_norm", None)
    amp_enabled = cfg.get("amp", False)
    scaler = GradScaler(device.type) if amp_enabled else None

    for epoch in tqdm(range(cfg.epochs), desc=f"{log_prefix}Epoch"):
        model.train()
        running_loss = 0.0
        for inputs_, targets_ in tqdm(train_loader):
            inputs = inputs_.to(device, non_blocking=True)
            if device.type == "cuda":
                inputs = inputs.contiguous(memory_format=torch.channels_last)
            targets = targets_.to(device, non_blocking=True)
            running_loss += train_fn(
                model,
                inputs,
                targets,
                optimizer,
                grad_clip_norm=grad_clip_norm,
                amp_enabled=amp_enabled,
                scaler=scaler,
                **train_kwargs,
            )
            if scheduler is not None and step_per_iter:
                scheduler.step()
        running_loss /= len(train_loader)

        val_loss: float | None = None
        log_data = {f"{log_prefix}train_loss": running_loss}
        if val_loader:
            val_loss, val_acc = val_fn(model, val_loader, device, amp_enabled, **train_kwargs)
            log_data[f"{log_prefix}val_loss"] = val_loss
            log_data[f"{log_prefix}val_acc"] = val_acc
        run.log(data=log_data)

        if best_tracker is not None and val_loss is not None:
            best_tracker.update(val_loss, model)

        if scheduler is not None and not step_per_iter:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)  # ty: ignore[invalid-argument-type]
            else:
                scheduler.step()

        if early_stopping is not None and val_loss is not None and early_stopping.should_stop(val_loss):
            run.summary[f"{log_prefix}early_stopped"] = True
            print(f"Early stopping at epoch {epoch}")
            break
    else:
        run.summary[f"{log_prefix}early_stopped"] = False

    if best_tracker is not None and best_tracker.best_state_dict is not None:
        model.load_state_dict(best_tracker.best_state_dict)
        run.summary[f"{log_prefix}best_val_loss"] = best_tracker.best_loss


@flexdispatch
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    cfg: DictConfig,
    device: torch.device,
    run: Any,  # noqa: ANN401
    train_kwargs: dict[str, Any],
) -> None:
    """Train a model. Dispatches on model type for ensemble vs. single-model training."""
    _training_loop(
        model,
        train_loader,
        val_loader,
        cfg,
        device,
        run,
        train_kwargs,
        train_fn=train_epoch,  # ty: ignore[invalid-argument-type]
        val_fn=validate,
    )


@train_model.register((EnsemblePredictor, CredalEnsemblingPredictor, CredalWrapperPredictor, SubensemblePredictor))
def _(
    model: EnsemblePredictor,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    cfg: DictConfig,
    device: torch.device,
    run: Any,  # noqa: ANN401
    train_kwargs: dict[str, Any],
) -> None:
    """Train an ensemble by training each member independently."""
    for i, member in enumerate(model):
        _training_loop(
            member,
            train_loader,
            val_loader,
            cfg,
            device,
            run,
            train_kwargs,
            train_fn=train_epoch_cross_entropy,
            val_fn=validate_cross_entropy,
            log_prefix=f"member_{i}/",
        )


@torch.no_grad()
def _compute_log_likelihood(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool = False,
) -> float:
    """Compute average log-likelihood of ``model`` on ``loader``."""
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_samples = 0
    for inputs_, targets_ in loader:
        inputs, targets = inputs_.to(device), targets_.to(device)
        with torch.amp.autocast(device.type, enabled=amp_enabled):
            outputs = model(inputs)
            total_loss += criterion(outputs, targets).item()
        total_samples += targets.size(0)
    return -total_loss / total_samples


def _training_loop_relative_likelihood(  # noqa: PLR0912
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    cfg: DictConfig,
    device: torch.device,
    run: Any,  # noqa: ANN401
    train_fn: Callable[..., float],
    val_fn: Callable[..., tuple[float, float]],
    max_ll: float,
    alpha: float,
    batch_check: bool = False,
    log_prefix: str = "",
) -> None:
    """Training loop that stops when the relative likelihood reaches ``alpha``."""
    optimizer = get_optimizer(
        cfg.optimizer.name,
        model.parameters(),
        **cfg.optimizer.get("params", {}),
    )
    scheduler = get_scheduler(
        cfg.scheduler.name,
        optimizer,
        cfg.epochs,
        len(train_loader),
        **cfg.scheduler.get("params", {}),
    )
    step_per_iter = getattr(scheduler, "_step_per_iter", False)

    grad_clip_norm = cfg.get("grad_clip_norm", None)
    amp_enabled = cfg.get("amp", False)
    scaler = GradScaler(device.type) if amp_enabled else None

    stopped = False
    for epoch in tqdm(range(cfg.epochs), desc=f"{log_prefix}Epoch"):
        model.train()
        running_loss = 0.0
        for inputs_, targets_ in tqdm(train_loader):
            inputs = inputs_.to(device, non_blocking=True)
            if device.type == "cuda":
                inputs = inputs.contiguous(memory_format=torch.channels_last)
            targets = targets_.to(device, non_blocking=True)
            running_loss += train_fn(
                model,
                inputs,
                targets,
                optimizer,
                grad_clip_norm=grad_clip_norm,
                amp_enabled=amp_enabled,
                scaler=scaler,
            )
            if scheduler is not None and step_per_iter:
                scheduler.step()

            if batch_check:
                model.eval()
                current_ll = _compute_log_likelihood(model, train_loader, device, amp_enabled)
                model.train()
                relative_likelihood = torch.exp(torch.tensor(current_ll - max_ll)).item()
                if relative_likelihood >= alpha:
                    stopped = True
                    break

        running_loss /= len(train_loader)

        if not stopped:
            model.eval()
            current_ll = _compute_log_likelihood(model, train_loader, device, amp_enabled)
            relative_likelihood = torch.exp(torch.tensor(current_ll - max_ll)).item()

        val_loss: float | None = None
        log_data = {
            f"{log_prefix}train_loss": running_loss,
            f"{log_prefix}relative_likelihood": relative_likelihood,
        }
        if val_loader:
            val_loss, val_acc = val_fn(model, val_loader, device, amp_enabled)
            log_data[f"{log_prefix}val_loss"] = val_loss
            log_data[f"{log_prefix}val_acc"] = val_acc
        run.log(data=log_data)

        if scheduler is not None and not step_per_iter:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)  # ty: ignore[invalid-argument-type]
            else:
                scheduler.step()

        if stopped or relative_likelihood >= alpha:
            run.summary[f"{log_prefix}rl_stopped"] = True
            run.summary[f"{log_prefix}stopped_epoch"] = epoch
            print(f"Relative likelihood stopping at epoch {epoch} (RL={relative_likelihood:.4f})")
            break
    else:
        run.summary[f"{log_prefix}rl_stopped"] = False


@train_model.register(BayesianPredictor)
def _(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    cfg: DictConfig,
    device: torch.device,
    run: Any,  # noqa: ANN401
    train_kwargs: dict[str, Any],
) -> None:
    """Train a BayesianPredictor with ELBO loss.

    The KL penalty is set to 1/N (N = dataset size) following
    Blundell et al., "Weight Uncertainty in Neural Networks", ICML 2015.
    """
    dataset = getattr(train_loader, "dataset", None)
    dataset_size = len(dataset) if dataset is not None else len(train_loader) * cfg.batch_size
    kl_penalty = 1.0 / dataset_size
    _training_loop(
        model,
        train_loader,
        val_loader,
        cfg,
        device,
        run,
        {**train_kwargs, "kl_penalty": kl_penalty},
        train_fn=train_epoch,  # ty: ignore[invalid-argument-type]
        val_fn=validate,
    )


@train_model.register(CredalRelativeLikelihoodPredictor)
def _(
    model: CredalRelativeLikelihoodPredictor,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    cfg: DictConfig,
    device: torch.device,
    run: Any,  # noqa: ANN401
    train_kwargs: dict[str, Any],
) -> None:
    """Train a credal relative likelihood ensemble."""
    members = list(model)
    alpha = train_kwargs.get("alpha", 0.5)
    batch_check = train_kwargs.get("batch_check", False)
    num_remaining = len(members) - 1

    # Train first member fully (standard training loop)
    _training_loop(
        members[0],
        train_loader,
        val_loader,
        cfg,
        device,
        run,
        {},  # empty, because train_epoch_cross_entropy and validate_cross_entropy don't use args
        train_fn=train_epoch_cross_entropy,
        val_fn=validate_cross_entropy,
        log_prefix="member_0/",
    )

    # Compute reference log-likelihood from the fully trained first member
    amp_enabled = cfg.get("amp", False)
    members[0].eval()
    max_ll = _compute_log_likelihood(members[0], train_loader, device, amp_enabled)
    run.summary["max_ll"] = max_ll

    # Thresholds from alpha to 1.0 (exclusive), one per remaining member
    thresholds = torch.linspace(alpha, 1.0, num_remaining + 1)[:-1].tolist()
    print(f"Thresholds: {thresholds}")

    # Train remaining members with per-member relative likelihood thresholds
    for i, (member, threshold) in enumerate(zip(members[1:], thresholds, strict=True), start=1):
        _training_loop_relative_likelihood(
            member,
            train_loader,
            val_loader,
            cfg,
            device,
            run,
            train_fn=train_epoch_cross_entropy,
            val_fn=validate_cross_entropy,
            max_ll=max_ll,
            alpha=threshold,
            batch_check=batch_check,
            log_prefix=f"member_{i}/",
        )


@torch.no_grad()
def _fit_ddu_density_head(
    model: DDUPredictor,
    train_loader: DataLoader,
    device: torch.device,
    amp_enabled: bool = False,
) -> None:
    """Fit the GMM density head of a DDU predictor on the full training set.

    Iterates through ``train_loader`` to extract encoder features batch-by-batch
    (on ``device``), accumulates them on CPU to avoid GPU memory exhaustion, then
    calls ``density_head.fit`` on CPU before moving the fitted buffers back to
    ``device``.

    Args:
        model: Trained DDU predictor with ``.encoder`` and ``.density_head`` attributes.
        train_loader: Training data loader used to collect features.
        device: Device on which the encoder runs.
        amp_enabled: Whether to use automatic mixed precision during feature extraction.
    """
    model.eval()
    all_features: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    for inputs_, targets_ in tqdm(train_loader, desc="Fitting DDU density head"):
        inputs = inputs_.to(device, non_blocking=True)
        if device.type == "cuda":
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        targets = targets_.to(device, non_blocking=True)
        with torch.amp.autocast(device.type, enabled=amp_enabled):
            features = model.encoder(inputs)
        all_features.append(features.detach().cpu())
        all_labels.append(targets.detach().cpu())
    features_cat = torch.cat(all_features)
    labels_cat = torch.cat(all_labels)
    density_head: GaussianMixtureHead = model.density_head
    density_head_device = density_head.means.device
    density_head.cpu()
    density_head.fit(features_cat, labels_cat)
    density_head.to(density_head_device)


@train_model.register(DDUPredictor)
def _(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    cfg: DictConfig,
    device: torch.device,
    run: Any,  # noqa: ANN401
    train_kwargs: dict[str, Any],
) -> None:
    """Train a DDU predictor and fit the GMM density head post-training.

    Phase 1 trains the spectrally-normalised network end-to-end with standard
    cross-entropy exactly as described in Mukhoti et al., CVPR 2023
    (https://arxiv.org/abs/2102.11582).  Phase 2 fits the per-class Gaussian
    density estimator (GDA) on all training features extracted from the frozen
    encoder, which is used at inference time for epistemic uncertainty scoring.
    """
    _training_loop(
        model,
        train_loader,
        val_loader,
        cfg,
        device,
        run,
        train_kwargs,
        train_fn=train_epoch,  # ty: ignore[invalid-argument-type]
        val_fn=validate,
    )
    amp_enabled = cfg.get("amp", False)
    _fit_ddu_density_head(model, train_loader, device, amp_enabled)
    run.summary["ddu_gmm_fitted"] = True


def _compute_efficient_credal_prediction_bounds(
    logits_train: torch.Tensor,
    targets_train: torch.Tensor,
    num_classes: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-class additive logit bounds via classwise relative-likelihood optimisation.

    For each class ``k`` and each direction (min/max), find the optimal additive
    logit perturbation ``x`` on column ``k`` that keeps the mean training
    relative likelihood at least ``alpha``. The relative likelihood is
    ``exp(ll(logits + x * e_k) - ll(logits))`` where ``ll`` is the mean
    per-sample log-likelihood.

    Based on :cite:`hofmanefficient` and the reference implementation at
    https://github.com/pwhofman/efficient-credal-prediction/blob/main/models.py.

    Args:
        logits_train: Training logits, shape ``(N, num_classes)``.
        targets_train: Integer training targets, shape ``(N,)``.
        num_classes: Number of classes.
        alpha: Relative-likelihood threshold in ``[0, 1]``.

    Returns:
        Tuple ``(lower, upper)`` of ``numpy.ndarray`` with shape ``(num_classes,)``
        and dtype ``float64``. ``lower[k]`` is the most-negative logit
        perturbation on class ``k`` that keeps the relative likelihood at least
        ``alpha``; ``upper[k]`` is the most-positive.
    """
    logits_np = logits_train.detach().cpu().numpy().astype(np.float64)
    targets_np = targets_train.detach().cpu().numpy().astype(np.int64)

    def _mean_log_likelihood(logits: np.ndarray, targets: np.ndarray) -> float:
        log_probs = scipy.special.log_softmax(logits, axis=1)
        return float(log_probs[np.arange(len(targets)), targets].mean())

    mll = _mean_log_likelihood(logits_np, targets_np)

    lower = np.zeros(num_classes, dtype=np.float64)
    upper = np.zeros(num_classes, dtype=np.float64)

    for k in tqdm(range(num_classes), desc="Credal bounds"):
        for direction in (1, -1):

            def fun(x: np.ndarray, direction: int = direction) -> float:
                return float(direction * x[0])

            def const(x: np.ndarray, k: int = k) -> float:
                perturbed = logits_np.copy()
                perturbed[:, k] += x[0]
                return float(np.exp(_mean_log_likelihood(perturbed, targets_np) - mll) - alpha)

            res = scipy.optimize.minimize(
                fun,
                x0=np.array([0.0]),
                constraints=[{"type": "ineq", "fun": const}],
                bounds=[(-1e4, 1e4)],
            )
            if not res.success:
                print(f"scipy.optimize.minimize did not converge for class {k} direction {direction}: {res.message}")
            if direction == 1:
                lower[k] = float(res.x[0])
            else:
                upper[k] = float(res.x[0])

    return lower, upper


@train_model.register(EfficientCredalPredictor)
def _(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    cfg: DictConfig,
    device: torch.device,
    run: Any,  # noqa: ANN401
    train_kwargs: dict[str, Any],
) -> None:
    """Train an EfficientCredalPredictor.

    First train the base predictor with cross-entropy, then compute classwise
    additive logit bounds on the training set and store them in the predictor's
    ``lower`` and ``upper`` buffers.
    """
    _training_loop(
        model,
        train_loader,
        val_loader,
        cfg,
        device,
        run,
        train_kwargs,
        train_fn=train_epoch,  # ty: ignore[invalid-argument-type]
        val_fn=validate,
    )
    model_ = cast("Any", model)
    amp_enabled = cfg.get("amp", False)
    alpha = train_kwargs.get("alpha", 0.5)
    num_classes = int(model_.lower.shape[0])

    logits_train, targets_train = utils.collect_outputs_targets_raw(
        model_.predictor,
        train_loader,
        device,
        amp_enabled,
    )
    lower, upper = _compute_efficient_credal_prediction_bounds(
        logits_train,
        targets_train,
        num_classes=num_classes,
        alpha=alpha,
    )
    model_.lower.copy_(torch.from_numpy(lower).to(model_.lower))
    model_.upper.copy_(torch.from_numpy(upper).to(model_.upper))
    run.summary["efficient_credal_alpha"] = alpha


@hydra.main(version_base=None, config_path="configs/", config_name="train")
def main(cfg: DictConfig) -> None:
    """Run the training script."""
    print("=== Training configuration ===")
    print(OmegaConf.to_yaml(cfg))

    run_id = wandb.util.generate_id()
    seed = cfg.get("seed", None)
    utils.set_seed(seed)

    run = wandb.init(
        id=run_id,
        name=f"{cfg.method.name}_{cfg.base_model}_{cfg.dataset}_{run_id}",
        entity=cfg.wandb.get("entity", None),
        project=cfg.wandb.project,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),  # ty: ignore
        mode="online" if cfg.wandb.enabled else "disabled",
        save_code=True,
    )
    run.config.update({"seed": seed})

    device = utils.get_device(cfg.get("device", None))
    print(f"Running on device: {device}")

    loaders = data.get_data_train(
        cfg.dataset,
        seed,
        val_split=cfg.val_split,
        cal_split=cfg.get("cal_split", 0.0),
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers,
        prefetch_factor=cfg.get("prefetch_factor", 4),
        shuffle=True,
    )
    train_loader = loaders.train
    val_loader = loaders.validation
    test_loader = loaders.test

    num_classes = metadata.DATASETS[cfg.dataset].num_classes

    method_kwargs: dict[str, Any] = (
        OmegaConf.to_container(cfg.method.params, resolve=True) if cfg.method.get("params") else {}
    )  # ty: ignore[invalid-assignment]
    train_kwargs: dict[str, Any] = (
        OmegaConf.to_container(cfg.method.train, resolve=True) if cfg.method.get("train") else {}
    )  # ty: ignore[invalid-assignment]

    context = BuildContext(
        base_model_name=cfg.base_model,
        model_type=cfg.model_type,
        num_classes=num_classes,
        pretrained=cfg.pretrained,
        train_loader=train_loader,
    )
    model = build_model(cfg.method.name, method_kwargs, context)
    if isinstance(model, EnsemblePredictor):
        for member in model:
            member.to(device)
    else:
        model = model.to(device)

    if cfg.val_split == 0:
        if cfg.scheduler.name.lower() == "plateau":
            msg = "ReduceLROnPlateau scheduler requires `val_split > 0` in the config."
            raise ValueError(msg)
        if cfg.early_stopping.patience:
            msg = (
                "Early stopping requires `val_split > 0` in the config. "
                "Disable early stopping or set a validation split."
            )
            raise ValueError(msg)

    train_model(model, train_loader, val_loader, cfg, device, run, train_kwargs)

    test_metrics = evaluate(model, test_loader, device, cfg.get("amp", False), **train_kwargs)
    run.summary.update(test_metrics)
    run.log(data=test_metrics)

    checkpoint = {
        "model_state_dict": _get_state_dict(model),
        "config": OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        "metrics": test_metrics,
    }

    artifact_name = f"{cfg.method.name}_{cfg.base_model}_{cfg.dataset}_{seed}"

    if cfg.save_to_disk:
        path = pathlib.Path(CHECKPOINT_PATH).joinpath(f"{artifact_name}.pt")
        torch.save(checkpoint, path)

        artifact = wandb.Artifact(
            name=artifact_name,
            type="model",
            metadata=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),  # ty: ignore
        )
        artifact.add_file(str(path))
        wandb.log_artifact(artifact)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp).joinpath(f"{artifact_name}.pt")
            torch.save(checkpoint, path)

            artifact = wandb.Artifact(
                name=artifact_name,
                type="model",
                metadata=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),  # ty: ignore
            )
            artifact.add_file(str(path))
            wandb.log_artifact(artifact)

    run.finish()


if __name__ == "__main__":
    main()
