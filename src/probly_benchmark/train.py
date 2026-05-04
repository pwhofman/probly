"""Script to train models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from torch.utils.data import DataLoader

    from probly.layers.torch import GaussianMixtureHead


import contextlib
import gc
import pathlib
import tempfile

import hydra
from laplace.baselaplace import BaseLaplace
from laplace.utils.feature_extractor import FeatureExtractor
from omegaconf import DictConfig, OmegaConf
from pytorch_optimizer import Lamb
import torch
from torch import nn, optim
from torch.amp import GradScaler
from tqdm import tqdm
import wandb
import wandb.util

from flextype import flexdispatch
from probly.layers.torch import BatchEnsembleConv2d, BatchEnsembleLinear
from probly.method.batchensemble import BatchEnsemblePredictor
from probly.method.bayesian import BayesianPredictor
from probly.method.credal_bnn import CredalBNNPredictor
from probly.method.credal_relative_likelihood import CredalRelativeLikelihoodPredictor
from probly.method.dare import DarePredictor
from probly.method.ddu import DDUPredictor
from probly.method.deup import DEUPPredictor
from probly.method.duq import DUQPredictor
from probly.method.efficient_credal_prediction import (
    EfficientCredalPredictor,
    compute_efficient_credal_prediction_bounds,
)
from probly.method.ensemble import EnsemblePredictor
from probly.method.subensemble import SubensemblePredictor
from probly_benchmark import conformal, data, metadata, utils
from probly_benchmark.builders import BuildContext, build_model
from probly_benchmark.paths import CHECKPOINT_PATH
from probly_benchmark.train_funcs import (
    BestModelTracker,
    EarlyStopping,
    ValidationMetrics,
    evaluate,
    train_epoch,
    train_epoch_batchensemble,
    train_epoch_cross_entropy,
    train_epoch_dare,
    train_epoch_deup,
    validate,
    validate_batchensemble,
    validate_cross_entropy,
    validate_deup,
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


def _move_to_device(model: nn.Module | EnsemblePredictor | BaseLaplace, device: torch.device) -> None:
    """Move model to device in-place, handling type-specific things."""
    if isinstance(model, EnsemblePredictor):
        for member in model:
            member.to(device)
    elif isinstance(model, BaseLaplace):
        model.model.to(device)
    else:
        model.to(device)


def _log_artifact_file(path: pathlib.Path, artifact_name: str, metadata: dict[str, Any]) -> None:
    """Log a checkpoint file as a wandb model artifact."""
    artifact = wandb.Artifact(name=artifact_name, type="model", metadata=metadata)
    artifact.add_file(str(path))
    wandb.log_artifact(artifact)


def _save_checkpoint_artifact(model: nn.Module, cfg: DictConfig, test_metrics: dict[str, float]) -> None:
    """Save and log the trained model checkpoint artifact."""
    metadata = cast("dict[str, Any]", OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
    checkpoint = {
        "model_state_dict": _get_state_dict(model),
        "config": metadata,
        "metrics": test_metrics,
    }
    artifact_name = utils.resolve_artifact_name(cfg)

    if cfg.save_to_disk:
        path = pathlib.Path(CHECKPOINT_PATH).joinpath(f"{artifact_name}.pt")
        torch.save(checkpoint, path)
        _log_artifact_file(path, artifact_name, metadata)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp).joinpath(f"{artifact_name}.pt")
            torch.save(checkpoint, path)
            _log_artifact_file(path, artifact_name, metadata)


def get_optimizer(
    name: str,
    params: Iterable[nn.Parameter] | list[dict[str, Any]],
    **kwargs: Any,  # noqa: ANN401
) -> optim.Optimizer:
    """Get optimizer function. ``params`` may be parameter iterables or param-group dicts."""
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

SUPERVISED_LOSS_METHODS = {
    "base",  # Special "method" for training a predictor without applying a method.
    "batchensemble",
    "credal_ensembling",
    "credal_wrapper",
    "dropconnect",
    "dropout",
    "efficient_credal_prediction",
    "ensemble",
    "het_net",
    "subensemble",
}
"""The set of method names that support non-default supervised losses."""


def _validate_supervised_loss_config(cfg: DictConfig) -> None:
    """Validate that non-default supervised losses are only used with supported methods."""
    supervised_loss_name = utils.get_supervised_loss_name(cfg)
    if supervised_loss_name == utils.DEFAULT_SUPERVISED_LOSS:
        return
    if cfg.method.name not in SUPERVISED_LOSS_METHODS:
        supported = ", ".join(sorted(SUPERVISED_LOSS_METHODS))
        msg = (
            f"supervised_loss.name={supervised_loss_name!r} is only supported for CE-compatible methods "
            f"({supported}); got method={cfg.method.name!r}."
        )
        raise ValueError(msg)


def _validate_conformal_config(cfg: DictConfig) -> None:
    """Reject split-conformal methods during training."""
    conformal_name = conformal.get_conformal_name(cfg)
    if conformal_name == conformal.DEFAULT_CONFORMAL:
        return
    conformal.validate_conformal_config(cfg)
    msg = (
        f"conformal.name={conformal_name!r} is a split-conformal post-hoc method. "
        "Train the base model first, then apply split-conformal prediction with conformalize.py."
    )
    raise ValueError(msg)


def _build_train_kwargs(cfg: DictConfig) -> dict[str, Any]:
    """Build keyword arguments forwarded to method-specific training and evaluation functions."""
    train_kwargs: dict[str, Any] = (
        OmegaConf.to_container(cfg.method.train, resolve=True) if cfg.method.get("train") else {}
    )  # ty: ignore[invalid-assignment]
    if cfg.method.name in SUPERVISED_LOSS_METHODS:
        train_kwargs["supervised_loss"] = OmegaConf.to_container(cfg.supervised_loss, resolve=True)
    return train_kwargs


def _build_method_kwargs(cfg: DictConfig) -> dict[str, Any]:
    """Build keyword arguments forwarded to the method constructor."""
    return OmegaConf.to_container(cfg.method.params, resolve=True) if cfg.method.get("params") else {}  # ty: ignore


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


def _maybe_compile_forward(model: nn.Module, device: torch.device) -> None:
    """Compile ``model.forward`` in place unless the device is MPS.

    Inductor's Metal backend currently generates invalid shader code for common
    ResNet kernels (non-constant threadgroup array size, undefined ``Min``),
    so compilation is skipped on MPS and a message is printed to surface the
    deviation from the default CUDA/CPU path.
    """
    if device.type == "mps" or isinstance(model, DUQPredictor):
        print("Skipping torch.compile on MPS (Inductor's Metal backend unsupported); using eager forward.")
        return
    if getattr(model, "_probly_skip_compile", False):
        return
    model.forward = torch.compile(model.forward)


def _training_loop(  # noqa: PLR0912, PLR0915
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    cfg: DictConfig,
    device: torch.device,
    run: Any,  # noqa: ANN401
    train_kwargs: dict[str, Any],
    train_fn: Callable[..., float],
    val_fn: Callable[..., ValidationMetrics],
    log_prefix: str = "",
    param_groups: list[dict[str, Any]] | None = None,
    extra_metrics: dict[str, Any] | None = None,
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
        param_groups: Optional list of parameter-group dicts forwarded to the optimizer
            in place of ``model.parameters()``. Each dict can override the optimizer's
            default ``lr`` / ``weight_decay`` for a subset of parameters; used by
            BatchEnsemble to apply a slower lr and disable weight decay on fast weights.
        extra_metrics: Optional dict of additional fixed metrics logged every epoch
            alongside the standard loss/accuracy keys (e.g. ``{"member_0/relative_likelihood": 1.0}``).
    """
    _maybe_compile_forward(model, device)

    optimizer = get_optimizer(
        cfg.optimizer.name,
        param_groups if param_groups is not None else model.parameters(),
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

    # Each prefixed member gets its own hidden epoch counter as x-axis so that
    # all member charts share a 0..n_epochs range rather than a global step.
    if log_prefix:
        epoch_key: str | None = f"{log_prefix.rstrip('/')}_epoch"
        wandb.define_metric(epoch_key, hidden=True)
        wandb.define_metric(f"{log_prefix}*", step_metric=epoch_key)
    else:
        epoch_key = None

    for epoch in tqdm(range(cfg.epochs), desc=f"{log_prefix}Epoch"):
        model.train()
        running_loss = 0.0
        for inputs_, targets_ in tqdm(train_loader):
            inputs = inputs_.to(device, non_blocking=True)
            if device.type == "cuda" and inputs.ndim >= 4:
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
                epoch=epoch,
                **train_kwargs,
            )
            if scheduler is not None and step_per_iter:
                scheduler.step()
        running_loss /= len(train_loader)

        val_loss: float | None = None
        log_data: dict[str, Any] = {f"{log_prefix}train_loss": running_loss}
        if epoch_key is not None:
            log_data[epoch_key] = epoch
        if extra_metrics:
            log_data.update(extra_metrics)
        if val_loader:
            metrics = val_fn(model, val_loader, device, amp_enabled, epoch=epoch, **train_kwargs)
            log_data.update({f"{log_prefix}val_{k}": v for k, v in metrics.items()})
            val_loss = metrics["loss"]
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


@train_model.register(EnsemblePredictor)
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


def _shutdown_dataloader_workers(loader: DataLoader, name: str) -> None:
    """Force any persistent worker processes held by ``loader`` to exit.

    Args:
        loader: The DataLoader whose persistent workers should be terminated.
        name: Human-readable loader name used in the log line (e.g. ``"train"``,
            ``"val"``).
    """
    iterator = getattr(loader, "_iterator", None)
    if iterator is None:
        return
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if shutdown is not None:
        # Iterator may already be partially torn down; best-effort cleanup.
        with contextlib.suppress(Exception):
            shutdown()
    loader._iterator = None  # noqa: SLF001
    gc.collect()
    print(
        f"[subensemble] Shut down persistent {name}_loader workers between members "
        f"to release file handles / shared memory."
    )


@train_model.register(SubensemblePredictor)
def _(
    model: SubensemblePredictor,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    cfg: DictConfig,
    device: torch.device,
    run: Any,  # noqa: ANN401
    train_kwargs: dict[str, Any],
) -> None:
    """Train a subensemble by training each head independently on a shared frozen backbone.

    Only the head (trainable) parameters are passed to the optimizer; the frozen
    backbone parameters are excluded to avoid inflating optimizer state and to
    prevent incorrect gradient-norm computations on zero-gradient parameters.
    """
    for i, member in enumerate(model):
        trainable = [p for p in member.parameters() if p.requires_grad]
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
            param_groups=[{"params": trainable}],
        )
        _shutdown_dataloader_workers(train_loader, "train")
        if val_loader is not None:
            _shutdown_dataloader_workers(val_loader, "val")


def _split_batchensemble_params(
    model: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Parameter], list[nn.Parameter]]:
    """Split parameters into ``(shared, fast_with_wd, fast_no_wd)`` for the BatchEnsemble optimizer.

    Mirrors the imagenet baseline: shared kernel and non-BE params get the recipe's lr and
    weight decay, per-member bias gets the slower lr with weight decay, and ``r`` / ``s``
    get the slower lr with weight decay disabled.
    """
    fast_no_wd_ids: set[int] = set()
    fast_wd_ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, BatchEnsembleLinear | BatchEnsembleConv2d):
            fast_no_wd_ids.add(id(module.r))
            fast_no_wd_ids.add(id(module.s))
            fast_wd_ids.add(id(module.bias))

    shared: list[nn.Parameter] = []
    fast_wd: list[nn.Parameter] = []
    fast_no_wd: list[nn.Parameter] = []
    for p in model.parameters():
        if id(p) in fast_no_wd_ids:
            fast_no_wd.append(p)
        elif id(p) in fast_wd_ids:
            fast_wd.append(p)
        else:
            shared.append(p)
    return shared, fast_wd, fast_no_wd


@train_model.register(BatchEnsemblePredictor)
def _(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    cfg: DictConfig,
    device: torch.device,
    run: Any,  # noqa: ANN401
    train_kwargs: dict[str, Any],
) -> None:
    """Train a BatchEnsemble predictor with the recipe of :cite:`wen2020batchensemble`.

    The shared kernel uses the recipe's full lr and weight decay; ``r``, ``s``, and per-member
    biases use ``fast_weight_lr_multiplier * base_lr``; ``r`` and ``s`` have weight decay disabled.
    """
    fast_lr_mult = train_kwargs.get("fast_weight_lr_multiplier", 0.25)
    base_lr = float(cfg.optimizer.params.lr)
    fast_lr = base_lr * fast_lr_mult

    shared, fast_wd, fast_no_wd = _split_batchensemble_params(model)
    param_groups: list[dict[str, Any]] = [{"params": shared}]
    if fast_wd:
        param_groups.append({"params": fast_wd, "lr": fast_lr})
    if fast_no_wd:
        param_groups.append({"params": fast_no_wd, "lr": fast_lr, "weight_decay": 0.0})

    _training_loop(
        model,
        train_loader,
        val_loader,
        cfg,
        device,
        run,
        train_kwargs,
        train_fn=train_epoch_batchensemble,
        val_fn=validate_batchensemble,
        param_groups=param_groups,
    )


@train_model.register(DarePredictor)
def _(
    model: DarePredictor,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    cfg: DictConfig,
    device: torch.device,
    run: Any,  # noqa: ANN401
    train_kwargs: dict[str, Any],
) -> None:
    """Train a DARE ensemble: each member with cross-entropy minus the anti-regularizer."""
    for i, member in enumerate(model):
        _training_loop(
            member,
            train_loader,
            val_loader,
            cfg,
            device,
            run,
            train_kwargs,
            train_fn=train_epoch_dare,
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


def _training_loop_relative_likelihood(  # noqa: PLR0912, PLR0915
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    cfg: DictConfig,
    device: torch.device,
    run: Any,  # noqa: ANN401
    train_fn: Callable[..., float],
    val_fn: Callable[..., ValidationMetrics],
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

    epoch_key = f"{log_prefix.rstrip('/')}_epoch"
    wandb.define_metric(epoch_key, hidden=True)
    wandb.define_metric(f"{log_prefix}*", step_metric=epoch_key)

    stopped = False
    for epoch in tqdm(range(cfg.epochs), desc=f"{log_prefix}Epoch"):
        model.train()
        running_loss = 0.0
        for inputs_, targets_ in tqdm(train_loader):
            inputs = inputs_.to(device, non_blocking=True)
            if device.type == "cuda" and inputs.ndim >= 4:
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
            epoch_key: epoch,
            f"{log_prefix}train_loss": running_loss,
            f"{log_prefix}relative_likelihood": relative_likelihood,
        }
        if val_loader:
            metrics = val_fn(model, val_loader, device, amp_enabled)
            log_data.update({f"{log_prefix}val_{k}": v for k, v in metrics.items()})
            val_loss = metrics["loss"]
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
def train_model_bayesian(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    cfg: DictConfig,
    device: torch.device,
    run: Any,  # noqa: ANN401
    train_kwargs: dict[str, Any],
    log_prefix: str = "",
) -> None:
    """Train a BayesianPredictor with ELBO loss.

    The KL penalty is set to 1/N (N = dataset size) following
    Blundell et al., "Weight Uncertainty in Neural Networks", ICML 2015.

    Args:
        model: The Bayesian predictor to train.
        train_loader: Training data loader.
        val_loader: Validation data loader, or ``None`` to skip validation.
        cfg: Hydra config with training hyperparameters.
        device: Device to train on.
        run: Weights & Biases run for logging.
        train_kwargs: Method-specific training keyword arguments.
        log_prefix: Prefix for W&B log keys (e.g. ``"member_0/"``). Defaults to empty
            so dispatched calls retain the original single-BNN logging behavior.
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
        log_prefix=log_prefix,
    )


@train_model.register(CredalBNNPredictor)
def _(
    model: CredalBNNPredictor,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    cfg: DictConfig,
    device: torch.device,
    run: Any,  # noqa: ANN401
    train_kwargs: dict[str, Any],
) -> None:
    """Train a credal BNN by training each member as an independent Bayesian predictor.

    Each member is trained with the same ELBO + KL-penalty recipe as a single
    :class:`BayesianPredictor` via :func:`train_model_bayesian`, and gets its own
    W&B namespace via ``log_prefix=f"member_{i}/"`` so per-member learning curves
    do not clobber each other.
    """
    for i, member in enumerate(model):
        train_model_bayesian(
            member,
            train_loader,
            val_loader,
            cfg,
            device,
            run,
            train_kwargs,
            log_prefix=f"member_{i}/",
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

    # Train first member fully; relative_likelihood is 1.0 by definition for the reference model
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
        extra_metrics={"member_0/relative_likelihood": 1.0},
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
        if device.type == "cuda" and inputs.ndim >= 4:
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


@train_model.register(DUQPredictor)
def _(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    cfg: DictConfig,
    device: torch.device,
    run: Any,  # noqa: ANN401
    train_kwargs: dict[str, Any],
) -> None:
    """Train a DUQ predictor :cite:`vanamersfoortDUQ2020`.

    Disables AMP because the gradient penalty requires a stable second-order
    autograd graph that ``torch.amp.autocast`` does not support across all
    backends. The standard training loop is otherwise reused: it dispatches to
    ``train_epoch_duq`` (BCE on kernel values + gradient penalty + EMA
    centroid update) and ``validate_duq`` via the flexdispatch registry.
    """
    cfg_ = cfg.copy()
    if cfg_.get("amp", False):
        print("DUQ: disabling AMP (incompatible with the second-order gradient penalty).")
        cfg_.amp = False
    _training_loop(
        model,
        train_loader,
        val_loader,
        cfg_,
        device,
        run,
        train_kwargs,
        train_fn=train_epoch,  # ty: ignore[invalid-argument-type]
        val_fn=validate,
    )


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
    # The density head buffer is ~16 GB at ImageNet scale and is unused during training
    # Parking it on CPU during training decreases footprint sufficiently to use H200 cards
    density_device = next(model.density_head.buffers()).device  # ty: ignore[unresolved-attribute]
    model.density_head.to("cpu")
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
    model.density_head.to(density_device)

    amp_enabled = cfg.get("amp", False)
    _fit_ddu_density_head(model, train_loader, device, amp_enabled)
    run.summary["ddu_gmm_fitted"] = True


@torch.no_grad()
def _collect_deup_error_targets(
    model_: Any,  # noqa: ANN401
    error_head_loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect ``(stationarizing_features, log10_ce_target)`` pairs over the OOS loader.

    The target follows the reference implementation: ``log10(CE_per_sample)``
    clamped from below at ``-5``, matching the scale used to train the error head.
    Encoder features are **not** returned — the error head takes only
    stationarizing features.  All returned tensors are on CPU.
    """
    bce_criterion = nn.BCELoss(reduction="none")
    model_.eval()
    all_phi: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    for inputs_, targets_ in tqdm(error_head_loader, desc="Collecting DEUP error targets"):
        inputs = inputs_.to(device, non_blocking=True)
        if device.type == "cuda" and inputs.ndim >= 4:
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        targets = targets_.to(device, non_blocking=True)
        with torch.amp.autocast(device.type, enabled=amp_enabled):
            features = model_.encoder(inputs)
            logits = model_.classification_head(features)
            phi = model_._compute_stationarizing_features(features, logits)  # noqa: SLF001
        # BCELoss is unsafe inside autocast; compute in float32
        probs = torch.softmax(logits.float(), dim=-1)
        one_hot = nn.functional.one_hot(targets, num_classes=probs.size(-1)).float()
        # Sum per-class BCE over classes, matching the reference implementation
        # (Lahlou et al. use BCELoss.sum(1) as the per-sample loss target)
        per_sample_bce = bce_criterion(probs, one_hot).sum(dim=-1)
        log10_target = torch.clamp(torch.log10(per_sample_bce.clamp(min=1e-10)), min=-5.0)
        all_phi.append(phi.detach().cpu())
        all_targets.append(log10_target.detach().cpu())
    return torch.cat(all_phi), torch.cat(all_targets)


def _train_deup_error_head_loop(
    model_: Any,  # noqa: ANN401
    phi_all: torch.Tensor,
    targets_all: torch.Tensor,
    device: torch.device,
    batch_size: int,
    epochs: int,
    lr: float,
    momentum: float,
    run: Any,  # noqa: ANN401
) -> None:
    """Train ``error_head`` with SGD+MSE against log10-scaled CE targets.

    ``phi_all`` contains the stationarizing features (only); encoder features
    are not passed to the error head.
    """
    model_.error_head.train()
    model_.error_head.to(device)
    optimizer = torch.optim.SGD(model_.error_head.parameters(), lr=lr, momentum=momentum)
    dataset = torch.utils.data.TensorDataset(phi_all, targets_all)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for phi_cpu, tgt_cpu in loader:
            phi = phi_cpu.to(device, non_blocking=True)
            tgt = tgt_cpu.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model_.error_head(phi), tgt)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        run.log({"deup/error_head_mse": epoch_loss / max(n_batches, 1), "deup/error_head_epoch": epoch})
    model_.error_head.eval()


def _fit_deup_error_head(
    model: DEUPPredictor,
    train_loader: DataLoader,
    error_head_loader: DataLoader,
    device: torch.device,
    train_kwargs: dict[str, Any],
    run: Any,  # noqa: ANN401
    amp_enabled: bool = False,
    batch_size: int = 256,
) -> None:
    """Train the DEUP error head on out-of-sample log10-scaled CE targets.

    Phase-2 training of DEUP :cite:`lahlou2021deup`.
    The ``encoder`` and ``classification_head`` are frozen.

    Steps:
    1. Fit each stationarizing feature provider on the training set.
    2. Collect ``(stationarizing_features, log10_ce_target)`` pairs from
       ``error_head_loader`` (calibration split when available).
    3. Train ``error_head`` with SGD+MSE against ``log10(CE_per_sample)``
       clamped from below at ``-5``.

    Args:
        model: Trained DEUP predictor with frozen ``encoder`` and
            ``classification_head`` and an untrained ``error_head``.
        train_loader: Training data loader used only to fit the
            stationarizing feature providers (e.g. the GMM density head).
        error_head_loader: Out-of-sample loader (calibration split when
            available) providing inputs/labels for computing error targets.
        device: Device to run inference and training on.
        train_kwargs: Reads ``error_head_epochs`` (default 5),
            ``error_head_lr`` (default 0.005), ``error_head_momentum``
            (default 0.9).
        run: Wandb run object used to log phase-2 metrics.
        amp_enabled: Whether to use automatic mixed precision.
        batch_size: Mini-batch size for phase-2 error-head training.
    """
    model_ = cast("Any", model)
    providers: list[Any] = list(getattr(model_, "providers", []))

    for provider in providers:
        provider.to(device)
        provider.fit(model_.encoder, model_.classification_head, train_loader, device, amp_enabled)

    phi_all, targets_all = _collect_deup_error_targets(model_, error_head_loader, device, amp_enabled)

    _train_deup_error_head_loop(
        model_,
        phi_all,
        targets_all,
        device,
        batch_size=batch_size,
        epochs=int(train_kwargs.get("error_head_epochs", 5)),
        lr=float(train_kwargs.get("error_head_lr", 0.005)),
        momentum=float(train_kwargs.get("error_head_momentum", 0.9)),
        run=run,
    )
    run.summary["deup_error_head_fitted"] = True


@train_model.register(DEUPPredictor)
def _(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    cfg: DictConfig,
    device: torch.device,
    run: Any,  # noqa: ANN401
    train_kwargs: dict[str, Any],
) -> None:
    """Train a DEUP predictor :cite:`lahlou2021deup`.

    Phase 1 trains ``encoder`` and ``classification_head`` end-to-end with
    standard cross-entropy, identical to a plain classifier.  The
    ``error_head`` parameters are excluded from the optimizer so they receive
    no gradient updates in this phase.

    Phase 2 freezes the main model and trains ``error_head`` on per-sample
    cross-entropy targets computed from out-of-sample data (see
    ``_fit_deup_error_head``).  When a calibration loader is available
    (``cfg.cal_split > 0``), it is used as the source of out-of-sample
    losses; otherwise the validation loader is used as a fallback, with
    the caveat that best-model selection / early stopping in phase 1
    biases val losses downward and consequently biases the trained
    error head.

    A loader for phase-2 targets is required; if neither a calibration
    nor a validation loader is provided a ``ValueError`` is raised.
    """
    error_head_loader = train_kwargs.get("cal_loader") or val_loader
    if error_head_loader is None:
        msg = "DEUP requires a calibration or validation loader for phase-2 error-head training."
        raise ValueError(msg)
    if train_kwargs.get("cal_loader") is None:
        print(
            "DEUP: no calibration loader configured (cal_split=0); falling back to "
            "the validation loader for phase-2 error-head targets. Note that phase-1 "
            "best-model selection on this loader biases per-sample CE downward."
        )

    # Phase 1: train encoder + classification_head only (exclude error_head).
    # Pass their parameters explicitly so the optimizer never touches error_head.
    phase1_params = [{"params": list(model.encoder.parameters()) + list(model.classification_head.parameters())}]  # ty:ignore[unresolved-attribute]

    # Strip cal_loader from train_kwargs forwarded into the inner training loop;
    # it is consumed only here and would otherwise leak into train_fn / val_fn.
    inner_train_kwargs = {k: v for k, v in train_kwargs.items() if k != "cal_loader"}

    _training_loop(
        model,
        train_loader,
        val_loader,
        cfg,
        device,
        run,
        inner_train_kwargs,
        train_fn=train_epoch_deup,
        val_fn=validate_deup,
        param_groups=phase1_params,
    )

    # Phase 2: fit any stationarizing-feature providers on training data,
    # then the error head on out-of-sample losses.
    amp_enabled = cfg.get("amp", False)
    _fit_deup_error_head(
        model, train_loader, error_head_loader, device, inner_train_kwargs, run, amp_enabled, cfg.batch_size
    )


@train_model.register(BaseLaplace)
def _(
    model: BaseLaplace,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    cfg: DictConfig,
    device: torch.device,
    run: Any,  # noqa: ANN401
    train_kwargs: dict[str, Any],
) -> None:
    """Fine-tune the underlying network, then fit the Laplace posterior post-hoc.

    Phase 1 runs the standard supervised loop on the underlying ``nn.Module`` (laplace-torch's wrapped model).
    Phase 2 calls ``BaseLaplace.fit(train_loader)`` and, if ``train_kwargs["optimize_prior"]`` is set, tunes
    the prior precision by marginal-likelihood maximization.
    """
    # Extract model from BaseLaplace, if we do last layer mode, model.model is a FeatureExtractor, so unwrap it
    inner_model = model.model.model if isinstance(model.model, FeatureExtractor) else model.model
    # Problems with cuda + triton + compile if we compile this model, so we set a flag to skip it.
    inner_model._probly_skip_compile = True  # ty: ignore[unresolved-attribute]  # noqa: SLF001
    _training_loop(
        inner_model,
        train_loader,
        val_loader,
        cfg,
        device,
        run,
        train_kwargs,
        train_fn=train_epoch_cross_entropy,
        val_fn=validate_cross_entropy,
    )
    fit_loader = data.build_laplace_fit_loader(train_loader, cfg, train_kwargs)
    model.fit(fit_loader)
    run.summary["laplace_fitted"] = True
    if train_kwargs.get("optimize_prior", False):
        # ``pred_type`` is required by laplace-torch's signature but unused when ``method='marglik'``
        # (marglik works directly on the closed-form log-marginal-likelihood); any value is fine.
        model.optimize_prior_precision(
            pred_type="glm",
            method="marglik",
            n_steps=train_kwargs.get("n_steps", 100),
            lr=train_kwargs.get("lr", 0.1),
        )
        run.summary["laplace_prior_precision"] = float(model.prior_precision)


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
    num_classes = metadata.DATASETS[cfg.dataset].num_classes

    logits_train, targets_train = utils.collect_outputs_targets_raw(
        model_.predictor,
        train_loader,
        device,
        amp_enabled,
    )
    lower, upper = compute_efficient_credal_prediction_bounds(
        logits_train,
        targets_train,
        num_classes=num_classes,
        alpha=alpha,
    )
    lower_t = lower.to(device)
    upper_t = upper.to(device)

    # Initialize the buffers if they are None, otherwise copy in-place
    if model_.lower is None:
        model_.lower = lower_t
        model_.upper = upper_t
    else:
        model_.lower.copy_(lower_t)
        model_.upper.copy_(upper_t)

    run.summary["efficient_credal_alpha"] = alpha


def _adjust_batch_size_for_method(cfg: DictConfig) -> None:
    """Divide ``cfg.batch_size`` for methods whose per-step memory is a multiple of the regular cost.

    - BatchEnsemble: scales by ``num_members`` (it tiles inputs along the batch dim).
    - credal_net: scales by 2 (every interval layer doubles channels, params, and activations).
    """
    name = cfg.method.name.lower()
    if name == "batchensemble":
        n = int(cfg.method.params.num_members)
        original = int(cfg.batch_size)
        cfg.batch_size = original // n
        print(f"BatchEnsemble: scaled batch_size {original} -> {cfg.batch_size} (num_members={n})")
    elif name == "credal_net":
        original = int(cfg.batch_size)
        cfg.batch_size = original // 2
        print(f"credal_net: scaled batch_size {original} -> {cfg.batch_size} (interval layers double per-step memory)")


@hydra.main(version_base=None, config_path="configs/", config_name="train")
def main(cfg: DictConfig) -> None:
    """Run the training script."""
    _adjust_batch_size_for_method(cfg)

    print("=== Training configuration ===")
    print(OmegaConf.to_yaml(cfg))

    _validate_supervised_loss_config(cfg)
    _validate_conformal_config(cfg)

    run_id = wandb.util.generate_id()
    seed = cfg.get("seed", None)
    utils.set_seed(seed)

    run_name = f"{cfg.method.name}_{cfg.base_model}_{cfg.dataset}{utils.supervised_loss_name_suffix(cfg)}_{run_id}"
    run = wandb.init(
        id=run_id,
        name=run_name,
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
    cal_loader = loaders.calibration
    test_loader = loaders.test

    num_classes = metadata.DATASETS[cfg.dataset].num_classes

    method_kwargs = _build_method_kwargs(cfg)
    train_kwargs = _build_train_kwargs(cfg)

    context = BuildContext(
        base_model_name=cfg.base_model,
        model_type=cfg.model_type,
        num_classes=num_classes,
        pretrained=cfg.pretrained,
        train_loader=train_loader,
    )
    model = build_model(cfg.method.name, method_kwargs, context)
    _move_to_device(model, device)

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

    if cal_loader is not None and cfg.method.name == "deup":
        train_kwargs.setdefault("cal_loader", cal_loader)
    train_model(model, train_loader, val_loader, cfg, device, run, train_kwargs)

    # Release training DataLoaders and force GC to terminate persistent training
    # workers before spawning test workers. On network filesystems (NFS, Lustre),
    # persistent training workers hold open file handles to training shards; their
    # combined count with test worker handles can exceed per-job limits and cause
    # test workers to SIGABRT when they cannot open additional shard files.
    del train_loader, val_loader, cal_loader, loaders
    gc.collect()

    test_metrics = evaluate(model, test_loader, device, cfg.get("amp", False), **train_kwargs)
    run.summary.update(test_metrics)
    run.log(data=test_metrics)

    _save_checkpoint_artifact(model, cfg, test_metrics)

    run.finish()


if __name__ == "__main__":
    main()
