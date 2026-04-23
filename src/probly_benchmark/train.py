"""Script to train models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from torch.utils.data import DataLoader


import pathlib
import secrets
import tempfile

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_optimizer import Lamb
import torch
from torch import nn, optim
from torch.amp import GradScaler
from tqdm import tqdm
import wandb
import wandb.util

from flextype import flexdispatch
from probly.method.credal_ensembling import CredalEnsemblingPredictor
from probly.method.credal_relative_likelihood import CredalRelativeLikelihoodPredictor
from probly.method.credal_wrapper import CredalWrapperPredictor
from probly.method.ensemble import EnsemblePredictor
from probly.method.subensemble import SubensemblePredictor
from probly_benchmark import data, metadata, utils
from probly_benchmark.builders import BuildContext, build_model
from probly_benchmark.paths import CHECKPOINT_PATH
from probly_benchmark.train_funcs import (
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
    val_fn: Callable[..., float],
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
    early_stopping = EarlyStopping(patience=cfg.early_stopping.patience) if cfg.early_stopping.patience else None

    grad_clip_norm = cfg.get("grad_clip_norm", None)
    amp_enabled = cfg.get("amp", False)
    scaler = GradScaler(device.type) if amp_enabled else None

    for epoch in tqdm(range(cfg.epochs), desc=f"{log_prefix}Epoch"):
        model.train()
        running_loss = 0.0
        for inputs_, targets_ in tqdm(train_loader):
            inputs = inputs_.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
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
        if val_loader:
            val_loss = val_fn(model, val_loader, device, amp_enabled, **train_kwargs)

        if scheduler is not None and not step_per_iter:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)  # ty: ignore[invalid-argument-type]
            else:
                scheduler.step()

        log_data = {f"{log_prefix}train_loss": running_loss}
        if val_loss is not None:
            log_data[f"{log_prefix}val_loss"] = val_loss
        run.log(data=log_data)

        if early_stopping is not None and val_loss is not None and early_stopping.should_stop(val_loss):
            run.summary[f"{log_prefix}early_stopped"] = True
            print(f"Early stopping at epoch {epoch}")
            break
    else:
        run.summary[f"{log_prefix}early_stopped"] = False


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
    val_fn: Callable[..., float],
    max_ll: float,
    alpha: float,
    batch_check: bool = False,
    log_prefix: str = "",
) -> None:
    """Training loop that stops when the relative likelihood reaches ``alpha``."""
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

    grad_clip_norm = cfg.get("grad_clip_norm", None)
    amp_enabled = cfg.get("amp", False)
    scaler = GradScaler(device.type) if amp_enabled else None

    stopped = False
    for epoch in tqdm(range(cfg.epochs), desc=f"{log_prefix}Epoch"):
        model.train()
        running_loss = 0.0
        for inputs_, targets_ in tqdm(train_loader):
            inputs = inputs_.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
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
        if val_loader:
            val_loss = val_fn(model, val_loader, device, amp_enabled)

        if scheduler is not None and not step_per_iter:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)  # ty: ignore[invalid-argument-type]
            else:
                scheduler.step()

        log_data = {
            f"{log_prefix}train_loss": running_loss,
            f"{log_prefix}relative_likelihood": relative_likelihood,
        }
        if val_loss is not None:
            log_data[f"{log_prefix}val_loss"] = val_loss
        run.log(data=log_data)

        if stopped or relative_likelihood >= alpha:
            run.summary[f"{log_prefix}rl_stopped"] = True
            run.summary[f"{log_prefix}stopped_epoch"] = epoch
            print(f"Relative likelihood stopping at epoch {epoch} (RL={relative_likelihood:.4f})")
            break
    else:
        run.summary[f"{log_prefix}rl_stopped"] = False


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


@hydra.main(version_base=None, config_path="configs/", config_name="train")
def main(cfg: DictConfig) -> None:
    """Run the training script."""
    print("=== Training configuration ===")
    print(OmegaConf.to_yaml(cfg))

    run_id = wandb.util.generate_id()
    seed = cfg.get("seed", None)
    if seed is None:
        seed = secrets.randbelow(2**32)
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

    train_loader, val_loader, test_loader = data.get_data_train(
        cfg.dataset,
        use_validation=cfg.validate,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers,
        prefetch_factor=cfg.get("prefetch_factor", 4),
        shuffle=True,
        seed=seed,
    )

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
    model = build_model(cfg.method.name, method_kwargs, context).to(device)
    # channels_last layout gives a large speedup for conv nets on recent NVIDIA GPUs.
    # only
    # model = model.to(memory_format=torch.channels_last) # noqa: ERA001
    model = model.to(device)

    if not cfg.validate:
        if cfg.scheduler.name.lower() == "plateau":
            msg = "ReduceLROnPlateau scheduler requires `validate: true` in the config."
            raise ValueError(msg)
        if cfg.early_stopping.patience:
            msg = "Early stopping requires `validate: true` in the config. Disable early stopping or enable validation."
            raise ValueError(msg)

    train_model(model, train_loader, val_loader, cfg, device, run, train_kwargs)

    test_metrics = evaluate(model, test_loader, device, cfg.get("amp", False), **train_kwargs)
    run.summary.update(test_metrics)
    run.log(data=test_metrics)

    checkpoint = {
        "model_state_dict": model.state_dict(),
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
