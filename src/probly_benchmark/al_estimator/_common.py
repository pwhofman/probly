"""Shared helpers for the AL estimators."""

from __future__ import annotations

from typing import TYPE_CHECKING

from omegaconf import DictConfig, OmegaConf
import torch
from torch import nn

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


def make_train_cfg(cfg: DictConfig) -> DictConfig:
    """Build a DictConfig shaped like ``probly_benchmark.train`` expects.

    The training loop reads ``cfg.epochs``, ``cfg.optimizer``, ``cfg.scheduler``,
    ``cfg.early_stopping``, ``cfg.get("grad_clip_norm")``, and ``cfg.get("amp")``.

    Args:
        cfg: Hydra config from :func:`probly_benchmark.active_learning.main`.

    Returns:
        A new ``DictConfig`` containing only the keys the training loop needs.
    """
    return OmegaConf.create(
        {
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "optimizer": OmegaConf.to_container(cfg.optimizer, resolve=True),
            "scheduler": OmegaConf.to_container(cfg.scheduler, resolve=True),
            "early_stopping": OmegaConf.to_container(cfg.early_stopping, resolve=True),
            "grad_clip_norm": cfg.get("grad_clip_norm", None),
            "amp": False,
        }
    )


def train_cross_entropy(
    model: nn.Module,
    train_loader: DataLoader,
    train_cfg: DictConfig,
    device: torch.device,
) -> None:
    """Vanilla cross-entropy training for baseline models.

    Uses the same optimizer/scheduler/epochs settings as the rest of the
    benchmark pipeline (read from ``train_cfg`` produced by
    :func:`make_train_cfg`) so baseline runs are comparable to wrapped
    methods. Re-creates optimizer/scheduler each call so this is safe to
    invoke once per AL iteration on a fresh model.

    Args:
        model: The base ``nn.Module`` to train. Already moved to ``device``.
        train_loader: Loader over the labeled pool for this AL iteration.
        train_cfg: Output of :func:`make_train_cfg`.
        device: Torch device for the forward/backward pass.
    """
    from probly_benchmark.train import get_optimizer, get_scheduler  # noqa: PLC0415

    optimizer = get_optimizer(
        train_cfg.optimizer["name"],
        model.parameters(),
        **train_cfg.optimizer.get("params", {}),
    )
    scheduler = get_scheduler(
        train_cfg.scheduler["name"],
        optimizer,
        train_cfg.epochs,
        len(train_loader),
        **train_cfg.scheduler.get("params", {}),
    )
    step_per_iter = getattr(scheduler, "_step_per_iter", False)
    criterion = nn.CrossEntropyLoss()
    for _epoch in range(train_cfg.epochs):
        model.train()
        for inputs_, targets_ in train_loader:
            inputs = inputs_.to(device, non_blocking=True)
            targets = targets_.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            if scheduler is not None and step_per_iter:
                scheduler.step()
        if scheduler is not None and not step_per_iter:
            scheduler.step()


@torch.no_grad()
def embed_last_linear(model: nn.Module, x_t: torch.Tensor, pred_batch_size: int) -> torch.Tensor:
    """Capture inputs to the last ``nn.Linear`` via a forward pre-hook.

    Used by the baseline estimator to provide BADGE-compatible penultimate
    features without method-specific introspection.

    Args:
        model: A torch module containing at least one ``nn.Linear``.
        x_t: Input tensor already on the same device as ``model``.
        pred_batch_size: Batch size for the forward pass.

    Returns:
        Tensor of shape ``(len(x_t), penultimate_dim)`` on CPU.

    Raises:
        RuntimeError: If ``model`` contains no ``nn.Linear``.
    """
    last_linear: nn.Linear | None = None
    for module in model.modules():
        if isinstance(module, nn.Linear):
            last_linear = module
    if last_linear is None:
        msg = "No nn.Linear found in model; cannot embed."
        raise RuntimeError(msg)

    captured: list[torch.Tensor] = []

    def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        captured.append(inputs[0].detach().cpu())

    handle = last_linear.register_forward_pre_hook(hook)
    try:
        model.eval()
        parts: list[torch.Tensor] = []
        for start in range(0, len(x_t), pred_batch_size):
            captured.clear()
            model(x_t[start : start + pred_batch_size])
            parts.append(captured[0])
        return torch.cat(parts)
    finally:
        handle.remove()
