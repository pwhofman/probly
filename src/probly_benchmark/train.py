"""Script to train models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable


import pathlib
import tempfile

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch import nn, optim
from torch.amp import GradScaler
from tqdm import tqdm
import wandb

from probly_benchmark import data, metadata, utils
from probly_benchmark.builders import BuildContext, build_model
from probly_benchmark.paths import CHECKPOINT_PATH
from probly_benchmark.train_funcs import EarlyStopping, evaluate, train_epoch, validate

OPTIMIZERS = {
    "adam": optim.Adam,
    "sgd": optim.SGD,
}


def get_optimizer(name: str, params: Iterable[nn.Parameter], **kwargs: Any) -> optim.Optimizer:  # noqa: ANN401
    """Get optimizer function."""
    name = name.lower()
    if name not in OPTIMIZERS:
        msg = f"Unknown optimizer: {name}"
        raise ValueError(msg)
    return OPTIMIZERS[name](params, **kwargs)


SCHEDULERS: dict[str, type[optim.lr_scheduler.LRScheduler] | None] = {
    "cosine": optim.lr_scheduler.CosineAnnealingLR,
    "step": optim.lr_scheduler.StepLR,
    "plateau": optim.lr_scheduler.ReduceLROnPlateau,
}


def get_scheduler(
    name: str,
    optimizer: optim.Optimizer,
    epochs: int,
    **kwargs: Any,  # noqa: ANN401
) -> optim.lr_scheduler.LRScheduler | None:
    """Get learning rate scheduler.

    Args:
        name: Scheduler name.
        optimizer: Optimizer to schedule.
        epochs: Total number of training epochs. Used as default ``T_max`` for cosine.
        **kwargs: Additional scheduler-specific kwargs. An explicit ``T_max`` here
            overrides the ``epochs`` default for cosine.
    """
    name = name.lower()
    if name not in SCHEDULERS:
        msg = f"Unknown scheduler: {name}"
        raise ValueError(msg)
    cls = SCHEDULERS[name]
    if cls is None:
        return None
    if name == "cosine":
        kwargs.setdefault("T_max", epochs)
    return cls(optimizer, **kwargs)


@hydra.main(version_base=None, config_path="configs/", config_name="train")
def main(cfg: DictConfig) -> None:  # noqa: PLR0912, PLR0915
    """Run the training script."""
    print("=== Training configuration ===")
    print(OmegaConf.to_yaml(cfg))

    run = wandb.init(
        project="test",
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),  # ty: ignore
        mode="online" if cfg.wandb else "disabled",
        save_code=True,
    )

    device = utils.get_device(cfg.get("device", None))
    print(f"Running on device: {device}")
    utils.set_seed(cfg.seed) if cfg.get("seed", None) else None

    train_loader, val_loader, test_loader = data.get_data_train(
        cfg.dataset,
        use_validation=cfg.validate,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers,
        shuffle=True,
    )

    num_classes = metadata.DATASETS[cfg.dataset].num_classes

    method_kwargs: dict[str, Any] = OmegaConf.to_container(cfg.method.params, resolve=True) or {}  # ty: ignore[invalid-assignment]
    train_kwargs: dict[str, Any] = OmegaConf.to_container(cfg.method.train, resolve=True) or {}  # ty: ignore[invalid-assignment]

    context = BuildContext(
        base_model_name=cfg.base_model,
        num_classes=num_classes,
        pretrained=cfg.pretrained,
        train_loader=train_loader,
    )
    model = build_model(cfg.method.name, method_kwargs, context).to(device)
    model.forward = torch.compile(model.forward)  # can only

    optimizer = get_optimizer(cfg.optimizer.name, model.parameters())

    scheduler = get_scheduler(cfg.scheduler.name, optimizer, cfg.epochs, **cfg.scheduler.get("params", {}))

    early_stopping = EarlyStopping(patience=cfg.early_stopping.patience) if cfg.early_stopping.patience else None

    if not cfg.validate:
        if cfg.scheduler.name.lower() == "plateau":
            msg = "ReduceLROnPlateau scheduler requires `validate: true` in the config."
            raise ValueError(msg)
        if early_stopping is not None:
            msg = "Early stopping requires `validate: true` in the config. Disable early stopping or enable validation."
            raise ValueError(msg)

    grad_clip_norm = cfg.get("grad_clip_norm", None)
    amp_enabled = cfg.get("amp", False)
    scaler = GradScaler(device.type) if amp_enabled else None

    for epoch in tqdm(range(cfg.epochs), desc="Epoch"):
        model.train()
        running_loss = 0.0
        for inputs_, targets_ in tqdm(train_loader):
            inputs, targets = inputs_.to(device, non_blocking=True), targets_.to(device, non_blocking=True)
            running_loss += train_epoch(
                model,
                inputs,
                targets,
                optimizer,
                grad_clip_norm=grad_clip_norm,
                amp_enabled=amp_enabled,
                scaler=scaler,
                **train_kwargs,
            )
        running_loss /= len(train_loader)

        val_loss: float | None = None
        if val_loader:
            val_loss = validate(model, val_loader, device, amp_enabled, **train_kwargs)

        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                # val_loss is guaranteed non-None by upfront validate/scheduler check
                scheduler.step(val_loss)  # ty: ignore[invalid-argument-type]
            else:
                scheduler.step()

        log_data = {"train_loss": running_loss}
        if val_loss is not None:
            log_data["val_loss"] = val_loss
        run.log(data=log_data)

        if early_stopping is not None and val_loss is not None and early_stopping.should_stop(val_loss):
            run.summary["early_stopped"] = True
            print(f"Early stopping at epoch {epoch}")
            break
    else:
        run.summary["early_stopped"] = False

    test_metrics = evaluate(model, test_loader, device, amp_enabled, **train_kwargs)
    run.summary.update(test_metrics)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        "metrics": test_metrics,
    }

    name = f"{cfg.method.name}_{cfg.base_model}_{cfg.dataset}"

    if cfg.save_to_disk:
        path = pathlib.Path(CHECKPOINT_PATH).joinpath(f"{name}.pt")
        torch.save(checkpoint, path)

        artifact = wandb.Artifact(
            name=name,
            type="model",
            metadata=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),  # ty: ignore
        )
        artifact.add_file(str(path))
        wandb.log_artifact(artifact)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp).joinpath(f"{name}.pt")
            torch.save(checkpoint, path)

            artifact = wandb.Artifact(
                name=name,
                type="model",
                metadata=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),  # ty: ignore
            )
            artifact.add_file(str(path))
            wandb.log_artifact(artifact)

    run.finish()


if __name__ == "__main__":
    main()
