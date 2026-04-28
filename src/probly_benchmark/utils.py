"""Utils for benchmarking."""

from __future__ import annotations

import pathlib
import random
import secrets
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from tqdm import tqdm
import wandb

from probly_benchmark import calibration, metadata
from probly_benchmark.builders import BuildContext, build_model

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch import nn
    from torch.utils.data import DataLoader

    from probly.representation import Representation
    from probly.representer import Representer


DEFAULT_SUPERVISED_LOSS = "cross_entropy"


def set_seed(seed: int | None) -> None:
    """Set seed for reproducibility.

    Args:
        seed: Seed for reproducibility.
    """
    if seed is None:
        seed = secrets.randbelow(2**32)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_id: int | None = None) -> torch.device:
    """Return the best available device, or a specific CUDA device if requested.

    Args:
        device_id: Optional CUDA device ID to use. If None, automatically selects the least utilized CUDA device.
            Ignored if CUDA is not available.

    Returns:
        The selected torch.device.

    """
    if torch.cuda.is_available():
        if device_id is not None:
            return torch.device(f"cuda:{device_id}")
        try:
            us = [torch.cuda.utilization(i) for i in range(torch.cuda.device_count())]
            return torch.device(f"cuda:{us.index(min(us))}")
        except ModuleNotFoundError:
            print("Warning: 'torch.cuda.utilization' is not available. Defaulting to 'cuda:0'.")
            return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def collect_outputs_targets(
    rep: Representer,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    amp_enabled: bool = False,
) -> tuple[Representation, torch.Tensor]:
    """Collect representer outputs and targets from a data loader.

    Args:
        rep: A representer to produce outputs.
        loader: DataLoader to iterate over.
        device: Device to run inference on.
        amp_enabled: Whether to use automatic mixed precision.

    Returns:
        A tuple containing:
            - outputs: List of representer outputs, one per batch.
            - targets: Concatenated target tensor.
    """
    outputs = []
    targets = []

    for inputs, targets_ in tqdm(loader, desc="Batch"):
        if amp_enabled:
            with torch.amp.autocast(device.type):
                outputs_ = rep.predict(inputs.to(device))
        else:
            outputs_ = rep.predict(inputs.to(device))
        outputs.append(outputs_)
        targets.append(targets_)

    return torch.cat(outputs), torch.cat(targets)


@torch.no_grad()
def collect_outputs_targets_raw(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect raw model outputs (e.g. logits) and targets from a data loader.

    Unlike :func:`collect_outputs_targets`, this helper does not go through a
    ``Representer``. It calls ``model(inputs)`` directly and returns the
    concatenated outputs on CPU, which is what callers need when they want the
    model's raw tensor output (for example, logits for a downstream numpy
    computation).

    Args:
        model: The model to evaluate.
        loader: DataLoader to iterate over.
        device: Device to run inference on.
        amp_enabled: Whether to use automatic mixed precision.

    Returns:
        A tuple ``(outputs, targets)`` of CPU tensors.
    """
    model.eval()
    outputs = []
    targets = []

    for inputs_, targets_ in tqdm(loader, desc="Batch"):
        inputs = inputs_.to(device, non_blocking=True)
        if amp_enabled:
            with torch.amp.autocast(device.type):
                outputs_ = model(inputs)
        else:
            outputs_ = model(inputs)
        outputs.append(outputs_.detach().cpu())
        targets.append(targets_.detach().cpu())

    return torch.cat(outputs), torch.cat(targets)


def get_supervised_loss_name(cfg: DictConfig) -> str:
    """Return the configured supervised loss name."""
    supervised_loss = cfg.get("supervised_loss", None)
    if not supervised_loss:
        return DEFAULT_SUPERVISED_LOSS
    return str(supervised_loss.get("name", DEFAULT_SUPERVISED_LOSS)).lower()


def _safe_artifact_token(value: object) -> str:
    """Make a value safe for use in a wandb artifact name segment."""
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in str(value))


def supervised_loss_name_suffix(cfg: DictConfig) -> str:
    """Return the name suffix for non-default supervised training losses."""
    name = get_supervised_loss_name(cfg)
    if name == DEFAULT_SUPERVISED_LOSS:
        return ""
    supervised_loss = cfg.get("supervised_loss", {})
    params = supervised_loss.get("params") or {}
    parts = [_safe_artifact_token(name)]
    for key, value in sorted(params.items()):
        parts.append(f"{_safe_artifact_token(key)}{_safe_artifact_token(value)}")
    return f"_{'_'.join(parts)}"


def calibration_name_suffix(cfg: DictConfig | dict) -> str:
    """Return the artifact name suffix for post-hoc calibration."""
    name = calibration.get_calibration_name(cfg)
    if name == calibration.DEFAULT_CALIBRATION:
        return ""
    calibration_cfg = cfg.get("calibration", {})
    params = calibration_cfg.get("params") or {}
    parts = [_safe_artifact_token(name)]
    for key, value in sorted(params.items()):
        parts.append(f"{_safe_artifact_token(key)}{_safe_artifact_token(value)}")
    return f"_{'_'.join(parts)}"


def resolve_artifact_name(cfg: DictConfig, *, include_calibration: bool = True) -> str:
    """Build the wandb artifact name from config fields.

    Matches the naming convention used in train.py.
    """
    suffix = calibration_name_suffix(cfg) if include_calibration else ""
    return f"{cfg.method.name}_{cfg.base_model}_{cfg.dataset}_{cfg.seed}{supervised_loss_name_suffix(cfg)}{suffix}"


def _download_checkpoint_from_wandb(
    artifact_name: str,
    entity: str,
    project: str,
    device: torch.device,
) -> tuple[dict[str, Any], str]:
    """Download a wandb checkpoint artifact."""
    api = wandb.Api()
    full_name = f"{entity}/{project}/{artifact_name}:latest"

    try:
        artifact = api.artifact(full_name)
    except Exception as e:  # noqa: BLE001
        msg = (
            f"No trained model found for '{artifact_name}' in {entity}/{project}. "
            f"Ensure the model has been trained and logged. Original error: {e}"
        )
        raise RuntimeError(msg) from None

    artifact_dir = artifact.download()

    pt_files = list(pathlib.Path(artifact_dir).glob("*.pt"))
    if len(pt_files) != 1:
        msg = f"Expected exactly one .pt file in artifact, found {len(pt_files)}"
        raise RuntimeError(msg)

    checkpoint = torch.load(pt_files[0], map_location=device, weights_only=False)
    run_id = artifact.logged_by().id
    return checkpoint, run_id


def _build_uncalibrated_model_from_checkpoint(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Reconstruct an uncalibrated benchmark model from a training checkpoint."""
    cfg = checkpoint["config"]

    num_classes = metadata.DATASETS[cfg["dataset"]].num_classes
    method_params = cfg["method"].get("params") or {}
    ctx = BuildContext(
        base_model_name=cfg["base_model"],
        model_type=cfg["model_type"],
        num_classes=num_classes,
        pretrained=cfg.get("pretrained", False),
        train_loader=None,
    )
    model = build_model(cfg["method"]["name"], method_params, ctx)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, cfg


def _build_calibrated_model_from_checkpoint(
    checkpoint: dict[str, Any],
    entity: str,
    project: str,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Reconstruct a calibrated model from source weights plus calibration-only state."""
    cfg = checkpoint["config"]
    source_artifact = checkpoint[calibration.SOURCE_ARTIFACT_KEY]
    source_checkpoint, _ = _download_checkpoint_from_wandb(source_artifact, entity, project, device)
    if source_checkpoint.get("artifact_type") == calibration.CALIBRATION_ARTIFACT_TYPE:
        msg = f"Calibration source artifact {source_artifact!r} is itself a calibration artifact."
        raise RuntimeError(msg)

    model, _ = _build_uncalibrated_model_from_checkpoint(source_checkpoint, device)
    calibrated_model = calibration.apply_calibration(model, cfg)
    calibration.load_calibration_state(
        calibrated_model,
        cfg,
        checkpoint[calibration.CALIBRATION_STATE_DICT_KEY],
    )
    calibrated_model.to(device)
    calibrated_model.eval()
    return calibrated_model, cfg


def load_model_from_wandb(
    artifact_name: str,
    entity: str,
    project: str,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], str]:
    """Download a model artifact from wandb and reconstruct the model.

    Calibration artifacts are restored by first loading their source model artifact,
    then applying the configured calibration wrapper and loading calibration-only state.

    Args:
        artifact_name: Name of the wandb artifact (without version tag).
        entity: Wandb entity.
        project: Wandb project.
        device: Device to load the model onto.

    Returns:
        A tuple containing:
            - model: The reconstructed model in eval mode.
            - cfg: The saved checkpoint config dict.
            - run_id: The ID of the run that logged ``artifact_name``.
    """
    checkpoint, run_id = _download_checkpoint_from_wandb(artifact_name, entity, project, device)

    if checkpoint.get("artifact_type") == calibration.CALIBRATION_ARTIFACT_TYPE:
        model, cfg = _build_calibrated_model_from_checkpoint(checkpoint, entity, project, device)
    else:
        model, cfg = _build_uncalibrated_model_from_checkpoint(checkpoint, device)

    return model, cfg, run_id
