"""Utils for benchmarking."""

from __future__ import annotations

import pathlib
import random
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from tqdm import tqdm
import wandb

from probly_benchmark import metadata
from probly_benchmark.builders import BuildContext, build_model

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from probly.representation import Representation
    from probly.representer import Representer


def set_seed(seed: int) -> None:
    """Set seed for reproducibility.

    Args:
        seed: Seed for reproducibility.
    """
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


def resolve_artifact_name(cfg: DictConfig) -> str:
    """Build the wandb artifact name from config fields.

    Matches the naming convention used in train.py.
    """
    return f"{cfg.method.name}_{cfg.base_model}_{cfg.dataset}_{cfg.seed}"


def load_model_from_wandb(
    artifact_name: str,
    entity: str,
    project: str,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], str]:
    """Download a model artifact from wandb and reconstruct the model.

    Args:
        artifact_name: Name of the wandb artifact (without version tag).
        entity: Wandb entity.
        project: Wandb project.
        device: Device to load the model onto.

    Returns:
        A tuple containing:
            - model: The reconstructed model in eval mode.
            - cfg: The saved training config dict.
            - run_id: The ID of the original training run.
    """
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

    run_id = artifact.logged_by().id

    return model, cfg, run_id
