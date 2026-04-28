"""Utils for benchmarking."""

from __future__ import annotations

import pathlib
import random
import secrets
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from omegaconf import OmegaConf
import torch
from tqdm import tqdm
import wandb

from probly_benchmark import metadata
from probly_benchmark.builders import BuildContext, build_model

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch import nn
    from torch.utils.data import DataLoader

    from probly.representation import Representation
    from probly.representer import Representer


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


def resolve_artifact_name(cfg: DictConfig) -> str:
    """Build the wandb artifact name from config fields.

    Matches the naming convention used in train.py.
    """
    return f"{cfg.method.name}_{cfg.base_model}_{cfg.dataset}_{cfg.seed}"


def _parse_load_from(load_from: str | DictConfig) -> tuple[str, int | None]:
    """Parse the ``load_from`` method config field.

    Args:
        load_from: Either a method name string, or a dict with ``method`` and
            optional ``member`` keys.

    Returns:
        A tuple of ``(source_method, member_index)`` where ``member_index`` is
        ``None`` when the full artifact should be loaded.
    """
    if isinstance(load_from, str):
        return load_from, None
    return load_from.method, load_from.get("member")


def load_model_from_wandb(
    artifact_name: str,
    entity: str,
    project: str,
    device: torch.device,
    target_method: str | None = None,
) -> tuple[torch.nn.Module, dict[str, Any], str]:
    """Download a model artifact from wandb and reconstruct the model.

    Args:
        artifact_name: Name of the wandb artifact (without version tag).
        entity: Wandb entity.
        project: Wandb project.
        device: Device to load the model onto.
        target_method: If provided, call this method's factory instead of the
            one stored in the checkpoint. The checkpoint's structural params
            (e.g. ``num_members``) are always used so that the built model
            matches the saved state dict. Use this when a wrapper predictor
            (e.g. ``credal_wrapper``) loads weights from a source artifact
            (e.g. ``ensemble``) and needs to register the model under the
            wrapper's predictor type for correct representer dispatch.

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

    build_method = target_method if target_method is not None else cfg["method"]["name"]
    model = build_model(build_method, method_params, ctx)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    run_id = artifact.logged_by().id

    return model, cfg, run_id


def load_model_for_evaluation(
    cfg: DictConfig,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], str]:
    """Load a model from wandb for evaluation, respecting ``load_from`` config.

    Handles three cases driven by the optional ``load_from`` key in the method
    config:

    - **Absent**: loads the method's own artifact and builds it as-is.
    - **String** (e.g. ``load_from: "ensemble"``): loads the named method's
      artifact and rebuilds the model under the current method type, so that a
      wrapper predictor (e.g. ``credal_wrapper``) can be evaluated using
      weights trained for a source method (e.g. ``ensemble``).
    - **Dict** (e.g. ``load_from: {method: "ensemble", member: 0}``): loads
      the named method's artifact and returns the single ensemble member at
      the given index.

    Args:
        cfg: Hydra evaluation config containing ``method``, ``base_model``,
            ``dataset``, ``seed``, and ``wandb`` fields.
        device: Device to load the model onto.

    Returns:
        A tuple containing:
            - model: The reconstructed model in eval mode.
            - train_cfg: The saved training config dict.
            - run_id: The ID of the original training run.
    """
    load_from = cfg.method.get("load_from")

    if load_from is None:
        artifact_name = resolve_artifact_name(cfg)
        return load_model_from_wandb(artifact_name, cfg.wandb.entity, cfg.wandb.project, device)

    source_method, member_index = _parse_load_from(load_from)
    artifact_name = f"{source_method}_{cfg.base_model}_{cfg.dataset}_{cfg.seed}"

    if member_index is not None:
        model, train_cfg, run_id = load_model_from_wandb(artifact_name, cfg.wandb.entity, cfg.wandb.project, device)
        return cast("torch.nn.ModuleList", model)[member_index], train_cfg, run_id

    return load_model_from_wandb(
        artifact_name,
        cfg.wandb.entity,
        cfg.wandb.project,
        device,
        target_method=cfg.method.name,
    )


def init_wandb_for_evaluation(
    cfg: DictConfig,
    run_id: str,
) -> wandb.sdk.wandb_run.Run:
    """Initialise a wandb run for an evaluation script.

    For methods that own their training run (no ``load_from``), the existing
    training run is resumed so that evaluation metrics are attached to it.
    For methods that load weights from another method's artifact, a fresh run
    is created, identified by ``<method>_<base_model>_<dataset>_<seed>``, with
    the source ``run_id`` stored in the run config for traceability.

    Args:
        cfg: Hydra evaluation config.
        run_id: Run ID returned by :func:`load_model_for_evaluation`.

    Returns:
        An initialised wandb run. The caller is responsible for calling
        ``run.finish()``.
    """
    if cfg.method.get("load_from"):
        return wandb.init(
            name=f"{cfg.method.name}_{cfg.base_model}_{cfg.dataset}_{cfg.seed}",
            config={
                "source_run_id": run_id,
                **cast("dict[str, Any]", OmegaConf.to_container(cfg, resolve=True)),
            },
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
        )
    return wandb.init(
        id=run_id,
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        resume="must",
    )
