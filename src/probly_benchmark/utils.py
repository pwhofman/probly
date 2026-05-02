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

from probly_benchmark import calibration, conformal, metadata
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


def cal_split_name_suffix(cfg: DictConfig | dict) -> str:
    """Return the artifact name suffix for a calibration holdout split."""
    cal_split = float(cfg.get("cal_split", 0.0) or 0.0)
    if cal_split <= 0:
        return ""
    return f"_cal_split{_safe_artifact_token(cal_split)}"


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


def conformal_name_suffix(cfg: DictConfig | dict) -> str:
    """Return the artifact name suffix for split-conformal prediction."""
    name = conformal.get_conformal_name(cfg)
    if name == conformal.DEFAULT_CONFORMAL:
        return ""
    conformal_cfg = cfg.get("conformal", {})
    params = conformal_cfg.get("params") or {}
    parts = [_safe_artifact_token(name)]
    alpha = conformal_cfg.get("alpha", None)
    if alpha is not None:
        parts.append(f"alpha{_safe_artifact_token(alpha)}")
    for key, value in sorted(params.items()):
        parts.append(f"{_safe_artifact_token(key)}{_safe_artifact_token(value)}")
    return f"_{'_'.join(parts)}"


def resolve_artifact_name(
    cfg: DictConfig,
    *,
    include_calibration: bool = True,
    include_conformal: bool = True,
) -> str:
    """Build the wandb artifact name from config fields.

    Matches the naming convention used in train.py.
    """
    calibration_suffix = calibration_name_suffix(cfg) if include_calibration else ""
    conformal_suffix = conformal_name_suffix(cfg) if include_conformal else ""
    return (
        f"{cfg.method.name}_{cfg.base_model}_{cfg.dataset}_{cfg.seed}"
        f"{cal_split_name_suffix(cfg)}"
        f"{supervised_loss_name_suffix(cfg)}"
        f"{calibration_suffix}"
        f"{conformal_suffix}"
    )


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


def _download_checkpoint_from_wandb(
    artifact_name: str,
    entity: str,
    project: str,
    device: torch.device,
) -> tuple[dict[str, Any], str]:
    """Download a model artifact from wandb and return the raw checkpoint."""
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
    target_method: str | None = None,
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

    build_method = target_method if target_method is not None else cfg["method"]["name"]
    model = build_model(build_method, method_params, ctx)
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


def _build_conformal_model_from_checkpoint(
    checkpoint: dict[str, Any],
    entity: str,
    project: str,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Reconstruct a conformal model from source artifact plus conformal-only state."""
    cfg = checkpoint["config"]
    source_artifact = checkpoint[conformal.SOURCE_ARTIFACT_KEY]
    source_checkpoint, _ = _download_checkpoint_from_wandb(source_artifact, entity, project, device)
    if source_checkpoint.get("artifact_type") == conformal.CONFORMAL_ARTIFACT_TYPE:
        msg = f"Conformal source artifact {source_artifact!r} is itself a conformal artifact."
        raise RuntimeError(msg)

    source_model, _ = _build_model_from_checkpoint(source_checkpoint, entity, project, device)
    conformal_model = conformal.apply_conformal(source_model, cfg)
    conformal.load_conformal_state(
        conformal_model,
        cfg,
        checkpoint[conformal.CONFORMAL_STATE_DICT_KEY],
    )
    conformal_model.to(device)
    conformal_model.eval()
    return conformal_model, cfg


def _build_model_from_checkpoint(
    checkpoint: dict[str, Any],
    entity: str,
    project: str,
    device: torch.device,
    target_method: str | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Reconstruct a benchmark model from any supported checkpoint artifact."""
    artifact_type = checkpoint.get("artifact_type")
    if artifact_type == calibration.CALIBRATION_ARTIFACT_TYPE:
        return _build_calibrated_model_from_checkpoint(checkpoint, entity, project, device)
    if artifact_type == conformal.CONFORMAL_ARTIFACT_TYPE:
        return _build_conformal_model_from_checkpoint(checkpoint, entity, project, device)
    return _build_uncalibrated_model_from_checkpoint(checkpoint, device, target_method=target_method)


def load_model_from_wandb(
    artifact_name: str,
    entity: str,
    project: str,
    device: torch.device,
    target_method: str | None = None,
) -> tuple[torch.nn.Module, dict[str, Any], str]:
    """Download a model artifact from wandb and reconstruct the model.

    Calibration artifacts are restored by first loading their source model artifact,
    then applying the configured calibration wrapper and loading calibration-only state.
    """
    checkpoint, run_id = _download_checkpoint_from_wandb(artifact_name, entity, project, device)
    model, cfg = _build_model_from_checkpoint(checkpoint, entity, project, device, target_method=target_method)
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


def _find_existing_run_id(entity: str, project: str, run_name: str) -> str | None:
    """Return the ID of the most recent wandb run with the given display name, or None."""
    api = wandb.Api()
    runs = api.runs(
        f"{entity}/{project}",
        filters={"display_name": run_name},
        order="-created_at",
    )
    try:
        return next(iter(runs)).id
    except StopIteration:
        return None


def init_wandb_for_evaluation(
    cfg: DictConfig,
    run_id: str,
) -> wandb.sdk.wandb_run.Run:
    """Initialise a wandb run for an evaluation script.

    For methods that own their training run (no ``load_from``), the existing
    training run is resumed so that evaluation metrics are attached to it.

    For methods that load weights from another method's artifact (``load_from``
    is set), the run is identified by ``<method>_<base_model>_<dataset>_<seed>``.
    If a run with that name already exists in the project it is resumed, so
    repeated evaluations accumulate metrics on the same run rather than creating
    duplicates. If no such run exists yet, a new one is created with the full
    evaluation config (plus the source ``run_id`` for traceability).

    Args:
        cfg: Hydra evaluation config.
        run_id: Run ID returned by :func:`load_model_for_evaluation`.

    Returns:
        An initialised wandb run. The caller is responsible for calling
        ``run.finish()``.
    """
    if cfg.method.get("load_from"):
        run_name = f"{cfg.method.name}_{cfg.base_model}_{cfg.dataset}_{cfg.seed}"
        existing_id = _find_existing_run_id(cfg.wandb.entity, cfg.wandb.project, run_name)
        if existing_id is not None:
            return wandb.init(
                id=existing_id,
                entity=cfg.wandb.entity,
                project=cfg.wandb.project,
                resume="must",
            )
        # Fetch the source run's config so training-relevant parameters are
        # carried over (e.g. base_model, dataset, seed, training hyperparams).
        # The current eval cfg is merged on top, so method-specific overrides win.
        api = wandb.Api()
        source_run = api.run(f"{cfg.wandb.entity}/{cfg.wandb.project}/{run_id}")
        source_config: dict[str, Any] = dict(source_run.config)
        return wandb.init(
            name=run_name,
            config={
                **source_config,
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
