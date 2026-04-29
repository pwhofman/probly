"""Post-hoc calibration script for benchmark models."""

from __future__ import annotations

import pathlib
import tempfile
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.nn import functional as F
import wandb
import wandb.util

from probly.train.calibration.torch import ExpectedCalibrationError
from probly_benchmark import calibration, data, utils
from probly_benchmark.paths import CHECKPOINT_PATH


def _log_artifact_file(path: pathlib.Path, artifact_name: str, metadata: dict[str, Any]) -> None:
    """Log a checkpoint file as a wandb model artifact."""
    artifact = wandb.Artifact(name=artifact_name, type="model", metadata=metadata)
    artifact.add_file(str(path))
    wandb.log_artifact(artifact)


def _classification_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    """Compute calibration metrics from logits and class labels."""
    targets = targets.long()
    probs = torch.softmax(logits, dim=1)
    return {
        "val_nll": float(F.cross_entropy(logits, targets).item()),
        "val_ece": float(ExpectedCalibrationError()(probs, targets).item()),
    }


def _save_calibrated_artifact(
    logit_calibrator: torch.nn.Module,
    cfg: DictConfig,
    metrics: dict[str, float],
    source_run_id: str,
) -> None:
    """Save and log the calibration-only checkpoint artifact."""
    metadata = cast("dict[str, Any]", OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
    checkpoint = {
        "artifact_type": calibration.CALIBRATION_ARTIFACT_TYPE,
        calibration.CALIBRATION_STATE_DICT_KEY: logit_calibrator.state_dict(),
        "config": metadata,
        "metrics": metrics,
        calibration.SOURCE_ARTIFACT_KEY: utils.resolve_artifact_name(cfg, include_calibration=False),
        calibration.SOURCE_RUN_ID_KEY: source_run_id,
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


@hydra.main(version_base=None, config_path="configs/", config_name="calibrate")
def main(cfg: DictConfig) -> None:
    """Run post-hoc calibration for a trained benchmark model."""
    print("=== Calibration configuration ===")
    print(OmegaConf.to_yaml(cfg))

    calibration.validate_calibration_config(cfg, allow_none=False)
    seed = cfg.get("seed", None)
    utils.set_seed(seed)

    run_id = wandb.util.generate_id()
    loss_suffix = utils.supervised_loss_name_suffix(cfg)
    calibration_suffix = utils.calibration_name_suffix(cfg)
    run_name = f"{cfg.method.name}_{cfg.base_model}_{cfg.dataset}{loss_suffix}{calibration_suffix}_{run_id}"
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

    source_artifact = utils.resolve_artifact_name(cfg, include_calibration=False)
    model, _, source_run_id = utils.load_model_from_wandb(
        source_artifact,
        cfg.wandb.entity,
        cfg.wandb.project,
        device,
    )
    print(f"Loaded model {source_artifact} from wandb run: {source_run_id}")

    use_val_as_cal = cfg.calibration.get("use-val-as-cal", True)
    split_name = "val_split" if use_val_as_cal else "cal_split"
    split_value = cfg.val_split if use_val_as_cal else cfg.get("cal_split", 0.0)
    if split_value <= 0:
        msg = f"Post-hoc calibration requires `{split_name} > 0` when `calibration.use-val-as-cal={use_val_as_cal}`."
        raise ValueError(msg)

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
        shuffle=False,
    )
    cal_loader = loaders.validation if use_val_as_cal else loaders.calibration
    if cal_loader is None:
        msg = f"Post-hoc calibration requires `{split_name} > 0` when `calibration.use-val-as-cal={use_val_as_cal}`."
        raise ValueError(msg)

    logits, targets = utils.collect_outputs_targets_raw(model, cal_loader, device, cfg.get("amp", False))
    metrics_before = _classification_metrics(logits, targets)

    logit_calibrator = calibration.fit_logit_calibrator(cfg, logits, targets)
    calibrated_logits = calibration.predict_calibrated_logits(logit_calibrator, logits)
    metrics = _classification_metrics(calibrated_logits, targets)
    metrics.update(calibration.extract_calibration_metrics(cfg, logit_calibrator))

    log_metrics = {
        "calibration/val_nll_before": metrics_before["val_nll"],
        "calibration/val_ece_before": metrics_before["val_ece"],
        "calibration/val_nll": metrics["val_nll"],
        "calibration/val_ece": metrics["val_ece"],
    }
    for key, value in metrics.items():
        if key not in {"val_nll", "val_ece"}:
            log_metrics[f"calibration/{key}"] = value

    run.summary.update(log_metrics)
    run.log(data=log_metrics)
    _save_calibrated_artifact(logit_calibrator, cfg, log_metrics, source_run_id)
    run.finish()


if __name__ == "__main__":
    main()
