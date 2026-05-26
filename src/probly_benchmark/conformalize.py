"""Split-conformalization script for benchmark models."""

from __future__ import annotations

import pathlib
import tempfile
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import wandb
import wandb.util

from probly.evaluation import coverage as compute_coverage, efficiency as compute_efficiency
from probly_benchmark import calibration, conformal, data, utils
from probly_benchmark.paths import CHECKPOINT_PATH


def _log_artifact_file(path: pathlib.Path, artifact_name: str, metadata: dict[str, Any]) -> None:
    """Log a checkpoint file as a wandb model artifact."""
    artifact = wandb.Artifact(name=artifact_name, type="model", metadata=metadata)
    artifact.add_file(str(path))
    wandb.log_artifact(artifact)


def _save_conformal_artifact(
    logit_conformalizer: torch.nn.Module,
    cfg: DictConfig,
    metrics: dict[str, float],
    source_artifact: str,
    source_run_id: str,
) -> None:
    """Save and log the conformal-only checkpoint artifact."""
    metadata = cast("dict[str, Any]", OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
    checkpoint = {
        "artifact_type": conformal.CONFORMAL_ARTIFACT_TYPE,
        conformal.CONFORMAL_STATE_DICT_KEY: logit_conformalizer.state_dict(),
        "config": metadata,
        "metrics": metrics,
        conformal.SOURCE_ARTIFACT_KEY: source_artifact,
        conformal.SOURCE_RUN_ID_KEY: source_run_id,
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


def _validate_selected_split(cfg: DictConfig) -> tuple[bool, str]:
    """Validate and return the split selected for conformal calibration."""
    use_val_as_cal = cfg.conformal.get("use-val-as-cal", False)
    split_name = "val_split" if use_val_as_cal else "cal_split"
    split_value = cfg.val_split if use_val_as_cal else cfg.get("cal_split", 0.0)
    if split_value <= 0:
        msg = (
            f"Split-conformal prediction requires `{split_name} > 0` when `conformal.use-val-as-cal={use_val_as_cal}`."
        )
        raise ValueError(msg)
    return use_val_as_cal, split_name


@hydra.main(version_base=None, config_path="configs/", config_name="conformalize")
def main(cfg: DictConfig) -> None:
    """Run split-conformalization for a trained benchmark model."""
    print("=== Conformalization configuration ===")
    print(OmegaConf.to_yaml(cfg))

    calibration.validate_calibration_config(cfg)
    conformal.validate_conformal_config(cfg, allow_none=False)
    use_val_as_cal, split_name = _validate_selected_split(cfg)

    seed = cfg.get("seed", None)
    utils.set_seed(seed)

    run_id = wandb.util.generate_id()
    loss_suffix = utils.supervised_loss_name_suffix(cfg)
    calibration_suffix = utils.calibration_name_suffix(cfg)
    conformal_suffix = utils.conformal_name_suffix(cfg)
    run_name = (
        f"{cfg.method.name}_{cfg.base_model}_{cfg.dataset}{loss_suffix}{calibration_suffix}{conformal_suffix}_{run_id}"
    )
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

    source_artifact = utils.resolve_artifact_name(cfg, include_conformal=False)
    model, _, source_run_id = utils.load_model_from_wandb(
        source_artifact,
        cfg.wandb.entity,
        cfg.wandb.project,
        device,
    )
    print(f"Loaded model {source_artifact} from wandb run: {source_run_id}")

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
        msg = (
            f"Split-conformal prediction requires `{split_name} > 0` when `conformal.use-val-as-cal={use_val_as_cal}`."
        )
        raise ValueError(msg)

    logits, targets = utils.collect_outputs_targets_raw(model, cal_loader, device, cfg.get("amp", False))
    logit_conformalizer = conformal.fit_logit_conformalizer(cfg, logits, targets)
    conformal_sets = conformal.predict_conformal_sets(logit_conformalizer, logits)

    metrics = {
        "coverage": float(compute_coverage(conformal_sets, targets)),
        "efficiency": float(compute_efficiency(conformal_sets)),
    }
    metrics.update(conformal.extract_conformal_metrics(cfg, logit_conformalizer))
    log_metrics = {f"conformal/{key}": value for key, value in metrics.items()}

    run.summary.update(log_metrics)
    run.log(data=log_metrics)
    _save_conformal_artifact(logit_conformalizer, cfg, log_metrics, source_artifact, source_run_id)
    run.finish()


if __name__ == "__main__":
    main()
