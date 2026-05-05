"""Conformal credal-set calibration on first-order data for benchmark models."""

from __future__ import annotations

import pathlib
import tempfile
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import wandb
import wandb.util

from probly.method.conformal_credal_set import (
    conformal_inner_product,
    conformal_kullback_leibler,
    conformal_total_variation,
    conformal_wasserstein_distance,
)
from probly_benchmark import calibration, conformal, data, utils
from probly_benchmark.paths import CHECKPOINT_PATH


def _conformal_tv(model: torch.nn.Module, **_: Any) -> torch.nn.Module:  # noqa: ANN401
    """Wrap a torch model with total-variation credal-set conformal prediction."""
    return cast("torch.nn.Module", conformal_total_variation(model))


def _conformal_kl(model: torch.nn.Module, **_: Any) -> torch.nn.Module:  # noqa: ANN401
    """Wrap a torch model with KL-divergence credal-set conformal prediction."""
    return cast("torch.nn.Module", conformal_kullback_leibler(model))


def _conformal_wasserstein(model: torch.nn.Module, **_: Any) -> torch.nn.Module:  # noqa: ANN401
    """Wrap a torch model with Wasserstein credal-set conformal prediction."""
    return cast("torch.nn.Module", conformal_wasserstein_distance(model))


def _conformal_inner_product(model: torch.nn.Module, **_: Any) -> torch.nn.Module:  # noqa: ANN401
    """Wrap a torch model with inner-product credal-set conformal prediction."""
    return cast("torch.nn.Module", conformal_inner_product(model))


def _quantile_metrics(conformalizer: torch.nn.Module) -> dict[str, float]:
    """Extract the calibrated conformal quantile."""
    quantile = getattr(conformalizer, "conformal_quantile", None)
    if quantile is None:
        return {}
    return {"quantile": float(quantile)}


_CREDAL_SET_SPEC_KWARGS: dict[str, Any] = {
    "supported_methods": frozenset({"base"}),
    "state_keys": frozenset({"_conformal_quantile"}),
    "metric_extractors": (_quantile_metrics,),
}

_CREDAL_SET_REGISTRY: dict[str, conformal.ConformalSpec] = {
    "conformal_tv": conformal.ConformalSpec(transform=_conformal_tv, **_CREDAL_SET_SPEC_KWARGS),
    "conformal_kl": conformal.ConformalSpec(transform=_conformal_kl, **_CREDAL_SET_SPEC_KWARGS),
    "conformal_wasserstein": conformal.ConformalSpec(transform=_conformal_wasserstein, **_CREDAL_SET_SPEC_KWARGS),
    "conformal_inner_product": conformal.ConformalSpec(transform=_conformal_inner_product, **_CREDAL_SET_SPEC_KWARGS),
}

# Register credal-set entries on the shared registry so downstream loaders find them.
for _name, _spec in _CREDAL_SET_REGISTRY.items():
    conformal.CONFORMAL_METHODS.setdefault(_name, _spec)


def _validate_credal_set_conformal_config(cfg: DictConfig | dict) -> None:
    """Raise if the configured conformal method is not a credal-set method."""
    name = conformal.get_conformal_name(cfg)
    if name not in _CREDAL_SET_REGISTRY:
        supported = ", ".join(sorted(_CREDAL_SET_REGISTRY))
        msg = (
            f"conformalize_credal_set.py only supports credal-set conformal methods "
            f"({supported}); got conformal.name={name!r}."
        )
        raise ValueError(msg)


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


@hydra.main(version_base=None, config_path="configs/", config_name="conformalize_credal_set")
def main(cfg: DictConfig) -> None:
    """Run conformal credal-set calibration on first-order data for a trained benchmark model."""
    print("=== Conformal credal-set configuration ===")
    print(OmegaConf.to_yaml(cfg))

    calibration.validate_calibration_config(cfg)
    _validate_credal_set_conformal_config(cfg)
    cal_split = float(cfg.get("cal_split", 0.0))
    if cal_split <= 0:
        msg = "Conformal credal-set calibration requires `cal_split > 0`."
        raise ValueError(msg)

    seed = cfg.get("seed", None)
    utils.set_seed(seed)

    run_id = wandb.util.generate_id()
    loss_suffix = utils.supervised_loss_name_suffix(cfg)
    calibration_suffix = utils.calibration_name_suffix(cfg)
    conformal_suffix = utils.conformal_name_suffix(cfg)
    run_name = (
        f"{cfg.method.name}_{cfg.base_model}_{cfg.dataset}"
        f"{loss_suffix}{calibration_suffix}{conformal_suffix}_credal_set_{run_id}"
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

    load_from = cfg.get("load_from")
    if load_from == "base":
        source_artifact = f"base_{cfg.base_model}_{cfg.dataset}_{cfg.seed}"
    else:
        source_artifact = utils.resolve_artifact_name(cfg, include_conformal=False)
    model, _, source_run_id = utils.load_model_from_wandb(
        source_artifact,
        cfg.wandb.entity,
        cfg.wandb.project,
        device,
    )
    print(f"Loaded model {source_artifact} from wandb run: {source_run_id}")

    cal_loader, _test_loader = data.get_data_first_order(
        cfg.first_order_dataset,
        seed=cfg.seed,
        cal_split=cal_split,
        batch_size=cfg.batch_size,
    )
    if cal_loader is None:
        msg = f"Empty calibration loader for first_order_dataset={cfg.first_order_dataset!r}, cal_split={cal_split}."
        raise RuntimeError(msg)

    logits, targets = utils.collect_outputs_targets_raw(model, cal_loader, device, cfg.get("amp", False))
    logit_conformalizer = conformal.fit_logit_conformalizer(cfg, logits, targets)

    metrics = conformal.extract_conformal_metrics(cfg, logit_conformalizer)
    log_metrics = {f"conformal/{key}": value for key, value in metrics.items()}
    run.summary.update(log_metrics)
    run.log(data=log_metrics)

    _save_conformal_artifact(logit_conformalizer, cfg, log_metrics, source_artifact, source_run_id)
    run.finish()


if __name__ == "__main__":
    main()
