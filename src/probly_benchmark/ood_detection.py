"""Perform out-of-distribution detection experiments."""

from __future__ import annotations

from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from probly.evaluation.ood import evaluate_ood
from probly.quantification import quantify
from probly.representer import representer
from probly_benchmark import calibration, data, utils
from probly_benchmark.utils import init_wandb_for_evaluation, load_model_for_evaluation

_SUPPORTED_DECOMPOSITIONS = ("aleatoric", "epistemic", "total")


@hydra.main(version_base=None, config_path="configs/", config_name="ood_detection")
def main(cfg: DictConfig) -> None:
    """Run out-of-distribution detection evaluation."""
    print("=== OOD detection configuration ===")
    print(OmegaConf.to_yaml(cfg))

    device = utils.get_device(cfg.get("device", None))
    print(f"Running on device: {device}")

    utils.set_seed(cfg.seed)
    calibration.validate_calibration_config(cfg)
    print("Loading model...")
    model, _, run_id = load_model_for_evaluation(cfg, device)
    print("Loading data...")
    id_loader, ood_loader = data.get_data_ood(
        cfg.dataset,
        cfg.ood_dataset,
        cfg.seed,
        val_split=cfg.val_split,
        cal_split=cfg.get("cal_split", 0.0),
        batch_size=cfg.batch_size,
    )

    rep_kwargs: dict[str, Any] = (
        OmegaConf.to_container(cfg.method.ood_detection, resolve=True) if cfg.method.get("ood_detection") else {}
    )  # ty: ignore[invalid-assignment]
    rep = representer(model, **rep_kwargs)
    print(f"Getting outputs...")
    id_outputs, _ = utils.collect_outputs_targets(rep, id_loader, device, cfg.get("amp", False))
    ood_outputs, _ = utils.collect_outputs_targets(rep, ood_loader, device, cfg.get("amp", False))

    if cfg.decomposition not in _SUPPORTED_DECOMPOSITIONS:
        msg = f"Unsupported decomposition: {cfg.decomposition!r}. Choose from {_SUPPORTED_DECOMPOSITIONS}."
        raise ValueError(msg)

    id_uncertainties = quantify(id_outputs)[cfg.decomposition].detach().cpu().numpy()  # ty:ignore[not-subscriptable]
    ood_uncertainties = quantify(ood_outputs)[cfg.decomposition].detach().cpu().numpy()  # ty:ignore[not-subscriptable]

    ood_metrics = evaluate_ood(id_uncertainties, ood_uncertainties, metrics=cfg.get("metrics", "all"))
    auroc = ood_metrics["auroc"]
    print(f"OOD detection AUROC: {auroc:.4f}")

    if cfg.wandb.enabled:
        run = init_wandb_for_evaluation(cfg, run_id)
        prefix = f"ood/{cfg.ood_dataset}/{cfg.measure}/{cfg.decomposition}"
        for metric_name, value in ood_metrics.items():
            run.summary[f"{prefix}/{metric_name}"] = value
        run.summary[f"{prefix}/ood_scores"] = ood_uncertainties.tolist()
        run.summary[f"ood/{cfg.measure}/{cfg.decomposition}/id_scores"] = id_uncertainties.tolist()
        run.finish()


if __name__ == "__main__":
    main()
