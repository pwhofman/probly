"""Perform out-of-distribution detection experiments."""

from __future__ import annotations

from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from probly.evaluation.ood import out_of_distribution_detection_auroc
from probly.quantification import quantify
from probly.representer import representer
from probly_benchmark import data, utils
from probly_benchmark.utils import load_model_from_wandb, resolve_artifact_name


@hydra.main(version_base=None, config_path="configs/", config_name="ood_detection")
def main(cfg: DictConfig) -> None:
    """Run out-of-distribution detection evaluation."""
    print("=== OOD detection configuration ===")
    print(OmegaConf.to_yaml(cfg))

    device = utils.get_device(cfg.get("device", None))
    print(f"Running on device: {device}")

    utils.set_seed(cfg.seed)

    artifact_name = resolve_artifact_name(cfg)
    model, _, run_id = load_model_from_wandb(
        artifact_name,
        cfg.wandb.entity,
        cfg.wandb.project,
        device,
    )

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

    id_outputs, _ = utils.collect_outputs_targets(rep, id_loader, device, cfg.get("amp", False))
    ood_outputs, _ = utils.collect_outputs_targets(rep, ood_loader, device, cfg.get("amp", False))

    id_uncertainties = quantify(id_outputs).epistemic.detach().cpu().numpy()  # ty:ignore[unresolved-attribute]
    ood_uncertainties = quantify(ood_outputs).epistemic.detach().cpu().numpy()  # ty:ignore[unresolved-attribute]

    auroc = out_of_distribution_detection_auroc(id_uncertainties, ood_uncertainties)
    print(f"OOD detection AUROC: {auroc:.4f}")

    if cfg.wandb.enabled:
        run = wandb.init(
            id=run_id,
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            resume="must",
        )
        run.summary["ood/auroc"] = auroc
        run.finish()


if __name__ == "__main__":
    main()
