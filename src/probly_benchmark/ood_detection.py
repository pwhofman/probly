"""Perform out-of-distribution detection experiments."""

from __future__ import annotations

from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import wandb

from probly.evaluation.ood import out_of_distribution_detection_auroc
from probly.quantification import quantify
from probly.representation.distribution.torch_categorical import (
    TorchCategoricalDistribution,
    TorchCategoricalDistributionSample,
)
from probly.representer import representer
from probly_benchmark import data, utils
from probly_benchmark.utils import load_model_from_wandb, resolve_artifact_name


def compute_epistemic_uncertainty(outputs: list[Any]) -> torch.Tensor:
    """Compute epistemic uncertainty (mutual information) from sampler outputs."""
    all_uncertainties = []
    for out in outputs:
        probs = torch.softmax(out.samples, dim=-1)  # (num_samples, batch, n_classes)
        dist = TorchCategoricalDistribution(probs)
        dist_sample = TorchCategoricalDistributionSample(tensor=dist, sample_dim=0)
        decomposition = quantify(dist_sample)
        all_uncertainties.append(decomposition.epistemic.cpu())  # ty: ignore[unresolved-attribute]
    return torch.cat(all_uncertainties)


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

    # Replace
    _, _, id_loader = data.get_data_train(
        cfg.dataset,
        use_validation=False,
        batch_size=cfg.batch_size,
    )
    _, _, ood_loader = data.get_data_train(
        cfg.dataset,
        use_validation=False,
        batch_size=cfg.batch_size,
    )

    rep_kwargs: dict[str, Any] = (
        OmegaConf.to_container(cfg.method.ood_detection, resolve=True) if cfg.method.get("ood_detection") else {}
    )  # ty: ignore[invalid-assignment]
    rep = representer(model, **rep_kwargs)

    id_outputs, _ = utils.collect_outputs_targets(rep, id_loader, device, cfg.get("amp", False))
    ood_outputs, _ = utils.collect_outputs_targets(rep, ood_loader, device, cfg.get("amp", False))

    id_uncertainties = compute_epistemic_uncertainty(id_outputs).numpy()
    ood_uncertainties = compute_epistemic_uncertainty(ood_outputs).numpy()

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
