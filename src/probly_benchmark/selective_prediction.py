"""Perform selective prediction experiments."""

from __future__ import annotations

from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from probly.evaluation.tasks import selective_prediction
from probly.quantification import quantify
from probly.representation._helpers import compute_mean_probs
from probly.representer import representer
from probly_benchmark import data, utils
from probly_benchmark.utils import load_model_from_wandb, resolve_artifact_name


@hydra.main(version_base=None, config_path="configs/", config_name="selective_prediction")
def main(cfg: DictConfig) -> None:
    """Run selective prediction evaluation."""
    print("=== Selective prediction configuration ===")
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
    print(f"Loaded model {artifact_name} from wandb run: {run_id}")

    test_loader = data.get_data_selective_prediction(
        cfg.dataset,
        cfg.seed,
        val_split=cfg.val_split,
        cal_split=cfg.get("cal_split", 0.0),
        batch_size=cfg.batch_size,
    )

    rep_kwargs: dict[str, Any] = (
        OmegaConf.to_container(cfg.method.selective_prediction, resolve=True)
        if cfg.method.get("selective_prediction")
        else {}
    )  # ty: ignore[invalid-assignment]
    rep = representer(model, **rep_kwargs)

    outputs, targets = utils.collect_outputs_targets(
        rep,
        test_loader,
        device,
        cfg.get("amp", False),
    )
    decomposition = quantify(outputs)
    uncertainties = decomposition.total.detach().cpu().numpy()  # ty: ignore[unresolved-attribute]

    mean_probs = compute_mean_probs(outputs).cpu().numpy()

    labels = targets.numpy()
    loss = (mean_probs.argmax(axis=1) != labels).astype(float)
    auroc, bin_losses = selective_prediction(uncertainties, loss, n_bins=cfg.n_bins)
    print(f"Selective prediction AUROC: {auroc:.4f}")

    if cfg.wandb.enabled:
        run = wandb.init(
            id=run_id,
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            resume="must",
        )
        run.summary["sp/auroc"] = auroc
        run.summary["sp/bin_losses"] = bin_losses.tolist()
        run.finish()


if __name__ == "__main__":
    main()
