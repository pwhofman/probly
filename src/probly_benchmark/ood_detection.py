"""Perform out-of-distribution detection experiments."""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import wandb

from probly.evaluation.ood import evaluate_ood
from probly.representer import representer
from probly_benchmark import calibration, data, utils
from probly_benchmark.uncertainty import SUPPORTED_DECOMPOSITIONS
from probly_benchmark.utils import init_wandb_for_evaluation, load_model_for_evaluation


def _log_array_artifact(
    run: wandb.sdk.wandb_run.Run,
    *,
    name: str,
    artifact_type: str,
    metadata: dict[str, Any],
    filename: str,
    array: np.ndarray,
) -> None:
    """Save ``array`` to a ``.npy`` file and log it as a wandb artifact."""
    art = wandb.Artifact(name=name, type=artifact_type, metadata=metadata)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / filename
        np.save(path, array)
        art.add_file(str(path))
    run.log_artifact(art)


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

    if cfg.decomposition not in SUPPORTED_DECOMPOSITIONS:
        msg = f"Unsupported decomposition: {cfg.decomposition!r}. Choose from {SUPPORTED_DECOMPOSITIONS}."
        raise ValueError(msg)

    # ID scores are identical across OOD datasets for a given (method, seed).
    # Try to fetch a previously logged artifact from wandb and skip the
    # ID-set inference pass entirely on cache hit.
    id_art_name = f"id_scores-{cfg.method.name}-{cfg.dataset}-{cfg.measure}-{cfg.decomposition}-seed{cfg.seed}"
    id_qualname = f"{cfg.wandb.entity}/{cfg.wandb.project}/{id_art_name}:latest"
    id_uncertainties: np.ndarray | None = None
    id_loaded_from_cache = False
    if cfg.wandb.enabled:
        try:
            api = wandb.Api(timeout=60)
            art = api.artifact(id_qualname)
            with tempfile.TemporaryDirectory() as td:
                art.download(root=td)
                id_uncertainties = np.load(Path(td) / "id_scores.npy")
            id_loaded_from_cache = True
            print(f"Loaded cached id_scores from {id_qualname} (shape={id_uncertainties.shape}).")
        except wandb.errors.CommError:
            pass

    print("Getting per-batch uncertainties...")
    if id_uncertainties is None:
        # Per-batch quantify preserves method-specific decomposition markers (PostNet, NatPN, EDL).
        id_uncertainties = (
            utils.collect_uncertainties(rep, id_loader, device, cfg.decomposition, cfg.get("amp", False))
            .detach()
            .cpu()
            .numpy()
        )
    ood_uncertainties = (
        utils.collect_uncertainties(rep, ood_loader, device, cfg.decomposition, cfg.get("amp", False))
        .detach()
        .cpu()
        .numpy()
    )

    ood_metrics = evaluate_ood(id_uncertainties, ood_uncertainties, metrics=cfg.get("metrics", "all"))
    auroc = ood_metrics["auroc"]
    print(f"OOD detection AUROC: {auroc:.4f}")

    if cfg.wandb.enabled:
        run = init_wandb_for_evaluation(cfg, run_id)
        prefix = f"ood/{cfg.ood_dataset}/{cfg.measure}/{cfg.decomposition}"
        for metric_name, value in ood_metrics.items():
            run.summary[f"{prefix}/{metric_name}"] = value

        common_meta = {
            "method": cfg.method.name,
            "dataset": cfg.dataset,
            "measure": cfg.measure,
            "decomposition": cfg.decomposition,
            "seed": cfg.seed,
        }

        # Log id_scores once per (method, seed); skip if we loaded from cache.
        if not id_loaded_from_cache:
            _log_array_artifact(
                run,
                name=id_art_name,
                artifact_type="id_scores",
                metadata=common_meta,
                filename="id_scores.npy",
                array=id_uncertainties,
            )

        _log_array_artifact(
            run,
            name=(
                f"ood_scores-{cfg.method.name}-{cfg.dataset}-{cfg.ood_dataset}"
                f"-{cfg.measure}-{cfg.decomposition}-seed{cfg.seed}"
            ),
            artifact_type="ood_scores",
            metadata={**common_meta, "ood_dataset": cfg.ood_dataset},
            filename="ood_scores.npy",
            array=ood_uncertainties,
        )

        run.finish()


if __name__ == "__main__":
    main()
