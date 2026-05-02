"""Perform selective prediction experiments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import hydra
from omegaconf import DictConfig, OmegaConf

from probly.evaluation.tasks import selective_prediction
from probly.quantification import quantify
from probly.representer import representer
from probly_benchmark import calibration, data, utils
from probly_benchmark.utils import (
    collect_outputs_decisions_targets,
    init_wandb_for_evaluation,
    load_model_for_evaluation,
)

if TYPE_CHECKING:
    import numpy as np

_SUPPORTED_LOSSES = ("zero_one",)
_SUPPORTED_DECOMPOSITIONS = ("aleatoric", "epistemic", "total")


def _compute_loss(mean_probs: np.ndarray, labels: np.ndarray, loss: str) -> np.ndarray:
    """Compute the per-sample loss for selective prediction.

    Args:
        mean_probs: Mean predicted probabilities of shape (n_instances, n_classes).
        labels: True labels of shape (n_instances,).
        loss: Loss function identifier. One of ``_SUPPORTED_LOSSES``.

    Returns:
        Per-sample loss values of shape (n_instances,).

    Raises:
        ValueError: If ``loss`` is not supported.
    """
    if loss == "zero_one":
        return (mean_probs.argmax(axis=-1) != labels).astype(float)
    msg = f"Unsupported loss: {loss!r}. Choose from {_SUPPORTED_LOSSES}."
    raise ValueError(msg)


@hydra.main(version_base=None, config_path="configs/", config_name="selective_prediction")
def main(cfg: DictConfig) -> None:
    """Run selective prediction evaluation."""
    print("=== Selective prediction configuration ===")
    print(OmegaConf.to_yaml(cfg))

    device = utils.get_device(cfg.get("device", None))
    print(f"Running on device: {device}")

    utils.set_seed(cfg.seed)
    calibration.validate_calibration_config(cfg)

    model, _, run_id = load_model_for_evaluation(cfg, device)
    print(f"Loaded model for {cfg.method.name} from wandb run: {run_id}")

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

    if cfg.decomposition not in _SUPPORTED_DECOMPOSITIONS:
        msg = f"Unsupported decomposition: {cfg.decomposition!r}. Choose from {_SUPPORTED_DECOMPOSITIONS}."
        raise ValueError(msg)

    outputs, mean_probs, targets = collect_outputs_decisions_targets(
        model,
        rep,
        test_loader,
        device,
        rep_kwargs=rep_kwargs or None,
        amp_enabled=cfg.get("amp", False),
    )

    decomposition = quantify(outputs)
    uncertainties = decomposition[cfg.decomposition].detach().cpu().numpy()  # ty:ignore[not-subscriptable]

    labels = targets.numpy()
    loss = _compute_loss(mean_probs, labels, cfg.loss)
    auroc, bin_losses = selective_prediction(uncertainties, loss, n_bins=cfg.n_bins)
    print(f"Selective prediction AUROC: {auroc:.4f}")

    if cfg.wandb.enabled:
        run = init_wandb_for_evaluation(cfg, run_id)
        run.summary["sp/auroc"] = auroc
        run.summary["sp/bin_losses"] = bin_losses.tolist()
        run.finish()


if __name__ == "__main__":
    main()
