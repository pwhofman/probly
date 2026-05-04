"""Collect raw representer outputs for first-order data comparison."""

from __future__ import annotations

from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from probly.representer import representer
from probly_benchmark import calibration, data, utils
from probly_benchmark.utils import (
    collect_outputs_targets,
    collect_outputs_targets_raw,
    load_model_for_evaluation,
)


@hydra.main(version_base=None, config_path="configs/", config_name="first_order_data")
def main(cfg: DictConfig) -> None:
    """Collect representer outputs on the test set for first-order data comparison."""
    print("=== First-order data configuration ===")
    print(OmegaConf.to_yaml(cfg))

    device = utils.get_device(cfg.get("device", None))
    print(f"Running on device: {device}")

    utils.set_seed(cfg.seed)
    calibration.validate_calibration_config(cfg)

    model, _, run_id = load_model_for_evaluation(cfg, device)
    print(f"Loaded model for {cfg.method.name} from wandb run: {run_id}")

    test_loader = data.get_data_first_order_comparison(
        cfg.dataset,
        cfg.seed,
        val_split=cfg.val_split,
        cal_split=cfg.get("cal_split", 0.0),
        batch_size=cfg.batch_size,
    )

    rep_kwargs: dict[str, Any] = (
        OmegaConf.to_container(cfg.method.first_order_data, resolve=True) if cfg.method.get("first_order_data") else {}
    )  # ty: ignore[invalid-assignment]
    rep = representer(model, **rep_kwargs)

    outputs, targets = collect_outputs_targets(rep, test_loader, device, amp_enabled=cfg.get("amp", False))
    print(outputs)
    print(type(outputs))
    print(f"Outputs shape: {outputs.shape}")  # ty: ignore[unresolved-attribute]
    print(f"Targets shape: {targets.shape}")
    print(f"Num samples: {len(targets)}")

    outputs, targets = collect_outputs_targets_raw(model, test_loader, device, amp_enabled=cfg.get("amp", False))
    print(outputs)
    print(type(outputs))
    print(f"Outputs shape: {outputs.shape}")  # ty: ignore[unresolved-attribute]
    print(f"Targets shape: {targets.shape}")
    print(f"Num samples: {len(targets)}")


if __name__ == "__main__":
    main()
