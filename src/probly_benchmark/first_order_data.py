"""Collect raw representer outputs for first-order data comparison."""

from __future__ import annotations

from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from probly.metrics import coverage, efficiency
from probly.representer import representer
from probly_benchmark import calibration, data, utils
from probly_benchmark.utils import (
    collect_outputs_targets,
    init_wandb_for_evaluation,
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

    model, train_cfg, run_id = load_model_for_evaluation(cfg, device)
    print(f"Loaded model for {cfg.method.name} from wandb run: {run_id}")

    # Replay the same cal/test split that conformalize_credal_set uses so every
    # model is evaluated on the same held-out test portion. Most artifacts (base,
    # credal_wrapper, ...) don't carry cal_split, so we default to 0.2 — the
    # default in conformalize_credal_set.yaml.
    cal_split = float(train_cfg.get("cal_split", 0.2) or 0.2)
    data_seed = int(train_cfg.get("seed", cfg.seed))
    _, data_loader = data.get_data_first_order(
        cfg.first_order_dataset,
        seed=data_seed,
        cal_split=cal_split,
        batch_size=cfg.batch_size,
    )

    rep_kwargs: dict[str, Any] = (
        OmegaConf.to_container(cfg.method.first_order_data, resolve=True) if cfg.method.get("first_order_data") else {}
    )  # ty: ignore[invalid-assignment]
    rep = representer(model, **rep_kwargs)

    outputs, targets = collect_outputs_targets(rep, data_loader, device, amp_enabled=cfg.get("amp", False))

    ### coverage
    cov = coverage(outputs, targets)

    ### efficiency
    eff = efficiency(outputs)

    if cfg.wandb.enabled:
        run = init_wandb_for_evaluation(cfg, run_id)
        run.summary["coverage"] = cov
        run.summary["efficiency"] = eff
        run.finish()


if __name__ == "__main__":
    main()
