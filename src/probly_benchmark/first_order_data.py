"""Collect raw representer outputs for first-order data comparison."""

from __future__ import annotations

from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from probly.metrics import convex_hull_coverage, coverage, efficiency
from probly.representation.distribution.torch_categorical import TorchProbabilityCategoricalDistribution
from probly.representer import ConvexCredalSetRepresenter, ProbabilityIntervalsRepresenter, representer
from probly_benchmark import (
    calibration,
    conformalize_credal_set as _conformalize_credal_set,  # noqa: F401  # registers credal-set conformal methods
    data,
    utils,
)
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

    amp = cfg.get("amp", False)

    # Natural representation: used for efficiency and as the primary output for
    # methods whose representer already yields a ProbabilityIntervalsCredalSet.
    rep = representer(model, **rep_kwargs)
    outputs, targets = collect_outputs_targets(rep, data_loader, device, amp_enabled=amp)
    eff = efficiency(outputs)

    # Interval coverage: always computed from a ProbabilityIntervalsCredalSet so
    # coverage() dispatches to the interval-containment check.  For iterable
    # predictors (credal_bnn, credal_ensembling, credal_wrapper, ...) we build a
    # dedicated ProbabilityIntervalsRepresenter; for non-iterable predictors
    # (credal_net, efficient_credal_prediction) the natural representation is
    # already a ProbabilityIntervalsCredalSet, so we fall back to that.
    try:
        rep_interval = ProbabilityIntervalsRepresenter(model)
        outputs_interval, _ = collect_outputs_targets(rep_interval, data_loader, device, amp_enabled=amp)
        cov = coverage(outputs_interval, targets)
    except (TypeError, NotImplementedError, ValueError):
        cov = coverage(outputs, targets)

    # Convex-hull coverage requires a vertex-based convex credal set.
    # For methods whose natural representation is already a ConvexCredalSet
    # (credal_bnn, credal_ensembling) we reuse the existing outputs directly.
    # For methods that produce a ProbabilityIntervalsCredalSet but whose
    # predictor is iterable (credal_wrapper, credal_relative_likelihood) we
    # build a ConvexCredalSetRepresenter from the same model and run a second
    # forward pass.
    # Predictors that are not iterable ensembles (credal_net,
    # efficient_credal_prediction) cannot produce a ConvexCredalSet; we skip.
    targets_dist = TorchProbabilityCategoricalDistribution(tensor=targets)
    chull_cov: float | None = None
    try:
        chull_cov = float(convex_hull_coverage(outputs, targets_dist, epsilon=0.005))
    except NotImplementedError:
        try:
            rep_convex = ConvexCredalSetRepresenter(model)
            outputs_convex, _ = collect_outputs_targets(rep_convex, data_loader, device, amp_enabled=amp)
            chull_cov = float(convex_hull_coverage(outputs_convex, targets_dist, epsilon=0.005))
        except (TypeError, NotImplementedError, ValueError):
            print(f"convex_hull_coverage not available for {type(model).__name__}, skipping.")

    if cfg.wandb.enabled:
        run = init_wandb_for_evaluation(cfg, run_id)
        run.summary["coverage"] = cov
        run.summary["efficiency"] = eff
        if chull_cov is not None:
            run.summary["convex_hull_coverage"] = chull_cov
        run.finish()


if __name__ == "__main__":
    main()
