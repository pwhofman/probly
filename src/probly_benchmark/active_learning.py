"""Hydra entry point for active learning experiments."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import wandb
import wandb.util

from probly.evaluation.active_learning import (
    ActiveLearningPool,
    BADGEQuery,
    MarginSampling,
    QueryStrategy,
    RandomQuery,
    UncertaintyQuery,
    active_learning_steps,
    compute_accuracy,
    compute_ece,
    compute_nauc,
)
from probly_benchmark.al_estimator import BenchmarkALEstimator
from probly_benchmark.data import get_data_al
from probly_benchmark.utils import set_seed

logger = logging.getLogger(__name__)


def _build_query_strategy(cfg: DictConfig) -> QueryStrategy:
    """Construct a query strategy from config."""
    name = cfg.al_strategy.name.lower()
    match name:
        case "uncertainty":
            return UncertaintyQuery()
        case "margin":
            return MarginSampling()
        case "badge":
            return BADGEQuery(seed=cfg.seed)
        case "random":
            return RandomQuery(seed=cfg.seed)
        case _:
            msg = f"Unknown AL strategy: {name!r}"
            raise ValueError(msg)


@hydra.main(version_base=None, config_path="configs/", config_name="active_learning")
def main(cfg: DictConfig) -> float:
    """Run a single active learning experiment.

    Returns:
        Final NAUC (for Hydra optimisation sweeps).
    """
    set_seed(cfg.seed)

    # --- Data ---
    dataset_kwargs: dict[str, Any] = {}
    if cfg.dataset.name == "openml":
        dataset_kwargs["openml_id"] = cfg.dataset.openml_id

    x_train, y_train, x_test, y_test, num_classes, in_features = get_data_al(
        cfg.dataset.name, seed=cfg.seed, **dataset_kwargs
    )
    if cfg.dataset.num_classes is None:
        cfg.dataset.num_classes = num_classes

    logger.info(
        "Dataset: %s | train=%d test=%d classes=%d features=%s",
        cfg.dataset.name,
        len(x_train),
        len(x_test),
        num_classes,
        in_features,
    )

    # --- Pool ---
    pool = ActiveLearningPool.from_dataset(
        x_train,
        y_train,
        x_test,
        y_test,
        initial_size=cfg.initial_size,
        seed=cfg.seed,
    )

    # --- Estimator ---
    device = torch.device(cfg.device)
    method_params: dict[str, Any] = (
        OmegaConf.to_container(cfg.method.params, resolve=True) if cfg.method.get("params") else {}
    )  # ty: ignore[invalid-assignment]
    train_kwargs: dict[str, Any] = (
        OmegaConf.to_container(cfg.method.train, resolve=True) if cfg.method.get("train") else {}
    )  # ty: ignore[invalid-assignment]

    estimator = BenchmarkALEstimator(
        method_name=cfg.method.name,
        method_params=method_params,
        train_kwargs=train_kwargs,
        cfg=cfg,
        base_model_name=cfg.dataset.base_model,
        model_type=cfg.model_type,
        num_classes=num_classes,
        device=device,
        in_features=in_features if cfg.dataset.type == "tabular" else None,
        quantifier=cfg.get("quantifier", None),
        num_samples=cfg.get("num_samples", 10),
    )

    # --- Strategy ---
    strategy = _build_query_strategy(cfg)

    if cfg.al_strategy.name == "uncertainty" and not hasattr(estimator, "uncertainty_scores"):
        msg = (
            f"Strategy 'uncertainty' requires uncertainty_scores(), "
            f"but method '{cfg.method.name}' does not provide one."
        )
        raise ValueError(msg)

    # --- WandB ---
    run_id = wandb.util.generate_id()
    run = wandb.init(
        id=run_id,
        name=f"al_{cfg.method.name}_{cfg.dataset.name}_{run_id}",
        entity=cfg.wandb.get("entity", None),
        project=cfg.wandb.project,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),  # ty: ignore
        mode="online" if cfg.wandb.enabled else "disabled",
        save_code=True,
    )
    run.config.update({"seed": cfg.seed})

    # --- AL Loop ---
    accuracy_scores: list[float] = []
    iterations: list[dict[str, float | int]] = []

    for state in active_learning_steps(
        pool,
        estimator,  # ty: ignore[invalid-argument-type]
        strategy,
        query_size=cfg.query_size,
        n_iterations=cfg.n_iterations,
    ):
        preds = state.estimator.predict(pool.x_test)
        probs = state.estimator.predict_proba(pool.x_test)
        acc = compute_accuracy(preds, pool.y_test)
        ece = compute_ece(probs, pool.y_test)
        accuracy_scores.append(acc)
        iterations.append({"iteration": state.iteration, "labeled_size": pool.n_labeled, "accuracy": acc, "ece": ece})

        run.log(
            {
                "iteration": state.iteration,
                "labeled_size": pool.n_labeled,
                "test_accuracy": acc,
                "test_ece": ece,
            }
        )
        logger.info(
            "Iter %d | labeled=%d | acc=%.4f | ece=%.4f",
            state.iteration,
            pool.n_labeled,
            acc,
            ece,
        )

    # --- Final metrics ---
    nauc = compute_nauc(accuracy_scores)
    final_acc = accuracy_scores[-1] if accuracy_scores else float("nan")
    run.summary["nauc"] = nauc
    run.summary["final_accuracy"] = final_acc
    logger.info("Done. NAUC=%.4f | Final acc=%.4f", nauc, final_acc)

    # --- Save results to JSON in Hydra's output directory ---
    from hydra.core.hydra_config import HydraConfig  # noqa: PLC0415

    out_dir = Path(HydraConfig.get().runtime.output_dir)
    results = {
        "method": cfg.method.name,
        "dataset": cfg.dataset.name,
        "strategy": cfg.al_strategy.name,
        "seed": cfg.seed,
        "nauc": nauc,
        "final_accuracy": final_acc,
        "iterations": iterations,
    }
    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2))
    logger.info("Results saved to %s", results_path)

    run.finish()
    return nauc


if __name__ == "__main__":
    main()
