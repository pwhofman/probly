"""Hydra entry point for active learning experiments."""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import wandb.util

from probly.evaluation.active_learning import (
    ActiveLearningPool,
    active_learning_steps,
    compute_accuracy,
    compute_ece,
    compute_nauc,
)
from probly_benchmark.al_builders import build_al_estimator, build_query_strategy
from probly_benchmark.data import get_data_al
from probly_benchmark.utils import set_seed

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs/", config_name="active_learning")
def main(cfg: DictConfig) -> float:
    """Run a single active learning experiment.

    Args:
        cfg: Hydra config.

    Returns:
        Final NAUC (for Hydra optimisation sweeps).
    """
    set_seed(cfg.seed)

    # --- Data ---
    dataset_kwargs = {}
    if cfg.dataset.name == "openml":
        dataset_kwargs["openml_id"] = cfg.dataset.openml_id

    x_train, y_train, x_test, y_test, num_classes, in_features = get_data_al(
        cfg.dataset.name, seed=cfg.seed, **dataset_kwargs
    )
    # Resolve num_classes if not set in config
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

    # --- Model + Strategy ---
    estimator = build_al_estimator(cfg, in_features=in_features if cfg.dataset.type == "tabular" else None)
    strategy = build_query_strategy(cfg)

    # --- WandB ---
    run_id = wandb.util.generate_id()
    run = wandb.init(
        id=run_id,
        name=f"al_{cfg.al_method.name}_{cfg.dataset.name}_{run_id}",
        entity=cfg.wandb.get("entity", None),
        project=cfg.wandb.project,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),  # ty: ignore
        mode="online" if cfg.wandb.enabled else "disabled",
        save_code=True,
    )
    run.config.update({"seed": cfg.seed})

    # --- AL Loop ---
    accuracy_scores: list[float] = []

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

    run.finish()
    return nauc


if __name__ == "__main__":
    main()
