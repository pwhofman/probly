"""Hydra entry point for active learning experiments."""

from __future__ import annotations

import fcntl
import json
import logging
from pathlib import Path
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import wandb
import wandb.util

from probly.evaluation.active_learning import (
    BADGEQuery,
    EntropySampling,
    LeastConfidentSampling,
    MarginSampling,
    RandomQuery,
    UncertaintyQuery,
    active_learning_steps,
    compute_accuracy,
    compute_nauc,
    from_dataset,
)
from probly.quantification.notion import EpistemicUncertainty, TotalUncertainty
from probly_benchmark.al_estimators import BaselineEstimator, ConformalEstimator, UncertaintyEstimator
from probly_benchmark.data import get_data_al
from probly_benchmark.metadata import AL_DATASETS
from probly_benchmark.utils import set_seed

logger = logging.getLogger(__name__)

_UNCERTAINTY_NOTIONS = {"EU": EpistemicUncertainty, "TU": TotalUncertainty}


def _build_estimator(
    cfg: DictConfig,
    *,
    base_model_name: str,
    num_classes: int,
    in_features: int | None,
    device: torch.device,
) -> BaselineEstimator | UncertaintyEstimator | ConformalEstimator:
    """Dispatch estimator based on method and strategy combination.

    ``plain`` and ``ensemble`` with baseline strategies (margin/badge/random)
    use :class:`BaselineEstimator`.  Methods starting with ``conformal_`` use
    :class:`ConformalEstimator`.  Everything else uses
    :class:`UncertaintyEstimator`.
    """
    method = cfg.method.name
    strategy = cfg.al_strategy.name
    raw_params = cfg.method.get("params")
    method_params = cast("dict[str, Any]", OmegaConf.to_container(raw_params, resolve=True)) if raw_params else {}

    # --- Conformal methods ---
    if method.startswith("conformal_"):
        al_section = cfg.method.get("active_learning", {})
        score_name = al_section.get("score", "lac")
        alpha = float(al_section.get("alpha", 0.1))
        cal_split = float(al_section.get("cal_split", 0.25))
        return ConformalEstimator(
            cfg=cfg,
            base_model_name=base_model_name,
            method_name=score_name,
            method_params=method_params,
            num_classes=num_classes,
            device=device,
            in_features=in_features,
            alpha=alpha,
            cal_split=cal_split,
        )

    # --- Baseline methods ---
    use_baseline = method == "plain" or (method == "ensemble" and strategy != "uncertainty")
    if method == "plain" and strategy == "uncertainty":
        msg = "Method 'plain' cannot use 'uncertainty' strategy — use margin, badge, or random."
        raise ValueError(msg)

    if use_baseline:
        return BaselineEstimator(
            cfg=cfg,
            base_model_name=base_model_name,
            method_name=method,
            method_params=method_params,
            num_classes=num_classes,
            device=device,
            in_features=in_features,
        )

    # --- Uncertainty methods ---
    raw_train = cfg.method.get("train")
    train_kwargs = cast("dict[str, Any]", OmegaConf.to_container(raw_train, resolve=True)) if raw_train else {}
    raw_al = cfg.method.get("active_learning")
    rep_kwargs = cast("dict[str, Any]", OmegaConf.to_container(raw_al, resolve=True)) if raw_al else {}

    return UncertaintyEstimator(
        cfg=cfg,
        base_model_name=base_model_name,
        method_name=method,
        method_params=method_params,
        train_kwargs=train_kwargs,
        num_classes=num_classes,
        device=device,
        in_features=in_features,
        rep_kwargs=rep_kwargs,
        uncertainty_notion=_UNCERTAINTY_NOTIONS[cfg.uncertainty_decomposition],
    )


def _build_strategy(
    cfg: DictConfig,
) -> EntropySampling | LeastConfidentSampling | MarginSampling | BADGEQuery | UncertaintyQuery | RandomQuery:
    """Dispatch query strategy based on config."""
    name = cfg.al_strategy.name.lower()
    match name:
        case "entropy":
            return EntropySampling()
        case "least_confident":
            return LeastConfidentSampling()
        case "margin":
            return MarginSampling()
        case "badge":
            return BADGEQuery(seed=cfg.seed)
        case "uncertainty":
            return UncertaintyQuery()
        case "random":
            return RandomQuery(seed=cfg.seed)
        case _:
            msg = f"Unknown AL strategy: {name!r}"
            raise ValueError(msg)


def _append_result(results_file: Path, result: dict[str, Any]) -> None:
    """Append result to a shared JSON list, deduplicating by key."""
    results_file.parent.mkdir(parents=True, exist_ok=True)
    key = (result["method"], result["dataset"], result["strategy"], result["seed"])

    with results_file.open("a+") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0)
            content = f.read()
            existing: list[dict[str, Any]] = json.loads(content) if content.strip() else []
            existing = [r for r in existing if (r["method"], r["dataset"], r["strategy"], r["seed"]) != key]
            existing.append(result)
            f.seek(0)
            f.truncate()
            json.dump(existing, f, indent=2)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


@hydra.main(version_base=None, config_path="configs/", config_name="active_learning")
def main(cfg: DictConfig) -> float:
    """Run a single active learning experiment."""
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    # --- Data ---
    dataset_kwargs: dict[str, Any] = {}
    if cfg.dataset.name == "openml":
        dataset_kwargs["openml_id"] = cfg.dataset.openml_id
    x_train, y_train, x_test, y_test, num_classes, in_features = get_data_al(
        cfg.dataset.name, seed=cfg.seed, **dataset_kwargs
    )
    logger.info(
        "Dataset: %s | train=%d test=%d classes=%d",
        cfg.dataset.name,
        len(x_train),
        len(x_test),
        num_classes,
    )

    # --- Pool ---
    pool = from_dataset(
        x_train,
        y_train,  # ty: ignore[invalid-argument-type]
        x_test,  # ty: ignore[invalid-argument-type]
        y_test,  # ty: ignore[invalid-argument-type]
        initial_size=cfg.initial_size,
        seed=cfg.seed,
    )

    # --- Resolve dataset metadata ---
    ds_name = cfg.dataset.name
    ds_key = f"{ds_name}_{cfg.dataset.openml_id}" if ds_name == "openml" else ds_name
    ds_meta = AL_DATASETS[ds_key]

    # --- Estimator + Strategy ---
    estimator = _build_estimator(
        cfg,
        base_model_name=ds_meta.base_model,
        num_classes=num_classes,
        in_features=in_features if ds_meta.type == "tabular" else None,
        device=device,
    )
    strategy = _build_strategy(cfg)

    # --- WandB ---
    run_id = wandb.util.generate_id()
    run = wandb.init(
        id=run_id,
        name=f"al_{cfg.method.name}_{cfg.dataset.name}_{run_id}",
        entity=cfg.wandb.get("entity", None),
        project=cfg.wandb.project,
        config=cast(
            "dict[str, Any]",
            OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        ),
        mode="online" if cfg.wandb.enabled else "disabled",
    )

    # --- AL Loop ---
    accuracy_scores: list[float] = []
    iterations: list[dict[str, float | int]] = []

    for state in active_learning_steps(
        pool,
        cast("Any", estimator),
        strategy,
        query_size=cfg.query_size,
        n_iterations=cfg.n_iterations,
    ):
        preds = state.estimator.predict(pool.x_test)
        acc = compute_accuracy(preds, pool.y_test)
        accuracy_scores.append(acc)
        iterations.append(
            {
                "iteration": state.iteration,
                "labeled_size": pool.n_labeled,
                "accuracy": acc,
            }
        )
        run.log(
            {
                "iteration": state.iteration,
                "labeled_size": pool.n_labeled,
                "test_accuracy": acc,
            }
        )
        logger.info(
            "Iter %d | labeled=%d | acc=%.4f",
            state.iteration,
            pool.n_labeled,
            acc,
        )

    # --- Final metrics ---
    nauc = compute_nauc(accuracy_scores)
    final_acc = accuracy_scores[-1] if accuracy_scores else float("nan")
    run.summary["nauc"] = nauc
    run.summary["final_accuracy"] = final_acc
    logger.info("Done. NAUC=%.4f | Final acc=%.4f", nauc, final_acc)

    # --- Local JSON (opt-in) ---
    if cfg.get("save_results", False):
        from hydra.utils import get_original_cwd  # noqa: PLC0415

        results_file = Path(cfg.results_file)
        if not results_file.is_absolute():
            results_file = Path(get_original_cwd()) / results_file
        _append_result(
            results_file,
            {
                "method": cfg.method.name,
                "dataset": cfg.dataset.name,
                "strategy": cfg.al_strategy.name,
                "seed": cfg.seed,
                "nauc": nauc,
                "final_accuracy": final_acc,
                "iterations": iterations,
            },
        )
        logger.info("Results appended to %s", results_file)

    run.finish()
    return nauc


if __name__ == "__main__":
    main()
