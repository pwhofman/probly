"""Hydra entry point for active learning experiments."""

from __future__ import annotations

import fcntl
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf
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
from probly_benchmark import utils
from probly_benchmark.al_estimators import BaseEstimator, UncertaintyEstimator
from probly_benchmark.data import get_data_al
from probly_benchmark.metadata import AL_DATASETS
from probly_benchmark.uncertainty import SUPPORTED_DECOMPOSITIONS

if TYPE_CHECKING:
    import torch

    from probly.quantification.notion import NotionName

logger = logging.getLogger(__name__)


def _resolve_notion(cfg: DictConfig) -> NotionName:
    """Read the uncertainty notion from the strategy config and validate it."""
    notion = cfg.al_strategy.get("notion", "epistemic")
    permitted = list(cfg.al_strategy.get("permitted_notions", list(SUPPORTED_DECOMPOSITIONS)))
    if notion not in permitted:
        msg = f"Notion {notion!r} is not permitted for strategy {cfg.al_strategy.name!r}. Allowed: {permitted}"
        raise ValueError(msg)
    if notion not in SUPPORTED_DECOMPOSITIONS:
        msg = f"Unknown uncertainty notion {notion!r}. Choose from {SUPPORTED_DECOMPOSITIONS}."
        raise ValueError(msg)
    return cast("NotionName", notion)


def _build_estimator(
    cfg: DictConfig,
    *,
    base_model_name: str,
    num_classes: int,
    in_features: int | None,
    device: torch.device,
) -> BaseEstimator | UncertaintyEstimator:
    """Dispatch estimator based on method name.

    ``base`` uses :class:`BaseEstimator` (with optional calibration/conformal
    via config composition).  Everything else uses :class:`UncertaintyEstimator`.
    """
    method = cfg.method.name
    raw_params = cfg.method.get("params")
    method_params = cast("dict[str, Any]", OmegaConf.to_container(raw_params, resolve=True)) if raw_params else {}
    raw_train = cfg.method.get("train")
    train_kwargs = cast("dict[str, Any]", OmegaConf.to_container(raw_train, resolve=True)) if raw_train else {}

    # ``method.active_learning`` may carry an optional ``train`` sub-block whose
    # entries override (only at AL run time) the corresponding keys from the
    # global ``method.train`` config. Anything else under ``method.active_learning``
    # is forwarded to the representer as ``rep_kwargs``.
    raw_al = cfg.method.get("active_learning")
    al_block: dict[str, Any] = cast("dict[str, Any]", OmegaConf.to_container(raw_al, resolve=True)) if raw_al else {}
    al_train_overrides = al_block.pop("train", None) or {}
    if al_train_overrides:
        train_kwargs = {**train_kwargs, **al_train_overrides}
    rep_kwargs: dict[str, Any] = al_block

    needs_conformal = cfg.conformal.name != "none"

    if method == "base" and not needs_conformal:
        return BaseEstimator(
            cfg=cfg,
            base_model_name=base_model_name,
            method_name=method,
            method_params=method_params,
            train_kwargs=train_kwargs,
            num_classes=num_classes,
            device=device,
            in_features=in_features,
        )

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
        uncertainty_notion=_resolve_notion(cfg),
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
    utils.set_seed(cfg.seed)
    device = utils.get_device(cfg.get("device", None))

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
