"""Hydra entry point for active learning experiments."""

from __future__ import annotations

import fcntl
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
from probly_benchmark.al_estimator import BaselineALEstimator, UQALEstimator
from probly_benchmark.data import get_data_al
from probly_benchmark.utils import set_seed

logger = logging.getLogger(__name__)


# Methods trained as plain cross-entropy baselines. Everything else goes
# through :class:`UQALEstimator` (probly's full UQ pipeline).
_BASELINE_METHODS = frozenset({"plain", "ensemble"})
# Strategies that consume only ``predict_proba`` / ``embed`` from a
# baseline-trained model. ``ensemble`` overlaps both sets via the
# ``ensemble`` method-name; the dispatch in :func:`_build_estimator`
# routes ``ensemble x uncertainty`` through :class:`UQALEstimator`.
_BASELINE_STRATEGIES = frozenset({"random", "margin", "badge"})
_UQ_STRATEGIES = frozenset({"random", "uncertainty"})


def _append_result(results_file: Path, result: dict[str, Any]) -> None:
    """Append ``result`` to a shared JSON list at ``results_file``.

    The file holds a list of run dicts. Existing entries with the same
    ``(method, dataset, strategy, seed)`` key are replaced so re-runs
    overwrite previous results. Uses POSIX advisory locking
    (:func:`fcntl.flock`) to serialize concurrent writers (e.g. Hydra
    multirun with multiple workers writing to the same file).
    """
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


def _resolve_al_overrides(
    cfg: DictConfig,
) -> tuple[dict[str, Any], dict[str, Any], int, str | None]:
    """Resolve method params/train with optional ``active_learning`` overrides.

    If ``cfg.method.active_learning`` exists, its ``params`` and ``train``
    sub-keys are deep-merged on top of the shared ``cfg.method.params`` and
    ``cfg.method.train`` respectively.  ``num_samples`` and ``measure`` are
    taken from the AL section if present, otherwise from the top-level config.

    Returns:
        Tuple of ``(method_params, train_kwargs, num_samples, measure)``.
    """
    al = cfg.method.get("active_learning")

    def _merge(key: str) -> dict[str, Any]:
        base = cfg.method.get(key)
        override = al.get(key) if al else None
        if base and override:
            return OmegaConf.to_container(OmegaConf.merge(base, override), resolve=True)  # ty: ignore[invalid-return-type]
        if override:
            return OmegaConf.to_container(override, resolve=True)  # ty: ignore[invalid-return-type]
        if base:
            return OmegaConf.to_container(base, resolve=True)  # ty: ignore[invalid-return-type]
        return {}

    return (
        _merge("params"),
        _merge("train"),
        int(al.get("num_samples", cfg.num_samples)) if al else cfg.num_samples,
        al.get("measure", cfg.measure) if al else cfg.measure,
    )


def _build_estimator(
    cfg: DictConfig,
    *,
    num_classes: int,
    in_features: int | None,
    device: torch.device,
) -> BaselineALEstimator | UQALEstimator:
    """Construct the AL estimator from the config.

    Routes ``(method, strategy)`` to the appropriate estimator and rejects
    nonsensical combinations (e.g. ``efficient_credal_prediction x margin``
    or ``plain x uncertainty``).

    Args:
        cfg: Hydra config.
        num_classes: Number of output classes.
        in_features: Input feature dimension for tabular base models;
            ``None`` for image base models.
        device: Torch device for training and inference.

    Raises:
        ValueError: If ``(cfg.method.name, cfg.al_strategy.name)`` is not
            a supported combination.
    """
    method = cfg.method.name
    strategy = cfg.al_strategy.name
    base_model_name = cfg.dataset.base_model

    method_params, train_kwargs, num_samples, measure = _resolve_al_overrides(cfg)

    if method in _BASELINE_METHODS and strategy in _BASELINE_STRATEGIES:
        num_members = int(method_params.get("num_members", 1)) if method == "ensemble" else 1
        return BaselineALEstimator(
            method_name=method,
            cfg=cfg,
            base_model_name=base_model_name,
            num_classes=num_classes,
            device=device,
            in_features=in_features,
            num_members=num_members,
        )

    if strategy in _UQ_STRATEGIES and method != "plain":
        return UQALEstimator(
            method_name=method,
            method_params=method_params,
            train_kwargs=train_kwargs,
            cfg=cfg,
            base_model_name=base_model_name,
            model_type=cfg.model_type,
            num_classes=num_classes,
            device=device,
            in_features=in_features,
            measure=measure,
            num_samples=num_samples,
        )

    msg = (
        f"Method {method!r} does not support strategy {strategy!r}. "
        f"Baselines (plain, ensemble) support {sorted(_BASELINE_STRATEGIES)}; "
        f"UQ methods support {sorted(_UQ_STRATEGIES)}. "
        f"`plain` is a baseline only and has no uncertainty measure; "
        f"UQ methods do not naturally expose the class probabilities or "
        f"embeddings that margin/badge require."
    )
    raise ValueError(msg)


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
    estimator = _build_estimator(
        cfg,
        num_classes=num_classes,
        in_features=in_features if cfg.dataset.type == "tabular" else None,
        device=device,
    )

    # --- Strategy ---
    strategy = _build_query_strategy(cfg)

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

    # --- Append results to the shared local JSON file (opt-in) ---
    # Off by default; cluster runs use wandb as the source of truth.
    # Enable locally with ``save_results=true`` (override ``results_file`` to
    # change the path). Resolved relative to Hydra's original cwd because
    # Hydra chdir's into the per-run output dir.
    if cfg.get("save_results", False):
        from hydra.utils import get_original_cwd  # noqa: PLC0415

        results_file = Path(cfg.results_file)
        if not results_file.is_absolute():
            results_file = Path(get_original_cwd()) / results_file
        results = {
            "method": cfg.method.name,
            "dataset": cfg.dataset.name,
            "strategy": cfg.al_strategy.name,
            "seed": cfg.seed,
            "nauc": nauc,
            "final_accuracy": final_acc,
            "iterations": iterations,
        }
        _append_result(results_file, results)
        logger.info("Results appended to %s", results_file)

    run.finish()
    return nauc


if __name__ == "__main__":
    main()
