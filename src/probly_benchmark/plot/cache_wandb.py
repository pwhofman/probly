r"""Pre-warm the local wandb cache for a comparison config.

Iterates over every method in a plot comparison config and pulls the SP/OOD
summary slice we care about from wandb into the on-disk cache. Run this
occasionally (e.g. when new wandb runs land); afterwards the plot scripts
can run offline with ``cache.mode=read``.

Usage::

    uv run cache_wandb.py comparison=cifar10_all_methods \
        wandb.project=cifar10-benchmark
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import hydra
from omegaconf import DictConfig, OmegaConf

from probly_benchmark.paths import CACHE_PATH
from probly_benchmark.plot import cache

_CONFIG_DIR = Path(__file__).parent.parent / "configs"


@hydra.main(version_base=None, config_path="../plot_configs", config_name="cache_wandb")
def main(cfg: DictConfig) -> None:
    """Refresh the local wandb cache for every method in the comparison config.

    Args:
        cfg: Hydra config composed from a ``cache_wandb`` comparison config.
    """
    recipe_raw = OmegaConf.load(_CONFIG_DIR / "recipe" / f"{cfg.recipe}.yaml")
    recipe = recipe_raw if isinstance(recipe_raw, DictConfig) else DictConfig({})
    dataset: str = cfg.get("dataset") or recipe.dataset
    base_model: str = cfg.get("base_model") or recipe.base_model
    mode: str = cfg.get("cache", DictConfig({})).get("mode", "refresh")
    default_seeds = list(cfg.seeds) if cfg.get("seeds") else None

    print(f"Cache root: {CACHE_PATH}")
    print(f"Mode: {mode}")
    print(f"Dataset: {dataset}, base model: {base_model}")
    print(f"Project: {cfg.wandb.entity}/{cfg.wandb.project}")
    print(f"Methods: {len(cfg.methods)}")
    print()

    total_records = 0
    for entry in cast("list[DictConfig]", cfg.methods):
        records = cache.fetch_with_cache(
            cfg.wandb.entity,
            cfg.wandb.project,
            dataset=dataset,
            base_model=base_model,
            method_entry=entry,
            default_seeds=default_seeds,
            mode=mode,
        )
        seeds = sorted(str(r["config"].get("seed")) for r in records)
        total_records += len(records)
        print(f"  {entry.name:<40s} {len(records):>2d} run(s) | seeds={seeds}")

    print()
    print(f"Cached {total_records} run record(s) under {CACHE_PATH}.")


if __name__ == "__main__":
    main()
