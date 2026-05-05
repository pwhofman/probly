r"""Pre-warm the local AL wandb cache for a comparison config.

Iterates ``methods x datasets x strategies x seeds`` and pulls each combo's
wandb run summary + history into the on-disk JSON cache. Run this when new
AL runs land.

Usage::

    uv run cache_wandb_al.py comparison=al_openml_all_methods \
        wandb.project=al_openml_v1600_0505
"""

from __future__ import annotations

from typing import cast

import hydra
from omegaconf import DictConfig

from probly_benchmark.paths import CACHE_PATH
from probly_benchmark.plot import cache_al


@hydra.main(version_base=None, config_path="../plot_configs", config_name="cache_wandb_al")
def main(cfg: DictConfig) -> None:
    """Refresh the AL cache for every method x dataset x strategy combo."""
    mode: str = cfg.get("cache", DictConfig({})).get("mode", "refresh")
    default_seeds = list(cfg.seeds) if cfg.get("seeds") else None
    datasets = list(cfg.datasets)
    strategies = list(cfg.strategies)

    print(f"Cache root: {CACHE_PATH}")
    print(f"Mode: {mode}")
    print(f"Project: {cfg.wandb.entity}/{cfg.wandb.project}")
    print(f"Methods: {len(cfg.methods)}  |  Datasets: {datasets}  |  Strategies: {strategies}")
    print()

    total = 0
    for ds_key in datasets:
        for strategy in strategies:
            print(f"--- {ds_key} / {strategy} ---")
            for entry in cast("list[DictConfig]", cfg.methods):
                records = cache_al.fetch_with_cache(
                    cfg.wandb.entity,
                    cfg.wandb.project,
                    ds_key=ds_key,
                    method_entry=entry,
                    strategy=strategy,
                    default_seeds=default_seeds,
                    mode=mode,
                )
                seeds = sorted(str(r["config"].get("seed")) for r in records)
                total += len(records)
                print(f"  {entry.name:<40s} {len(records):>2d} run(s) | seeds={seeds}")
            print()

    print(f"Cached {total} run record(s) under {CACHE_PATH}.")


if __name__ == "__main__":
    main()
