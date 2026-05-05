r"""Pre-warm the local AL wandb cache for a sweep matrix.

Iterates the ``matrices`` block list in the order it appears in the YAML
(mirroring the user's ``run_local_*.sh`` block ordering: Block 2 -- UQ x
uncertainty x notions; Block 3 -- UQ x {random, margin}; Block 1 -- baselines
x losses; Block 4 -- baselines x calibration). For each block, every
``methods x strategies x notions x supervised_losses x calibrations x
datasets x seeds`` combination is fetched into the on-disk JSON cache.

Usage::

    uv run cache_wandb_al.py wandb.project=al_openml_v1600_0505
"""

from __future__ import annotations

from itertools import product
from typing import Any

import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf

from probly_benchmark.paths import CACHE_PATH
from probly_benchmark.plot import cache_al


def _as_list(value: Any) -> list[Any]:  # noqa: ANN401
    if isinstance(value, ListConfig):
        materialized = OmegaConf.to_container(value, resolve=True)
        if isinstance(materialized, list):
            return list(materialized)
        return [materialized]
    if isinstance(value, list | tuple):
        return list(value)
    return [value]


@hydra.main(version_base=None, config_path="../plot_configs", config_name="cache_wandb_al")
def main(cfg: DictConfig) -> None:
    """Refresh the AL cache for every block in ``cfg.matrices``."""
    mode: str = cfg.get("cache", DictConfig({})).get("mode", "refresh")
    default_seeds = list(cfg.seeds) if cfg.get("seeds") else None
    datasets_global = list(cfg.datasets)
    matrices = cfg.get("matrices") or []

    print(f"Cache root: {CACHE_PATH}")
    print(f"Mode: {mode}")
    print(f"Project: {cfg.wandb.entity}/{cfg.wandb.project}")
    print(f"Default datasets: {datasets_global} | Default seeds: {default_seeds}")
    print(f"Blocks: {len(matrices)}")
    print()

    grand_total = 0
    for block_idx, block in enumerate(matrices):
        methods = _as_list(block.get("methods", []))
        strategies = _as_list(block.get("strategies", ["uncertainty"]))
        notions = _as_list(block.get("notions", ["epistemic"]))
        supervised_losses = _as_list(block.get("supervised_losses", ["cross_entropy"]))
        calibrations = _as_list(block.get("calibrations", ["none"]))
        datasets = _as_list(block.get("datasets", datasets_global))
        seeds = _as_list(block.get("seeds", default_seeds)) if block.get("seeds") is not None else default_seeds

        n_combos = (
            len(methods) * len(strategies) * len(notions) * len(supervised_losses) * len(calibrations) * len(datasets)
        )
        print(f"=== Block {block_idx + 1}/{len(matrices)}: {n_combos} combos ===")
        print(f"  methods={methods}")
        print(f"  strategies={strategies} | notions={notions}")
        print(f"  losses={supervised_losses} | calibrations={calibrations}")
        print(f"  datasets={datasets} | seeds={seeds}")

        block_total = 0
        for ds_key, strategy, notion, loss, cal in product(
            datasets, strategies, notions, supervised_losses, calibrations
        ):
            for method in methods:
                records = cache_al.fetch_with_cache(
                    cfg.wandb.entity,
                    cfg.wandb.project,
                    ds_key=ds_key,
                    method_entry={"name": method},
                    strategy=strategy,
                    notion=notion,
                    supervised_loss=loss,
                    calibration=cal,
                    default_seeds=seeds,
                    mode=mode,
                )
                block_total += len(records)
                seeds_str = sorted(str(r["config"].get("seed")) for r in records)
                print(
                    f"  {ds_key}/{method}/{strategy}/notion={notion}/loss={loss}/cal={cal} "
                    f"-> {len(records)} run(s) | seeds={seeds_str}"
                )
        print(f"  block total: {block_total} record(s)\n")
        grand_total += block_total

    print(f"Cached {grand_total} run record(s) under {CACHE_PATH}.")


if __name__ == "__main__":
    main()
