"""Load the camera-ready CSV exports into the local plot caches.

The plot scripts only read the JSON caches that the wandb fetchers write. This
script fills the same caches from ``results_long.csv`` and ``sp_curves.csv``, so
every plot script runs unchanged and no wandb access is needed::

    uv run python -m paper_plots.import_csv results_long.csv sp_curves.csv

Per-sample OOD scores are not part of the export, so ``paper_plots.ood_hist``
renders nothing until they are fetched separately.

The import fails rather than picks a winner when two runs claim the same cell,
when a curve is incomplete, or when the export contradicts itself.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from pathlib import Path
from typing import Any

from paper_plots import cache, sweep_cache
from paper_plots.fetch_first_order_sweep import SWEEP_METHODS
from paper_plots.targets import ENTITY, TARGETS

# Method settings the paper reports. Rows with any other alpha only feed the
# coverage-efficiency sweep.
ALPHA_PINS: dict[str, float] = {
    "credal_relative_likelihood": 1.0,
    "efficient_credal_prediction": 0.8,
    "base_label_relaxation": 0.05,
}
CONSISTENCY_TOLERANCE = 1e-9
PROJECTS = {(t.dataset, t.base_model): t.project for t in TARGETS}


def method_key(row: dict[str, str]) -> str:
    """Return the cache key of a row; variants of ``base`` get a synthetic name."""
    if row["method"] == "base" and row["variant"] != "none":
        return f"base_{row['variant']}"
    return row["method"]


def _alpha(row: dict[str, str]) -> float | None:
    return float(row["alpha"]) if row["alpha"] else None


def _is_pinned(method: str, alpha: float | None) -> bool:
    """``True`` when a row with this alpha belongs to the method's reported setting."""
    return method not in ALPHA_PINS or alpha == ALPHA_PINS[method]


def _read(path: Path) -> list[dict[str, str]]:
    """Read a CSV, dropping rows the export marks as duplicates of another run."""
    with path.open(newline="") as handle:
        return [row for row in csv.DictReader(handle) if not row["note"]]


def _set_once(summary: dict[str, Any], key: str, value: Any, *, context: str) -> None:  # noqa: ANN401
    """Store ``value`` under ``key``, refusing to overwrite a different value."""
    if key in summary and summary[key] != value:
        msg = f"conflicting values for {key} in {context}: {summary[key]} vs {value}"
        raise ValueError(msg)
    summary[key] = value


def _build_records(rows: list[dict[str, str]]) -> tuple[dict[tuple, dict], dict[tuple, dict]]:
    """Group result rows into canonical cache cells and first-order sweep cells."""
    canonical: dict[tuple, dict] = defaultdict(lambda: {"summary": {}, "run_ids": set()})
    sweep: dict[tuple, dict] = defaultdict(lambda: {"summary": {}, "run_ids": set()})
    for row in rows:
        task, metric = row["task"], row["metric"]
        if task == "test":
            continue
        method, alpha = method_key(row), _alpha(row)
        cell = (row["dataset"], row["base_model"], method, int(row["seed"]))
        context = f"{cell} (alpha={alpha}, run {row['run_id']})"
        value = float(row["value"])

        if task == "first_order" and alpha is not None and method in SWEEP_METHODS:
            # Sweep points stay out of the canonical cell unless they are the reported setting.
            target = sweep[(*cell, alpha)]
            _set_once(target["summary"], metric, value, context=context)
            target["run_ids"].add(row["run_id"])
            if ALPHA_PINS.get(method) != alpha:
                continue
        elif not _is_pinned(method, alpha):
            continue

        target = canonical[cell]
        target["run_ids"].add(row["run_id"])
        if task == "first_order":
            _set_once(target["summary"], metric, value, context=context)
        elif task == "sp" and metric == "aurc":
            # The wandb summary key is called "auroc", but it holds the area under the risk curve.
            _set_once(target["summary"], f"sp/{row['measure']}/{row['decomposition']}/auroc", value, context=context)
        elif task == "ood":
            key = f"ood/{row['ood_dataset']}/{row['measure']}/{row['decomposition']}/{metric}"
            _set_once(target["summary"], key, value, context=context)
    return canonical, sweep


def _check_acc_auc(rows: list[dict[str, str]]) -> None:
    """Check that every ``acc_rejection_auc`` row equals ``1 - aurc`` of its run."""
    aurc = {(r["run_id"], r["measure"]): float(r["value"]) for r in rows if r["task"] == "sp" and r["metric"] == "aurc"}
    for row in rows:
        if row["task"] == "sp" and row["metric"] == "acc_rejection_auc":
            expected = 1.0 - aurc[(row["run_id"], row["measure"])]
            if abs(float(row["value"]) - expected) > CONSISTENCY_TOLERANCE:
                msg = f"acc_rejection_auc != 1 - aurc for run {row['run_id']}"
                raise ValueError(msg)


def _attach_curves(canonical: dict[tuple, dict], rows: list[dict[str, str]]) -> None:
    """Add per-seed ``bin_losses`` arrays, checked for completeness, to the canonical cells."""
    curves: dict[tuple, dict[int, float]] = defaultdict(dict)
    sizes: dict[tuple, int] = {}
    for row in rows:
        method = method_key(row)
        if not _is_pinned(method, _alpha(row)):
            continue
        key = (row["dataset"], row["base_model"], method, int(row["seed"]), row["measure"], row["decomposition"])
        curves[key][int(row["bin_index"])] = float(row["bin_loss"])
        sizes[key] = int(row["n_bins"])
    for (*cell, measure, decomposition), bins in curves.items():
        key = (*cell, measure, decomposition)
        if sorted(bins) != list(range(sizes[key])):
            msg = f"incomplete curve for {key}: {len(bins)} of {sizes[key]} bins"
            raise ValueError(msg)
        if tuple(cell) not in canonical:
            msg = f"curve without a matching result row: {key}"
            raise ValueError(msg)
        canonical[tuple(cell)]["summary"][f"sp/{measure}/{decomposition}/bin_losses"] = [bins[i] for i in sorted(bins)]


def _wipe(root: Path) -> None:
    """Remove JSON records below ``root`` so runs dropped from the export do not linger."""
    if root.exists():
        for path in root.rglob("seed_*.json"):
            path.unlink()


def import_csvs(results: Path, curves: Path) -> None:
    """Replace the SP/OOD and sweep caches with the contents of the two CSV exports."""
    rows = _read(results)
    _check_acc_auc(rows)
    canonical, sweep = _build_records(rows)
    _attach_curves(canonical, _read(curves))

    _wipe(cache.cache_root())
    _wipe(sweep_cache.cache_root())
    for (dataset, base_model, method, seed), record in canonical.items():
        cache.save_record(
            entity=ENTITY,
            project=PROJECTS[(dataset, base_model)],
            dataset=dataset,
            base_model=base_model,
            method=method,
            seed=seed,
            run_id=",".join(sorted(record["run_ids"])),
            name=f"{method}_{base_model}_{dataset}",
            config={"seed": seed, "dataset": dataset, "base_model": base_model, "method": {"name": method}},
            summary=record["summary"],
        )
    for (dataset, base_model, method, seed, alpha), record in sweep.items():
        sweep_cache.save_record(
            entity=ENTITY,
            project=PROJECTS[(dataset, base_model)],
            dataset=dataset,
            base_model=base_model,
            method=method,
            alpha=alpha,
            seed=seed,
            run_id=",".join(sorted(record["run_ids"])),
            name=f"{method}_{base_model}_{dataset}",
            config={"seed": seed, "dataset": dataset, "base_model": base_model, "method": {"name": method}},
            summary=record["summary"],
        )
    print(f"wrote {len(canonical)} records to {cache.cache_root()}")
    print(f"wrote {len(sweep)} sweep records to {sweep_cache.cache_root()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load the CSV exports into the plot caches.")
    parser.add_argument("results", type=Path, help="Path to results_long.csv.")
    parser.add_argument("curves", type=Path, help="Path to sp_curves.csv.")
    args = parser.parse_args()
    import_csvs(args.results, args.curves)
