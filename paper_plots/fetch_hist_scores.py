"""Fetch the per-sample OOD scores behind the paper's histograms into the plot cache.

The CSV exports have no per-sample scores, so this is the one step that needs
wandb. Run it after ``paper_plots.import_csv``, which clears the cache::

    uv run python -m paper_plots.fetch_hist_scores results_long.csv

Scores come from the artifacts logged by the exact run named in the export, not
from ``<artifact>:latest``: OOD artifact names do not contain alpha, so ``latest``
can belong to another point of a sweep. A run may have logged several versions
after a re-evaluation, so the newest pair that reproduces the exported AUROC is
kept. A histogram whose scores do not reproduce it is reported and not saved.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np

from paper_plots import cache
from paper_plots.import_csv import _alpha, _is_pinned, _read, method_key
from paper_plots.targets import ENTITY

# --- edit me -----------------------------------------------------------------
# (dataset, method, OOD dataset) of every histogram the paper shows.
PAPER_HISTOGRAMS: tuple[tuple[str, str, str], ...] = (
    ("imagenet", "het_net", "inaturalist"),
    ("imagenet", "bayesian", "inaturalist"),
    ("imagenet", "credal_bnn", "inaturalist"),
    ("cifar10", "base", "places365"),
    ("cifar10", "credal_relative_likelihood", "places365"),
    ("cifar10", "sngp", "places365"),
)
MEASURE = "default"
DECOMPOSITION = "epistemic"
AUROC_TOLERANCE = 1e-3
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Cell:
    """One seed of one histogram, with the exported AUROC its scores must reproduce."""

    project: str
    run_id: str
    dataset: str
    base_model: str
    method: str
    seed: int
    ood_dataset: str
    auroc: float


def auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """Compute the AUROC for OOD samples scoring higher, counting ties as one half."""
    all_scores = np.concatenate([id_scores, ood_scores])
    _, inverse, counts = np.unique(all_scores, return_inverse=True, return_counts=True)
    midranks = np.cumsum(counts) - (counts - 1) / 2
    n_id, n_ood = id_scores.size, ood_scores.size
    rank_sum = midranks[inverse][n_id:].sum()
    return float((rank_sum - n_ood * (n_ood + 1) / 2) / (n_id * n_ood))


def find_cells(rows: list[dict[str, str]]) -> list[Cell]:
    """Select the export rows that name the run behind each paper histogram."""
    wanted = set(PAPER_HISTOGRAMS)
    cells: dict[tuple[str, str, str, int], Cell] = {}
    for row in rows:
        if row["task"] != "ood" or row["metric"] != "auroc" or row["measure"] != MEASURE:
            continue
        method = method_key(row)
        if (row["dataset"], method, row["ood_dataset"]) not in wanted or not _is_pinned(method, _alpha(row)):
            continue
        key = (row["dataset"], method, row["ood_dataset"], int(row["seed"]))
        if key in cells:
            msg = f"two runs export the same histogram cell: {key}"
            raise ValueError(msg)
        cells[key] = Cell(
            project=row["project"],
            run_id=row["run_id"],
            dataset=row["dataset"],
            base_model=row["base_model"],
            method=method,
            seed=int(row["seed"]),
            ood_dataset=row["ood_dataset"],
            auroc=float(row["value"]),
        )
    missing = wanted - {(c.dataset, c.method, c.ood_dataset) for c in cells.values()}
    if missing:
        msg = f"the export has no runs for these histograms: {sorted(missing)}"
        raise ValueError(msg)
    return list(cells.values())


def _newest_first(artifacts: list[Any], artifact_type: str, name_part: str) -> list[Any]:
    matches = [a for a in artifacts if a.type == artifact_type and name_part in a.name]
    return sorted(matches, key=lambda a: a.created_at, reverse=True)


def _load(artifact: Any, filename: str) -> np.ndarray:  # noqa: ANN401
    with tempfile.TemporaryDirectory() as directory:
        artifact.download(root=directory)
        return np.asarray(np.load(Path(directory) / filename, allow_pickle=False), dtype=float)


def resolve_scores(api: Any, cell: Cell) -> tuple[np.ndarray, np.ndarray, float | None]:  # noqa: ANN401
    """Return the newest score pair of the run that reproduces the exported AUROC.

    Returns:
        ``(id_scores, ood_scores, recomputed_auroc)``. Both arrays are empty when no
        pair matches; the AUROC is then that of the closest pair, or ``None`` when
        the run logged no usable artifacts.
    """
    logged = list(api.run(f"{ENTITY}/{cell.project}/{cell.run_id}").logged_artifacts())
    suffix = f"-{MEASURE}-{DECOMPOSITION}"
    id_artifacts = _newest_first(logged, "id_scores", suffix)
    ood_artifacts = _newest_first(logged, "ood_scores", f"-{cell.ood_dataset}{suffix}")
    empty = np.empty((0,), dtype=float)
    closest: float | None = None
    id_arrays = {id(a): _load(a, "id_scores.npy") for a in id_artifacts}
    for id_artifact, ood_artifact in product(id_artifacts, ood_artifacts):
        id_scores = id_arrays[id(id_artifact)]
        ood_scores = _load(ood_artifact, "ood_scores.npy")
        value = auroc(id_scores, ood_scores)
        if abs(value - cell.auroc) <= AUROC_TOLERANCE:
            return id_scores, ood_scores, value
        if closest is None or abs(value - cell.auroc) < abs(closest - cell.auroc):
            closest = value
    return empty, empty, closest


def save_scores(cell: Cell, id_scores: np.ndarray, ood_scores: np.ndarray) -> None:
    """Merge one seed's score arrays into its existing cache record."""
    records = cache.load_records(ENTITY, cell.project, cell.dataset, cell.base_model, cell.method, [cell.seed])
    if len(records) != 1:
        msg = f"expected one cache record for {cell}, found {len(records)}; run paper_plots.import_csv first"
        raise ValueError(msg)
    record = records[0]
    summary = record["summary"]
    summary[f"ood/{MEASURE}/{DECOMPOSITION}/id_scores"] = id_scores
    summary[f"ood/{cell.ood_dataset}/{MEASURE}/{DECOMPOSITION}/ood_scores"] = ood_scores
    cache.save_record(
        entity=ENTITY,
        project=cell.project,
        dataset=cell.dataset,
        base_model=cell.base_model,
        method=cell.method,
        seed=cell.seed,
        run_id=record["run_id"],
        name=record["name"],
        config=record["config"],
        summary=summary,
    )


def fetch(api: Any, rows: list[dict[str, str]]) -> int:  # noqa: ANN401
    """Fetch, verify, and cache all histogram scores; return the number of failed cells."""
    cells = find_cells(rows)
    failed = 0
    for cell in cells:
        label = f"{cell.dataset}/{cell.method}/{cell.ood_dataset}/seed_{cell.seed} ({cell.run_id})"
        id_scores, ood_scores, value = resolve_scores(api, cell)
        if id_scores.size == 0:
            failed += 1
            found = "no score artifacts" if value is None else f"closest AUROC {value:.4f}"
            print(f"FAILED {label}: export says {cell.auroc:.4f}, {found}; not saved")
            continue
        save_scores(cell, id_scores, ood_scores)
        print(f"ok     {label}: AUROC {value:.4f} (n_id={id_scores.size}, n_ood={ood_scores.size})")
    print(f"done: {len(cells) - failed} of {len(cells)} cells verified and cached.")
    return failed


def main() -> None:
    """Parse the arguments and run the fetch against the wandb API."""
    parser = argparse.ArgumentParser(description="Fetch the OOD scores for the paper's histograms.")
    parser.add_argument("results", type=Path, help="Path to results_long.csv.")
    args = parser.parse_args()

    import wandb  # noqa: PLC0415

    sys.exit(1 if fetch(wandb.Api(timeout=60), _read(args.results)) else 0)


if __name__ == "__main__":
    main()
