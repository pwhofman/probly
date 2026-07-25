#!/usr/bin/env python3
r"""Run the AL sweep for one dataset, skipping anything already finished — fully offline.

Covers all settings of the AL benchmark (no conformal-prediction blocks):

- Baselines:        ``method=base`` x {margin, entropy, least_confident} x 10 seeds
- Calibration:      ``method=base`` + {temperature_scaling, vector_scaling}
- Supervised loss:  ``method=base`` + {label_smoothing, label_relaxation}
- UQ methods x {margin, random}
- UQ uncertainty:epistemic per method
- UQ uncertainty:total per method (excluding ddu)

The target dataset is selected with ``--dataset`` (an ``al_dataset`` config
name, e.g. ``openml_6`` or ``cifar10``). Formerly ``scripts/run_openml6.py``.

Determines what's already done by reading a local ``.pkl`` / ``.csv`` (defaults to
``scripts/al_analysis_out/wandb_cache_runs.pkl`` produced by
``scripts/inspect_al_runs.py``). No wandb roundtrip happens here — refresh the
cache first if you want fresh state.

A combo is considered "done" if a finished run with a matching tuple of
``(method, strategy, notion, seed, calibration, supervised_loss)`` exists in the
seed file(s) for the selected dataset. Legacy notion spellings in old runs
(``EU``/``TU``/``AU``) are matched against the current names
(``epistemic``/``total``/``aleatoric``).

Usage::

    # Pre-req: populate the cache (one-time or whenever you want fresh state).
    uv run python scripts/inspect_al_runs.py --refresh

    # Dry-run: print the missing combos and a summary; takes ~1s.
    uv run python scripts/run_al_sweep.py --dataset cifar10

    # Actually run the missing combos sequentially (continues on failure).
    uv run python scripts/run_al_sweep.py --dataset cifar10 --execute

    # Smoke test: run only the first 3 missing combos and stop.
    uv run python scripts/run_al_sweep.py --dataset cifar10 --execute --limit 3

    # Use a different seed file (e.g. a hand-curated CSV).
    uv run python scripts/run_al_sweep.py --dataset openml_6 --seed-file my_done.csv

    # Combine multiple seed files (the union counts as 'done').
    uv run python scripts/run_al_sweep.py --dataset openml_6 \\
        --seed-file scripts/al_analysis_out/wandb_cache_runs.pkl \\
        --seed-file extra_runs.csv

Seed-file schema (CSV or pickled DataFrame):

    Required: method, strategy, seed, state, dataset
    Optional: notion, calibration, supervised_loss

Only rows with ``state == "finished"`` and a ``dataset`` matching ``--dataset``
are counted.

New runs are launched with ``wandb.project=max-test`` and ``+wandb.entity=probly``;
edit the ``WANDB_PROJECT`` / ``WANDB_ENTITY`` constants if you want to redirect.
Default is dry-run; pass ``--execute`` to actually launch missing combos.
"""
# ruff: noqa: T201, ANN401, D103

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys
from typing import TYPE_CHECKING, Any, cast

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

# ---- Sweep spec --------------------------------------------------------------------

DEFAULT_SEED_FILE = "scripts/al_analysis_out/wandb_cache_runs.pkl"
# Per-dataset default device passed as ``device=...``. ``None`` omits the
# override so ``utils.get_device`` auto-selects (cuda > mps > cpu).
DATASET_DEVICES: dict[str, str | None] = {
    "openml_6": "cpu",
    "openml_155": "cpu",
    "openml_156": "cpu",
    "cifar10": None,
}
SEEDS: tuple[int, ...] = tuple(range(10))
BASE_STRATEGIES: tuple[str, ...] = ("margin", "entropy", "least_confident")
# Ensemble-based methods are slow to train (multiple base models per run); they
# are placed at the tail of UQ_METHODS so the per-block iteration runs them
# last, and the global execution order also pulls them to the very end via
# ENSEMBLE_METHODS below.
ENSEMBLE_METHODS: frozenset[str] = frozenset({"ensemble", "credal_ensembling"})

UQ_METHODS: tuple[str, ...] = (
    "dropout",
    "dropconnect",
    "bayesian",
    "dare",
    "evidential_classification",
    "posterior_network",
    "credal_relative_likelihood",
    "ddu",
    # ensemble-based — kept last
    "ensemble",
    "credal_ensembling",
)
UQ_TU_METHODS: tuple[str, ...] = tuple(m for m in UQ_METHODS if m != "ddu")
CALIBRATIONS: tuple[str, ...] = ("temperature_scaling", "vector_scaling")
SUPERVISED_LOSSES: tuple[str, ...] = ("label_smoothing", "label_relaxation")

# Where to send the new runs we launch.
WANDB_PROJECT = "max-test"
WANDB_ENTITY = "probly"

Combo = dict[str, Any]


# ---- Key normalization & matching --------------------------------------------------

# Per-field default values that mean "no override". Treated as None when keying so
# e.g. ``supervised_loss == "cross_entropy"`` (explicit default in wandb) matches a
# combo where supervised_loss isn't set.
_FIELD_DEFAULTS: dict[str, tuple[str, ...]] = {
    "notion": (),
    "calibration": ("none",),
    "supervised_loss": ("cross_entropy",),
    "conformal": ("none",),
}

# Legacy notion spellings (pre-#462) still present in old wandb runs.
_NOTION_ALIASES: dict[str, str] = {"EU": "epistemic", "TU": "total", "AU": "aleatoric"}


def _norm_field(value: Any, field: str) -> str | None:
    if value is None:
        return None
    s = str(value)
    if field == "notion":
        s = _NOTION_ALIASES.get(s, s)
    if s in _FIELD_DEFAULTS.get(field, ()):
        return None
    return s


def _key(combo: Combo) -> tuple[Any, ...]:
    return (
        str(combo["method"]),
        str(combo["strategy"]),
        _norm_field(combo.get("notion"), "notion"),
        int(combo["seed"]),
        _norm_field(combo.get("calibration"), "calibration"),
        _norm_field(combo.get("supervised_loss"), "supervised_loss"),
    )


# ---- File-based seeding (read finished runs from CSV / pickle) ---------------------


def _row_value(row: pd.Series, col: str) -> Any:
    if col not in row.index:
        return None
    v = row[col]
    return None if pd.isna(v) else v


def load_seed_file(path: Path, dataset_full: str) -> set[tuple[Any, ...]]:
    """Load finished combos from a .pkl or .csv with the inspect_al_runs cache schema.

    Required columns: ``method``, ``strategy``, ``seed``, ``state``, ``dataset``.
    Optional: ``notion``, ``calibration``, ``supervised_loss``.
    """
    if path.suffix == ".pkl":
        df = cast("pd.DataFrame", pd.read_pickle(path))  # noqa: S301
    elif path.suffix in (".csv", ".tsv"):
        sep = "\t" if path.suffix == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
    else:
        msg = f"Unsupported seed-file extension: {path.suffix} ({path})"
        raise ValueError(msg)

    required = {"method", "strategy", "seed", "state", "dataset"}
    missing = required - set(df.columns)
    if missing:
        msg = f"Seed file {path} is missing required columns: {sorted(missing)}"
        raise ValueError(msg)

    df = df[(df["state"] == "finished") & (df["dataset"] == dataset_full)]
    keys: set[tuple[Any, ...]] = set()
    for _, row in df.iterrows():
        strategy = _row_value(row, "strategy")
        notion = _row_value(row, "notion") if strategy == "uncertainty" else None
        method = _row_value(row, "method")
        seed = _row_value(row, "seed")
        if method is None or strategy is None or seed is None:
            continue
        keys.add(
            _key(
                {
                    "method": method,
                    "strategy": strategy,
                    "notion": notion,
                    "seed": int(seed),
                    "calibration": _row_value(row, "calibration"),
                    "supervised_loss": _row_value(row, "supervised_loss"),
                }
            )
        )
    print(f"  [{path}] loaded {len(keys)} finished combos for {dataset_full}")
    return keys


# ---- Block definitions -------------------------------------------------------------


def block_combos() -> Iterator[tuple[str, list[Combo]]]:
    yield (
        "Baselines (base, 3 strategies, 10 seeds)",
        [{"method": "base", "strategy": s, "seed": seed} for s in BASE_STRATEGIES for seed in SEEDS],
    )
    yield (
        f"Calibration (base + {{{', '.join(CALIBRATIONS)}}})",
        [
            {"method": "base", "strategy": s, "seed": seed, "calibration": cal}
            for cal in CALIBRATIONS
            for s in BASE_STRATEGIES
            for seed in SEEDS
        ],
    )
    yield (
        f"Supervised Loss (base + {{{', '.join(SUPERVISED_LOSSES)}}})",
        [
            {"method": "base", "strategy": s, "seed": seed, "supervised_loss": sup}
            for sup in SUPERVISED_LOSSES
            for s in BASE_STRATEGIES
            for seed in SEEDS
        ],
    )
    yield (
        "UQ (margin + random)",
        [
            {"method": m, "strategy": s, "seed": seed}
            for m in UQ_METHODS
            for s in ("margin", "random")
            for seed in SEEDS
        ],
    )
    yield (
        "UQ uncertainty:epistemic",
        [
            {"method": m, "strategy": "uncertainty", "notion": "epistemic", "seed": seed}
            for m in UQ_METHODS
            for seed in SEEDS
        ],
    )
    yield (
        "UQ uncertainty:total",
        [
            {"method": m, "strategy": "uncertainty", "notion": "total", "seed": seed}
            for m in UQ_TU_METHODS
            for seed in SEEDS
        ],
    )


# ---- Filtering ---------------------------------------------------------------------


def parse_seeds(spec: str) -> list[int]:
    """Parse a seed spec like ``"0-4"``, ``"0,2,5"`` or ``"3"`` into a sorted list.

    Args:
        spec: Comma-separated list of integers and/or inclusive ``a-b`` ranges.

    Raises:
        ValueError: If a token is neither an integer nor an ``a-b`` range.
    """
    seeds: set[int] = set()
    for token in spec.split(","):
        part = token.strip()
        if not part:
            continue
        if "-" in part.lstrip("-"):
            lo, _, hi = part.partition("-")
            seeds.update(range(int(lo), int(hi) + 1))
        else:
            seeds.add(int(part))
    if not seeds:
        msg = f"No seeds parsed from {spec!r}"
        raise ValueError(msg)
    return sorted(seeds)


def filter_combos(
    combos: list[Combo],
    *,
    methods: list[str] | None,
    strategies: list[str] | None,
    seeds: list[int] | None,
    supervised_losses: list[str] | None = None,
    calibrations: list[str] | None = None,
) -> list[Combo]:
    """Keep only combos matching every supplied filter (``None`` means no filter).

    ``supervised_losses`` / ``calibrations`` also drop combos that carry no such
    variant, so e.g. ``--supervised-loss label_smoothing`` selects only the
    label-smoothing runs rather than every plain run alongside them.
    """
    out = combos
    if methods:
        out = [c for c in out if c["method"] in methods]
    if strategies:
        out = [c for c in out if c["strategy"] in strategies]
    if seeds is not None:
        out = [c for c in out if c["seed"] in seeds]
    if supervised_losses:
        out = [c for c in out if c.get("supervised_loss") in supervised_losses]
    if calibrations:
        out = [c for c in out if c.get("calibration") in calibrations]
    return out


def summarize(combos: list[Combo]) -> str:
    """Return a compact ``method:count`` summary of a combo list, ordered by method."""
    counts: dict[str, int] = {}
    for c in combos:
        counts[c["method"]] = counts.get(c["method"], 0) + 1
    return ", ".join(f"{m}:{n}" for m, n in sorted(counts.items()))


# ---- Command construction ----------------------------------------------------------


def make_command(combo: Combo, dataset: str, *, device: str | None, project: str = WANDB_PROJECT) -> list[str]:
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "probly_benchmark.active_learning",
        f"method={combo['method']}",
        f"al_strategy={combo['strategy']}",
        f"seed={combo['seed']}",
        f"al_dataset={dataset}",
        "wandb.enabled=true",
        f"wandb.project={project}",
        # No ``+`` prefix: ``wandb.entity`` exists in the base config since #462,
        # and Hydra rejects appending an existing key.
        f"wandb.entity={WANDB_ENTITY}",
        "save_results=false",
    ]
    if device is not None:
        cmd.append(f"device={device}")
    if combo.get("notion"):
        cmd.append(f"al_strategy.notion={combo['notion']}")
    if combo.get("calibration"):
        cmd.append(f"calibration={combo['calibration']}")
    if combo.get("supervised_loss"):
        cmd.append(f"supervised_loss={combo['supervised_loss']}")
    return cmd


def collect_missing(
    args: argparse.Namespace,
    dataset: str,
    *,
    device: str | None,
    finished: set[tuple[Any, ...]],
) -> tuple[list[tuple[str, Combo]], int]:
    """Print the per-block plan and return ``(missing_combos, total_after_filters)``.

    Applies the ``--block`` / ``--method`` / ``--strategy`` / ``--seeds`` filters,
    drops combos already present in ``finished``, and prints one summary line per
    block (plus full commands when ``--show-commands`` is set).
    """
    seeds = parse_seeds(args.seeds) if args.seeds else None
    blocks = [b.lower() for b in args.block] if args.block else None

    missing: list[tuple[str, Combo]] = []
    grand_total = 0
    for block_name, all_combos in block_combos():
        if blocks and not any(b in block_name.lower() for b in blocks):
            continue
        combos = filter_combos(
            all_combos,
            methods=args.method,
            strategies=args.strategy,
            seeds=seeds,
            supervised_losses=args.supervised_loss,
            calibrations=args.calibration,
        )
        if not combos:
            continue
        block_missing = [c for c in combos if _key(c) not in finished]
        print(f"=== {block_name}: {len(block_missing)}/{len(combos)} missing ===")
        if block_missing:
            print(f"  {summarize(block_missing)}")
        if args.show_commands:
            for c in block_missing:
                print("  $", shlex.join(make_command(c, dataset, device=device, project=args.wandb_project)))
        missing.extend((block_name, c) for c in block_missing)
        grand_total += len(combos)
        print()
    return missing, grand_total


# ---- CLI entry point ---------------------------------------------------------------


def main(argv: Iterable[str] | None = None) -> int:  # noqa: PLR0912, PLR0915
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        required=True,
        metavar="NAME",
        help=(
            f"al_dataset config name to sweep (e.g. {', '.join(sorted(DATASET_DEVICES))}). "
            f"Also used to filter the seed file(s)."
        ),
    )
    p.add_argument(
        "--seed-file",
        action="append",
        metavar="PATH",
        help=(
            f"file(s) holding finished AL runs (.pkl or .csv with the inspect_al_runs schema). "
            f"Repeatable; the union of all files is treated as 'done'. "
            f"Default: {DEFAULT_SEED_FILE}"
        ),
    )
    p.add_argument(
        "--device",
        default=None,
        help=(
            "Hydra device override for launched runs (e.g. cpu, cuda:0, null for auto-select). "
            "Default: per-dataset (cpu for openml_*, auto-select otherwise)."
        ),
    )
    p.add_argument(
        "--wandb-project",
        default=WANDB_PROJECT,
        metavar="NAME",
        help=f"wandb project for launched runs (entity stays {WANDB_ENTITY!r}). Default: {WANDB_PROJECT!r}.",
    )
    p.add_argument(
        "--method",
        action="append",
        metavar="NAME",
        help="only sweep this method (repeatable). Default: all methods in the blocks.",
    )
    p.add_argument(
        "--strategy",
        action="append",
        metavar="NAME",
        help="only sweep this al_strategy (repeatable), e.g. margin, uncertainty, random.",
    )
    p.add_argument(
        "--seeds",
        metavar="SPEC",
        help=f"seeds to sweep, e.g. '0-2' or '0,3,5'. Default: {SEEDS[0]}-{SEEDS[-1]}.",
    )
    p.add_argument(
        "--supervised-loss",
        action="append",
        metavar="NAME",
        help=f"only these supervised-loss variants (repeatable): {', '.join(SUPERVISED_LOSSES)}.",
    )
    p.add_argument(
        "--calibration",
        action="append",
        metavar="NAME",
        help=f"only these calibration variants (repeatable): {', '.join(CALIBRATIONS)}.",
    )
    p.add_argument(
        "--block",
        action="append",
        metavar="TEXT",
        help="only blocks whose name contains TEXT, case-insensitive (repeatable), e.g. 'uncertainty'.",
    )
    p.add_argument(
        "--show-commands",
        action="store_true",
        help="print the full command for every missing combo (default: per-block summary only).",
    )
    p.add_argument("--execute", action="store_true", help="run missing combos (default: dry-run)")
    p.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop on first failed combo (default: continue and report a tally)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="when --execute, run at most N missing combos and stop.",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    dataset: str = args.dataset
    device: str | None = args.device if args.device is not None else DATASET_DEVICES.get(dataset)

    # An explicitly passed --seed-file that is missing is a typo: fail loudly. The
    # default cache merely may not exist yet (fresh campaign) -- treat that as
    # "nothing finished" so the first sweep is not blocked on running inspect first.
    explicit_seed_files = args.seed_file is not None
    seed_files = [Path(s).expanduser() for s in (args.seed_file or [DEFAULT_SEED_FILE])]
    for sf in seed_files:
        if not sf.exists() and explicit_seed_files:
            p.error(f"seed file not found: {sf}")

    seed_files = [sf for sf in seed_files if sf.exists()]
    finished: set[tuple[Any, ...]] = set()
    if seed_files:
        print(f"Reading {len(seed_files)} seed file(s) (no wandb roundtrip):")
        for sf in seed_files:
            finished |= load_seed_file(sf, dataset)
    else:
        print(
            f"No seed file at {DEFAULT_SEED_FILE}; assuming nothing has finished yet.\n"
            f"Run `uv run python scripts/inspect_al_runs.py --refresh` to skip already-finished runs."
        )
    print(f"Total finished {dataset} combos: {len(finished)}\n")

    missing, grand_total = collect_missing(args, dataset, device=device, finished=finished)

    if not grand_total:
        print("No combos matched the given --method/--strategy/--seeds/--block filters.")
        return 1

    print(f"Summary: {len(missing)} of {grand_total} combos still need to run.")
    est = len(missing) * (11 if dataset == "cifar10" else 1)
    if dataset == "cifar10" and missing:
        print(f"         ~{est} from-scratch ResNet-18 fits (11 per run at n_iterations=10).")

    if not args.execute:
        print("Dry run; pass --execute to launch them. Add --show-commands to see each command.")
        return 0

    # Stable-sort: non-ensemble combos first, ensemble-based last (across blocks).
    missing.sort(key=lambda item: item[1]["method"] in ENSEMBLE_METHODS)
    n_ensemble = sum(1 for _, c in missing if c["method"] in ENSEMBLE_METHODS)
    if n_ensemble:
        print(f"Reordered: {n_ensemble} ensemble-based combos pulled to the end.")

    if args.limit is not None:
        print(f"--limit={args.limit}: will execute at most {args.limit} of {len(missing)} combos.")
        missing = missing[: args.limit]

    failures: list[tuple[Combo, int]] = []
    for i, (block_name, combo) in enumerate(missing, start=1):
        cmd = make_command(combo, dataset, device=device, project=args.wandb_project)
        print(f"\n[{i}/{len(missing)}] {block_name}")
        print("  $", shlex.join(cmd), flush=True)
        rc = subprocess.run(cmd, check=False).returncode  # noqa: S603
        if rc != 0:
            failures.append((combo, rc))
            print(f"  (rc={rc})")
            if args.fail_fast:
                print("--fail-fast: stopping early.")
                return rc

    succeeded = len(missing) - len(failures)
    print(f"\nDone. {succeeded} succeeded, {len(failures)} failed.")
    if failures:
        print("Failures:")
        for combo, rc in failures:
            print(f"  rc={rc} {combo}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
