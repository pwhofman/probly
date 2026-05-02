"""Slim Agrawal+ARF runner: prequential loop over a few seeds, write one JSON per stream.

Reuses :func:`river_uq.prequential.run_prequential` and persists only the
columns needed for plotting. Companion to :mod:`scripts.plot_stream`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Final

import pandas as pd
from river_uq.prequential import run_prequential
from river_uq.streams import STREAM_NAMES, get_drift_window

DEFAULT_STREAMS: Final[tuple[str, ...]] = (
    "agrawal_drift_7to4",
    "agrawal_drift_4to0",
    "agrawal_drift_9to2",
)
DEFAULT_SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
DEFAULT_N_STEPS: Final[int] = 3000
DEFAULT_METHOD: Final[str] = "arf"
_DEFAULT_RESULTS_DIR: Final[Path] = Path(__file__).resolve().parent.parent / "results"

_PLOT_COLUMNS: Final[tuple[str, ...]] = (
    "t",
    "seed",
    "y_true",
    "y_pred",
    "correct",
    "total",
    "alea",
    "epi",
)


def run_streams(
    streams: list[str] | tuple[str, ...] = DEFAULT_STREAMS,
    seeds: list[int] | tuple[int, ...] = DEFAULT_SEEDS,
    n_steps: int = DEFAULT_N_STEPS,
    method: str = DEFAULT_METHOD,
    out_dir: Path = _DEFAULT_RESULTS_DIR,
) -> list[Path]:
    """Run the prequential loop a few times per stream and dump one JSON per stream.

    Args:
        streams: Stream names to run. Each must be in
            :data:`river_uq.streams.STREAM_NAMES`.
        seeds: Seeds to run for each stream.
        n_steps: Number of stream samples per (stream, seed) run.
        method: UQ method name forwarded to ``run_prequential``.
        out_dir: Directory to write ``<stream>.json`` files into.

    Returns:
        Paths to the written JSON files, in input stream order.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for stream in streams:
        if stream not in STREAM_NAMES:
            msg = f"unknown stream: {stream!r} (known: {sorted(STREAM_NAMES)})"
            raise ValueError(msg)
        frames = [run_prequential(method=method, stream_name=stream, seed=seed, n_steps=n_steps) for seed in seeds]
        df = pd.concat(frames, ignore_index=True)

        true_drift_raw = df["true_drift_t"].iloc[0]
        true_drift_t = None if pd.isna(true_drift_raw) else int(true_drift_raw)

        records = df.loc[:, list(_PLOT_COLUMNS)].to_dict(orient="records")
        payload: dict = {
            "stream": stream,
            "method": method,
            "n_steps": int(n_steps),
            "true_drift_t": true_drift_t,
            "seeds": [int(s) for s in seeds],
            "records": records,
        }
        window = get_drift_window(stream)
        if window is not None:
            payload["drift_start"], payload["drift_end"] = (int(window[0]), int(window[1]))
        path = out_dir / f"{stream}.json"
        with path.open("w") as fh:
            json.dump(payload, fh)
        print(f"Wrote {path} ({len(records)} records)")
        written.append(path)
    return written


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--streams",
        nargs="+",
        default=list(DEFAULT_STREAMS),
        help=f"Streams to run. Choose from: {', '.join(STREAM_NAMES)}.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Seeds to run.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=DEFAULT_N_STEPS,
        help="Number of stream samples per run.",
    )
    parser.add_argument(
        "--method",
        default=DEFAULT_METHOD,
        help="UQ method name (default: arf).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_DEFAULT_RESULTS_DIR,
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_streams(
        streams=args.streams,
        seeds=args.seeds,
        n_steps=args.n_steps,
        method=args.method,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
