# River Streams Experiment (paper §5.3)

Train UQ methods on a set of synthetic and real-world streams with abrupt
drift, then plot per-step epistemic / accuracy trajectories.

## Quick start

    uv sync -p 3.13
    uv run python scripts/run_experiment.py                  # full: 10 seeds x all 9 streams x 3 methods
    uv run python scripts/run_experiment.py --quick          # fast: 2 seeds, 1000 steps
    uv run python scripts/plot_results.py --stream agrawal_drift_7to4
    uv run pytest tests -v                                   # smoke tests

`run_experiment.py` accepts `--streams`, `--methods`, `--seeds`, `--n-steps`,
`--results-dir`, `--output-name`, `--serial`. Run with `--help` for the full
list. Streams are listed in `river_uq.streams.STREAM_NAMES`.

`plot_results.py` plots one stream per invocation; pass any stream that's
present in the parquet via `--stream`.

## Outputs (in `results/`)

- `run_records.parquet` — tidy per-step DataFrame (one row per (method, stream, seed, t))
- `<stream>.{pdf,png}` — one figure per call to `plot_results.py`

## Layout

    src/river_uq/
        streams.py         build_stream(name, seed); STREAM_NAMES tuple
        models.py          build_model(kind, seed)  -> uniform-interface wrapper
        detectors.py       ProblyUQDetector, ARFNativeDetector, PageHinkleyErrorDetector
        prequential.py     run_prequential(...) -> DataFrame
    scripts/
        run_experiment.py  trains and writes parquet
        plot_results.py    reads parquet, plots one stream
    tests/
        test_smoke.py
