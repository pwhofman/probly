# River Streams Experiment (paper §5.3)

Reproduces the headline figure and latency table for the
"Using probly on Incremental Datastreams" subsection.

## Quick start

    uv sync -p 3.13
    uv run python scripts/run_paper_experiment.py            # full: 10 seeds × 3 models × 2 streams
    uv run python scripts/run_paper_experiment.py --quick    # fast: 2 seeds × 1000 steps
    uv run python scripts/run_listing_demo.py                # the ~5-line snippet from the paper
    uv run pytest tests -v                                   # smoke tests

## Outputs (in `results/`)

- `run_records.parquet` — tidy per-step DataFrame (source of truth)
- `headline_figure.{pdf,png}` — 3×2 grid (methods × streams)
- `appendix_stationary.pdf` — Agrawal null
- `latency_table.{csv,tex}` — detection latency table
- `manifest.json` — git SHA, seeds, hyperparams, runtime

## Layout

    src/river_uq/
        streams.py         build_stream(name, seed) -> (iter, true_drift_t)
        models.py          build_model(kind, seed)  -> uniform-interface wrapper
        detectors.py       ProblyUQDetector, ARFNativeDetector, PageHinkleyErrorDetector
        prequential.py     run_prequential(...) -> DataFrame
        plotting.py        build_headline_figure(df), build_appendix_figure(df)
        tables.py          build_latency_table(df) -> DataFrame
    scripts/
        run_paper_experiment.py
        run_listing_demo.py
    tests/
        test_smoke.py
