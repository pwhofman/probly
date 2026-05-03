# River Streams Experiment (paper §5.3)

Online learning on Agrawal-family streams with [river](https://riverml.xyz)
ARF, decomposed into total / aleatoric / epistemic uncertainty via probly.
Two scripts, with a JSON between them, so plotting is decoupled from training.

## Quick start

```
uv sync -p 3.13
uv run python scripts/run_stream.py     # 12 streams x 3 seeds x 3000 steps
uv run python scripts/plot_stream.py --all
```

The defaults are reviewer-fast (~minutes). Paper-quality figures used 10 seeds
at 8000 steps:

```
uv run python scripts/run_stream.py --seeds 0 1 2 3 4 5 6 7 8 9 --n-steps 8000
uv run python scripts/plot_stream.py --all
```

Run `--help` on either script for the full flag list.

## What it does

For each (stream, seed) pair, `scripts/run_stream.py` runs a prequential
test-then-train loop (`src/river_uq/prequential.py`):

1. Pull `(x, y)` from the stream.
2. Predict, then decompose `total / aleatoric / epistemic` via probly's
   `representer` + `quantify`.
3. `model.learn_one(x, y)` updates the model.
4. Append a row.

It writes one `<stream>.json` per stream into `results/`.

`scripts/plot_stream.py` then turns each JSON into PDFs:

- `<stream>_decomposition.pdf` — total / aleatoric / epistemic over time.
- `<stream>_accuracy.pdf` — epistemic + rolling accuracy on twin axes.
- `<stream>_combined{,_alt}.pdf` — both side-by-side, two layouts.
- plus a `_nolegend` variant of each, and four standalone `legend_*.pdf`.

All plots use `probly.plot.PlotConfig` for fonts and colors.

## Streams

Run defaults cover every Agrawal variant exercised in the paper figures:

- abrupt label drifts: `agrawal_drift`, `agrawal_drift_0to9`,
  `agrawal_drift_4to0`, `agrawal_drift_7to4`, `agrawal_drift_9to2`;
- gradual label drifts: `agrawal_gradual_drift_500`, `agrawal_gradual_drift_1000`;
- covariate / virtual drifts: `agrawal_covariate_drift`,
  `agrawal_virtual_drift_join`, `agrawal_virtual_drift_replace`,
  `agrawal_virtual_drift_stacked`, `agrawal_virtual_drift_stacked_gradual_1000`.

All drifts land at `t=2000` (gradual variants fade in over a window — see
`get_drift_window` in `streams.py`).

Non-Agrawal streams (`stagger_drift`, `sea_drift`, `agrawal_stationary`,
`electricity`) are registered in `src/river_uq/streams.py::STREAM_NAMES` for
exploration via `--streams`, but are not part of the paper grid.

## Layout

```
src/river_uq/
    streams.py       build_stream(name, seed); STREAM_NAMES
    models.py        build_model(kind, seed); only kind="arf" is implemented
    prequential.py   run_prequential(...) -> DataFrame
scripts/
    run_stream.py    paper runner: writes <stream>.json per stream
    plot_stream.py   paper plotter: JSON -> PDFs
tests/               smoke tests for streams, models, prequential
results/             JSON inputs + PDF outputs (PDFs are gitignored)
```

## Tests

```
uv run pytest tests -v
```
