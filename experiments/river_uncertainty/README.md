# River Streams Experiment (paper §5.3)

Online learning on Agrawal-family streams with [river](https://riverml.xyz)
ARF, decomposed into total / aleatoric / epistemic uncertainty via probly.
Two scripts, with a JSON between them, so plotting is decoupled from training.

## Quick start

```
uv sync -p 3.13
uv run python scripts/run_stream.py     # ~minutes; 3 streams x 3 seeds x 3000 steps
uv run python scripts/plot_stream.py --all
```

This reproduces the paper figures into `results/`. Run `--help` on either
script for the full flag list.

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

Run defaults are the three Agrawal abrupt drifts shown in the paper:

| name | what changes at `t=2000` |
|---|---|
| `agrawal_drift_7to4` | classification fn 7 -> 4 (EU detects: members disagree post-drift) |
| `agrawal_drift_4to0` | classification fn 4 -> 0 (EU misses: confidently wrong regime) |
| `agrawal_drift_9to2` | classification fn 9 -> 2 (EU detects strongly: epi 0.0 -> 0.30) |

Other streams (covariate drift, virtual drift, gradual drift, electricity,
…) are registered in `src/river_uq/streams.py::STREAM_NAMES` and selectable
via `--streams`. They are not run by default.

## Layout

```
src/river_uq/
    streams.py       build_stream(name, seed); STREAM_NAMES
    models.py        build_model(kind, seed); paper uses kind="arf"
    detectors.py     drift detectors (label-free + tailored)
    prequential.py   run_prequential(...) -> DataFrame
scripts/
    run_stream.py    paper runner: writes <stream>.json per stream
    plot_stream.py   paper plotter: JSON -> PDFs
tests/               smoke tests for streams, models, prequential, detectors
results/             JSON inputs + PDF outputs (PDFs are gitignored)
```

## Tests

```
uv run pytest tests -v
```
