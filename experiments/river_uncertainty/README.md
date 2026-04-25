# River Streams Experiment (paper §5.3)

Train three UQ methods (`arf`, `deep_ensemble`, `mc_dropout`) on synthetic and
real-world data streams in a prequential test-then-train loop, then plot
per-step epistemic / accuracy trajectories.

The pipeline is two scripts with a parquet between them: `run_experiment.py`
trains and records, `plot_results.py` reads and plots. Decoupling means you
can re-plot or plot a different stream without retraining.

## Quick start

    uv sync -p 3.13
    uv run python scripts/run_experiment.py                  # full: 10 seeds x 9 streams x 3 methods
    uv run python scripts/run_experiment.py --quick          # fast: 2 seeds, 1000 steps
    uv run python scripts/plot_results.py --stream agrawal_drift_7to4
    uv run pytest tests -v                                   # smoke tests

Iterate on one slice without retraining the whole grid:

    uv run python scripts/run_experiment.py \
        --methods arf --streams agrawal_drift_7to4 --n-seeds 3 \
        --output-name arf_only
    uv run python scripts/plot_results.py \
        --records-path results/arf_only.parquet --stream agrawal_drift_7to4

`run_experiment.py` flags: `--streams`, `--methods`, `--seeds` / `--n-seeds`,
`--n-steps`, `--results-dir`, `--output-name`, `--serial`, `--quick`. Run with
`--help` for full details.

## What one step does

In `run_prequential` (test-then-train loop, `src/river_uq/prequential.py`):

1. Pull `(x, y)` from the stream.
2. Predict `y_pred`, then decompose `total / aleatoric / epistemic` via
   probly's `representer` + `quantify`.
3. Two drift detectors observe the step: a label-free `ProblyUQDetector`
   (watches epistemic only) and a method-tailored detector
   (`ARFNativeDetector` for ARF, `PageHinkleyErrorDetector` for the deep
   methods).
4. `model.learn_one(x, y)` updates the model.
5. Append a row: `t, seed, method, stream, y_true, y_pred, correct, total,
   alea, epi, alarm_probly, alarm_tailored, true_drift_t`.

Combos run in parallel via `multiprocessing.Pool` (`--serial` for debugging).

## Streams (`river_uq.streams.STREAM_NAMES`)

All drift streams place an abrupt change at `t = 2000`.

| name | what it does | EU regime |
|---|---|---|
| `stagger_drift` | STAGGER, classification function 0 -> 2 | EU fires |
| `sea_drift` | SEA, variant 0 -> 3 | EU fires |
| `agrawal_stationary` | Agrawal fn 0, no drift | null control |
| `agrawal_drift` | Agrawal fn 0 -> 4 (label flip, same inputs) | EU **misses** -- inputs unchanged |
| `agrawal_drift_7to4` | Agrawal fn 7 -> 4 | EU works -- members disagree post-drift |
| `agrawal_drift_4to0` | Agrawal fn 4 -> 0 | EU fails -- "confidently wrong" regime |
| `agrawal_drift_9to2` | Agrawal fn 9 -> 2 | EU works dramatically (epi 0.0 -> 0.30) |
| `agrawal_covariate_drift` | Agrawal fn 0, salary shifted +80k post-drift | genuine OOD covariate shift |
| `electricity` | real Elec2 data | no labeled drift point |

## Outputs (in `results/`)

- `run_records.parquet` -- one row per (method, stream, seed, t). Source of
  truth for every plot or downstream analysis.
- `<stream>.pdf` -- one figure per `plot_results.py` call. Per panel
  (one per method): epistemic median + IQR band on the left axis, rolling
  accuracy on the right axis, dashed line at `true_drift_t` for drift
  streams.

## Layout

    src/river_uq/
        streams.py         build_stream(name, seed); STREAM_NAMES tuple
        models.py          build_model(kind, seed) -> uniform-interface wrapper
        detectors.py       ProblyUQDetector, ARFNativeDetector, PageHinkleyErrorDetector
        prequential.py     run_prequential(...) -> DataFrame
    scripts/
        run_experiment.py  trains and writes parquet
        plot_results.py    reads parquet, plots one stream
    tests/
        test_smoke.py
