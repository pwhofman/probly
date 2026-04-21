"""Level 2 - Ensemble-size ablation.

How reliable is the epistemic-uncertainty estimate as a function of
``n_models``? With only three trees the "ensemble variance" is dominated by
sampling noise; with 50 trees the estimate is much smoother but training
is slower.

For each ``n_models`` in ``N_MODELS_GRID`` and each seed in ``SEEDS`` we run
a full prequential pass over the STAGGER drift stream (same recipe as
Level 1). We report:

1. Post-warmup, pre-drift **accuracy** - baseline performance.
2. The **epistemic spike** at the drift (mean over a 100-sample window just
   after the drift minus the mean over a 200-sample window just before).
3. The across-seed **standard deviation** of each of the above - a proxy
   for how noisy the estimate is.

Writes ``results/level2_ensemble_size_ablation.{png,csv}``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from river import forest

from river_uncertainty import make_synthetic_stream, run_prequential

# ---- easy-to-tweak settings ------------------------------------------------
N_MODELS_GRID: tuple[int, ...] = (3, 5, 10, 20, 40)
SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
N_SAMPLES = 4_000
STREAM_KIND = "stagger_drift"
# ---------------------------------------------------------------------------

RESULTS = Path(__file__).resolve().parent.parent / "results"
DRIFT_POS = N_SAMPLES // 2
PRE_WINDOW = 200
POST_WINDOW = 100


def _run_single(n_models: int, seed: int) -> dict[str, float]:
    arf = forest.ARFClassifier(n_models=n_models, seed=seed)
    stream = make_synthetic_stream(STREAM_KIND, n_samples=N_SAMPLES, seed=seed)  # type: ignore[arg-type]
    trace = run_prequential(arf, stream, warmup=50, record_every=1)
    data = trace.as_arrays()
    steps = data["step"]
    pre = (steps >= DRIFT_POS - PRE_WINDOW) & (steps < DRIFT_POS)
    post = (steps >= DRIFT_POS) & (steps < DRIFT_POS + POST_WINDOW)
    return {
        "accuracy_pre": float(data["correct"][pre].mean()),
        "accuracy_post": float(data["correct"][post].mean()),
        "epistemic_pre": float(data["epistemic_entropy"][pre].mean()),
        "epistemic_post": float(data["epistemic_entropy"][post].mean()),
        "epistemic_spike": float(data["epistemic_entropy"][post].mean() - data["epistemic_entropy"][pre].mean()),
    }


def _run_grid() -> dict[int, list[dict[str, float]]]:
    grid: dict[int, list[dict[str, float]]] = {n: [] for n in N_MODELS_GRID}
    for n in N_MODELS_GRID:
        for seed in SEEDS:
            grid[n].append(_run_single(n, seed))
            print(f"  n_models={n:3d} seed={seed}: {grid[n][-1]}")
    return grid


def _aggregate(grid: dict[int, list[dict[str, float]]]) -> dict[str, np.ndarray]:
    ns = np.asarray(list(N_MODELS_GRID))
    keys = ("accuracy_pre", "accuracy_post", "epistemic_pre", "epistemic_post", "epistemic_spike")
    out: dict[str, np.ndarray] = {"n_models": ns}
    for k in keys:
        vals = np.asarray([[s[k] for s in grid[n]] for n in N_MODELS_GRID])
        out[f"{k}_mean"] = vals.mean(axis=1)
        out[f"{k}_std"] = vals.std(axis=1, ddof=1) if vals.shape[1] > 1 else np.zeros_like(vals.mean(axis=1))
    return out


def _plot(agg: dict[str, np.ndarray]) -> Path:
    fig, (ax_acc, ax_eps) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax_acc.errorbar(agg["n_models"], agg["accuracy_pre_mean"], yerr=agg["accuracy_pre_std"], marker="o", label="pre-drift")
    ax_acc.errorbar(agg["n_models"], agg["accuracy_post_mean"], yerr=agg["accuracy_post_std"], marker="s", label="post-drift")
    ax_acc.set_xscale("log")
    ax_acc.set_xlabel("n_models")
    ax_acc.set_ylabel("accuracy (window mean)")
    ax_acc.set_title("Prequential accuracy around drift")
    ax_acc.legend()

    ax_eps.errorbar(agg["n_models"], agg["epistemic_pre_mean"], yerr=agg["epistemic_pre_std"], marker="o", label="pre-drift")
    ax_eps.errorbar(agg["n_models"], agg["epistemic_post_mean"], yerr=agg["epistemic_post_std"], marker="s", label="post-drift")
    ax_eps.errorbar(agg["n_models"], agg["epistemic_spike_mean"], yerr=agg["epistemic_spike_std"], marker="^", label="spike (post - pre)")
    ax_eps.set_xscale("log")
    ax_eps.set_xlabel("n_models")
    ax_eps.set_ylabel("epistemic (entropy, nats)")
    ax_eps.set_title("Epistemic uncertainty vs. ensemble size")
    ax_eps.legend()

    fig.suptitle(f"Level 2: ensemble-size ablation ({STREAM_KIND}, {len(SEEDS)} seeds)", fontsize=13)
    fig.tight_layout()
    out = RESULTS / "level2_ensemble_size_ablation.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _dump_csv(agg: dict[str, np.ndarray]) -> Path:
    out = RESULTS / "level2_ensemble_size_ablation.csv"
    keys = [k for k in agg if k != "n_models"]
    with out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["n_models", *keys])
        for i, n in enumerate(agg["n_models"]):
            writer.writerow([int(n), *[f"{agg[k][i]:.6f}" for k in keys]])
    return out


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    print(f"Running ensemble-size ablation on {STREAM_KIND!r}:")
    print(f"  n_models_grid = {N_MODELS_GRID}")
    print(f"  seeds         = {SEEDS}")
    grid = _run_grid()
    agg = _aggregate(grid)
    png = _plot(agg)
    csv_path = _dump_csv(agg)
    print(f"Wrote {png}\nWrote {csv_path}")


if __name__ == "__main__":
    main()
