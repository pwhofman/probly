"""Level 1 - Stream-response survey.

Train an identical ARF on three contrasting streams and watch how the
uncertainty decomposition reacts to each regime:

* ``agrawal``       - stationary, moderate feature noise; shows the
  baseline shape of the decomposition.
* ``stagger_drift`` - a noise-free abrupt switch; drift is a textbook
  spike in epistemic uncertainty.
* ``sea_drift``     - a harder abrupt-ish switch where aleatoric and
  epistemic components both move; ARF has to replace trees.

The "settings I found interesting" are baked in as the defaults so that
``uv run python experiments/01_stream_response.py`` reproduces the figures
in the README. Tweak the ``STREAMS`` tuple or ``N_MODELS`` at the top to
explore further.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from river import forest
from river_uncertainty import (
    RESULTS_DIR as RESULTS,
    make_synthetic_stream,
    rolling_mean,
    run_prequential,
)

# ---- easy-to-tweak settings ------------------------------------------------
N_MODELS = 15
N_SAMPLES = 4_000
SEED = 42
ROLLING_WINDOW = 150
STREAMS: tuple[tuple[str, str, int | None], ...] = (
    # (stream kind, panel title, drift position for the vertical line - None
    #  if the stream is stationary)
    ("agrawal", "Stationary (Agrawal)", None),
    ("stagger_drift", "Abrupt drift (STAGGER 0->2)", N_SAMPLES // 2),
    ("sea_drift", "Harder drift (SEA 0->3)", N_SAMPLES // 2),
)
# ---------------------------------------------------------------------------


def _run(kind: str) -> dict[str, np.ndarray]:
    arf = forest.ARFClassifier(n_models=N_MODELS, seed=SEED)
    stream = make_synthetic_stream(kind, n_samples=N_SAMPLES, seed=SEED)  # type: ignore[arg-type]
    trace = run_prequential(arf, stream, warmup=50, record_every=1)
    return trace.as_arrays()


def _plot(datasets: dict[str, dict[str, np.ndarray]]) -> Path:
    fig, axes = plt.subplots(2, len(STREAMS), figsize=(4.5 * len(STREAMS), 7), sharex=True)
    if len(STREAMS) == 1:
        axes = axes.reshape(2, 1)

    for col, (kind, title, drift_pos) in enumerate(STREAMS):
        data = datasets[kind]
        steps = data["step"]

        ax_acc = axes[0, col]
        ax_acc.plot(steps, rolling_mean(data["correct"], ROLLING_WINDOW), color="black")
        ax_acc.set_ylim(0, 1.02)
        ax_acc.set_title(title)
        if col == 0:
            ax_acc.set_ylabel(f"rolling accuracy (w={ROLLING_WINDOW})")
        if drift_pos is not None:
            ax_acc.axvline(drift_pos, color="crimson", linestyle="--", alpha=0.7)

        ax_unc = axes[1, col]
        ax_unc.plot(steps, rolling_mean(data["total_entropy"], ROLLING_WINDOW), label="total", color="tab:blue")
        ax_unc.plot(
            steps, rolling_mean(data["aleatoric_entropy"], ROLLING_WINDOW), label="aleatoric", color="tab:green"
        )
        ax_unc.plot(
            steps, rolling_mean(data["epistemic_entropy"], ROLLING_WINDOW), label="epistemic", color="tab:orange"
        )
        if drift_pos is not None:
            ax_unc.axvline(drift_pos, color="crimson", linestyle="--", alpha=0.7)
        if col == 0:
            ax_unc.set_ylabel("entropy decomposition (nats)")
        ax_unc.set_xlabel("step")
        if col == len(STREAMS) - 1:
            ax_unc.legend(loc="upper right")

    fig.suptitle(f"Level 1: stream-response survey (ARF n_models={N_MODELS}, seed={SEED})", fontsize=13)
    fig.tight_layout()
    out = RESULTS / "level1_stream_response.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _summarise(datasets: dict[str, dict[str, np.ndarray]]) -> None:
    print(f"{'stream':<18s} {'acc':>6s} {'total':>8s} {'alea':>8s} {'epi':>8s}")
    for kind, _, drift_pos in STREAMS:
        data = datasets[kind]
        steps = data["step"]
        if drift_pos is not None:
            pre = (steps >= drift_pos - 200) & (steps < drift_pos)
            post = (steps >= drift_pos) & (steps < drift_pos + 100)
            print(
                f"{kind + ' (pre)':<18s} "
                f"{data['correct'][pre].mean():>6.3f} "
                f"{data['total_entropy'][pre].mean():>8.3f} "
                f"{data['aleatoric_entropy'][pre].mean():>8.3f} "
                f"{data['epistemic_entropy'][pre].mean():>8.3f}",
            )
            print(
                f"{kind + ' (post)':<18s} "
                f"{data['correct'][post].mean():>6.3f} "
                f"{data['total_entropy'][post].mean():>8.3f} "
                f"{data['aleatoric_entropy'][post].mean():>8.3f} "
                f"{data['epistemic_entropy'][post].mean():>8.3f}",
            )
        else:
            mask = steps >= steps[-1] - 500
            print(
                f"{kind + ' (tail)':<18s} "
                f"{data['correct'][mask].mean():>6.3f} "
                f"{data['total_entropy'][mask].mean():>8.3f} "
                f"{data['aleatoric_entropy'][mask].mean():>8.3f} "
                f"{data['epistemic_entropy'][mask].mean():>8.3f}",
            )


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    datasets = {kind: _run(kind) for kind, _, _ in STREAMS}
    np.savez(
        RESULTS / "level1_stream_response.npz", **{f"{k}_{key}": v for k, d in datasets.items() for key, v in d.items()}
    )
    path = _plot(datasets)
    print(f"Wrote {path}")
    _summarise(datasets)


if __name__ == "__main__":
    main()
