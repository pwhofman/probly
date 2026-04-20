"""Level 3 - Uncertainty-based drift detection.

A drift detector based purely on the epistemic-uncertainty signal:
estimate a rolling baseline ``mu +/- sigma`` on a warmup window, then flag
a drift the first time the rolling epistemic entropy rises above
``mu + k * sigma`` for ``min_consecutive`` consecutive steps. We compare the
detection latency against ARF's internal ADWIN-based drift detector, which
is exposed through ``arf.n_drifts_detected()``.

The point is **not** to beat ARF's own detector: ARF already sees the
labels, our detector only sees the ensemble state. The point is to check
whether the second-order representation is a meaningful drift signal in
its own right.

Writes ``results/level3_uncertainty_drift_detector.{png,npz}``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from river import forest

from river_uncertainty import make_synthetic_stream, rolling_mean, run_prequential

# ---- easy-to-tweak settings ------------------------------------------------
STREAM_KIND = "stagger_drift"
N_SAMPLES = 4_000
N_MODELS = 15
SEED = 42
DRIFT_POS = N_SAMPLES // 2

# Detector settings
BASELINE_WINDOW = (500, 1_500)  # [start, end) - baseline mu/sigma measured here
ROLLING_WINDOW = 30             # smooth the epistemic signal before thresholding
K_SIGMA = 4.0                   # flag when rolling epistemic exceeds mu + k*sigma
MIN_CONSECUTIVE = 5             # require this many consecutive exceedances
# ---------------------------------------------------------------------------

RESULTS = Path(__file__).resolve().parent.parent / "results"


def _run() -> dict[str, np.ndarray]:
    arf = forest.ARFClassifier(n_models=N_MODELS, seed=SEED)
    stream = make_synthetic_stream(STREAM_KIND, n_samples=N_SAMPLES, seed=SEED)  # type: ignore[arg-type]
    trace = run_prequential(arf, stream, warmup=50, record_every=1)
    return trace.as_arrays()


def _detect_drift(epistemic: np.ndarray, steps: np.ndarray) -> tuple[int | None, float, float, np.ndarray]:
    """Threshold detector on the smoothed epistemic-uncertainty signal.

    Returns:
        detect_step: the first step where the detector fires, or ``None``.
        mu, sigma:   baseline statistics actually used.
        smoothed:    the smoothed epistemic signal (aligned with ``steps``).
    """
    smoothed = rolling_mean(epistemic, ROLLING_WINDOW)
    in_baseline = (steps >= BASELINE_WINDOW[0]) & (steps < BASELINE_WINDOW[1])
    baseline_vals = smoothed[in_baseline]
    mu = float(baseline_vals.mean())
    sigma = float(baseline_vals.std(ddof=1))
    threshold = mu + K_SIGMA * sigma

    # Only start scanning once the baseline window has closed: we are not
    # allowed to "detect" drift during the window from which we learnt mu/sigma.
    scan_start = int(np.searchsorted(steps, BASELINE_WINDOW[1], side="left"))
    above = smoothed > threshold
    detect_step: int | None = None
    streak = 0
    for i in range(scan_start, len(above)):
        if above[i]:
            streak += 1
            if streak >= MIN_CONSECUTIVE:
                detect_step = int(steps[i])
                break
        else:
            streak = 0
    return detect_step, mu, sigma, smoothed


def _first_arf_drift_after(n_drifts: np.ndarray, steps: np.ndarray, after: int) -> int | None:
    """Return the first step where ``n_drifts_detected`` increments past its value at ``after``."""
    pre_idx = np.searchsorted(steps, after, side="right") - 1
    if pre_idx < 0:
        baseline = 0
    else:
        baseline = int(n_drifts[pre_idx])
    mask = (steps >= after) & (n_drifts > baseline)
    if not mask.any():
        return None
    return int(steps[mask][0])


def _plot(data: dict[str, np.ndarray], detect_step: int | None, mu: float, sigma: float, smoothed: np.ndarray) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    steps = data["step"]
    threshold = mu + K_SIGMA * sigma

    ax = axes[0]
    ax.plot(steps, rolling_mean(data["correct"], 100), color="black")
    ax.set_ylabel("rolling accuracy (w=100)")
    ax.set_ylim(0, 1.02)
    ax.axvline(DRIFT_POS, color="crimson", linestyle="--", alpha=0.7, label="true drift")
    if detect_step is not None:
        ax.axvline(detect_step, color="tab:green", linestyle="-.", alpha=0.8, label="uncertainty detector")
    ax.legend(loc="lower right")

    ax = axes[1]
    ax.plot(steps, smoothed, color="tab:orange", label=f"epistemic (w={ROLLING_WINDOW})")
    ax.axhline(mu, color="gray", linestyle=":", label="baseline mu")
    ax.axhline(threshold, color="tab:red", linestyle="--", label=f"mu + {K_SIGMA}*sigma")
    ax.axvspan(BASELINE_WINDOW[0], BASELINE_WINDOW[1], color="gray", alpha=0.15, label="baseline window")
    ax.axvline(DRIFT_POS, color="crimson", linestyle="--", alpha=0.7)
    if detect_step is not None:
        ax.axvline(detect_step, color="tab:green", linestyle="-.", alpha=0.8)
    ax.set_ylabel("epistemic entropy (nats)")
    ax.legend(loc="upper right")

    ax = axes[2]
    ax.plot(steps, data["n_drifts_detected"], color="tab:purple", drawstyle="steps-post", label="ARF internal detections")
    ax.axvline(DRIFT_POS, color="crimson", linestyle="--", alpha=0.7)
    if detect_step is not None:
        ax.axvline(detect_step, color="tab:green", linestyle="-.", alpha=0.8)
    ax.set_ylabel("ARF n_drifts_detected")
    ax.set_xlabel("step")
    ax.legend(loc="upper left")

    fig.suptitle(f"Level 3: uncertainty-based drift detection ({STREAM_KIND})", fontsize=13)
    fig.tight_layout()
    out = RESULTS / "level3_uncertainty_drift_detector.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    print(f"Running drift-detection comparison on {STREAM_KIND!r} "
          f"(n_samples={N_SAMPLES}, drift at step {DRIFT_POS}).")
    data = _run()

    detect_step, mu, sigma, smoothed = _detect_drift(data["epistemic_entropy"], data["step"])
    arf_detect_step = _first_arf_drift_after(data["n_drifts_detected"], data["step"], after=DRIFT_POS - 10)

    print(f"\nBaseline epistemic stats (window {BASELINE_WINDOW}): mu={mu:.4f}, sigma={sigma:.4f}")
    print(f"Threshold mu + {K_SIGMA}*sigma = {mu + K_SIGMA * sigma:.4f}")
    print(f"Uncertainty detector fired at step: {detect_step}  (latency vs truth: "
          f"{(detect_step - DRIFT_POS) if detect_step is not None else 'n/a'})")
    print(f"First ARF drift after truth      : {arf_detect_step}  (latency vs truth: "
          f"{(arf_detect_step - DRIFT_POS) if arf_detect_step is not None else 'n/a'})")

    np.savez(
        RESULTS / "level3_uncertainty_drift_detector.npz",
        smoothed_epistemic=smoothed,
        mu=mu,
        sigma=sigma,
        detect_step=detect_step if detect_step is not None else -1,
        arf_detect_step=arf_detect_step if arf_detect_step is not None else -1,
        **data,
    )

    path = _plot(data, detect_step, mu, sigma, smoothed)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
