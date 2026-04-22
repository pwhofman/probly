"""Level 5 - Uncertainty-based drift detection with deep models.

Recreates Level 3 using the deep ensemble and MC Dropout models from
Level 4.  The same threshold detector (baseline ``mu + k * sigma`` on
a warmup window) is applied to the epistemic entropy signal of each
model.  Detection latency is compared across the three approaches
(deep ensemble, MC Dropout, ARF internal detector from Level 3).

Writes ``results/level5_deep_drift_detector.{png,npz}``.

Usage::

    uv run python experiments/05_deep_drift_detector.py
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from river import forest
from river_uncertainty import (
    RESULTS_DIR as RESULTS,
    DropoutMLP,
    OnlineClassifier,
    deep_ensemble_to_probly_sample,
    detect_drift,
    first_arf_drift_after,
    make_synthetic_stream,
    mc_dropout_to_probly_sample,
    rolling_mean,
    run_prequential,
    run_prequential_generic,
)
import torch

# ---- easy-to-tweak settings ------------------------------------------------
STREAM_KIND = "stagger_drift"
N_SAMPLES = 4_000
SEED = 42
DRIFT_POS = N_SAMPLES // 2

# Model settings
N_MEMBERS = 15
LR = 0.01
HIDDEN_SIZES = (64, 32)
DROPOUT_RATE = 0.2
N_MODELS_ARF = 15

# Detector settings
BASELINE_WINDOW = (500, 1_500)
ROLLING_WINDOW = 30
K_SIGMA = 4.0
MIN_CONSECUTIVE = 5
# ---------------------------------------------------------------------------

# ---- model runners ---------------------------------------------------------


def _run_arf() -> dict[str, np.ndarray]:
    arf = forest.ARFClassifier(n_models=N_MODELS_ARF, seed=SEED)
    stream = make_synthetic_stream(STREAM_KIND, n_samples=N_SAMPLES, seed=SEED)  # type: ignore[arg-type]
    trace = run_prequential(arf, stream, warmup=50, record_every=1)
    return trace.as_arrays()


def _run_ensemble() -> dict[str, np.ndarray]:
    stream_list = list(make_synthetic_stream(STREAM_KIND, n_samples=N_SAMPLES, seed=SEED))
    classifiers = [
        OnlineClassifier(
            module_factory=partial(DropoutMLP, hidden_sizes=HIDDEN_SIZES, dropout_rate=0.0),
            optimizer_fn=torch.optim.Adam,
            lr=LR,
            seed=SEED + i,
        )
        for i in range(N_MEMBERS)
    ]

    def learn_fn(x, y):
        for clf in classifiers:
            clf.learn_one(x, y)

    def predict_fn(x):
        rep = deep_ensemble_to_probly_sample(classifiers, x)
        return rep.sample, rep.classes

    trace = run_prequential_generic(learn_fn, predict_fn, iter(stream_list), warmup=50)
    return trace.as_arrays()


def _run_mc_dropout() -> dict[str, np.ndarray]:
    stream_list = list(make_synthetic_stream(STREAM_KIND, n_samples=N_SAMPLES, seed=SEED))
    clf = OnlineClassifier(
        module_factory=partial(DropoutMLP, hidden_sizes=HIDDEN_SIZES, dropout_rate=DROPOUT_RATE),
        optimizer_fn=torch.optim.Adam,
        lr=LR,
        seed=SEED,
    )

    def learn_fn(x, y):
        clf.learn_one(x, y)

    def predict_fn(x):
        rep = mc_dropout_to_probly_sample(clf, x, n_forward_passes=N_MEMBERS)
        return rep.sample, rep.classes

    trace = run_prequential_generic(learn_fn, predict_fn, iter(stream_list), warmup=50)
    return trace.as_arrays()


# ---- drift detection -------------------------------------------------------


def _detect(data: dict[str, np.ndarray]) -> tuple[int | None, float, float, np.ndarray]:
    return detect_drift(
        data["epistemic_entropy"],
        data["step"],
        rolling_window=ROLLING_WINDOW,
        baseline_window=BASELINE_WINDOW,
        k_sigma=K_SIGMA,
        min_consecutive=MIN_CONSECUTIVE,
    )


# ---- plotting --------------------------------------------------------------

_COLORS: list[str] = ["tab:blue", "tab:green", "tab:purple"]


def _plot(
    results: list[tuple[str, dict[str, np.ndarray], int | None, float, float, np.ndarray]],
    arf_detect_step: int | None,
) -> Path:
    n_methods = len(results)
    fig, axes = plt.subplots(n_methods, 2, figsize=(14, 3.5 * n_methods), sharex=True)

    for row, (name, data, det_step, mu, sigma, smoothed) in enumerate(results):
        steps = data["step"]
        threshold = mu + K_SIGMA * sigma
        color = _COLORS[row]

        # Left column: accuracy + detection lines
        ax = axes[row, 0]
        ax.plot(steps, rolling_mean(data["correct"], 100), color="black")
        ax.set_ylim(0, 1.02)
        ax.set_ylabel(f"{name}\nrolling accuracy (w=100)")
        ax.axvline(DRIFT_POS, color="crimson", ls="--", alpha=0.7, label="true drift")
        if det_step is not None:
            ax.axvline(det_step, color=color, ls="-.", alpha=0.8, label=f"detect (step {det_step})")
        if arf_detect_step is not None and row == 0:
            ax.axvline(arf_detect_step, color="gray", ls=":", alpha=0.6, label=f"ARF internal ({arf_detect_step})")
        ax.legend(loc="lower right", fontsize=8)

        # Right column: epistemic signal + threshold
        ax = axes[row, 1]
        ax.plot(steps, smoothed, color=color, label=f"epistemic (w={ROLLING_WINDOW})")
        ax.axhline(mu, color="gray", ls=":", label="baseline mu")
        ax.axhline(threshold, color="tab:red", ls="--", label=f"mu + {K_SIGMA}*sigma")
        ax.axvspan(BASELINE_WINDOW[0], BASELINE_WINDOW[1], color="gray", alpha=0.12, label="baseline window")
        ax.axvline(DRIFT_POS, color="crimson", ls="--", alpha=0.7)
        if det_step is not None:
            ax.axvline(det_step, color=color, ls="-.", alpha=0.8)
        ax.set_ylabel("epistemic entropy (nats)")
        ax.legend(loc="upper right", fontsize=8)

        if row == n_methods - 1:
            axes[row, 0].set_xlabel("step")
            axes[row, 1].set_xlabel("step")

    fig.suptitle(
        f"Level 5: deep drift detection ({STREAM_KIND}, true drift at {DRIFT_POS})",
        fontsize=13,
    )
    fig.tight_layout()
    out = RESULTS / "level5_deep_drift_detector.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ---- main ------------------------------------------------------------------


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    print(
        f"Running drift-detection comparison on {STREAM_KIND!r} (n_samples={N_SAMPLES}, drift at step {DRIFT_POS}).\n"
    )

    # Run all three models
    print("Running ARF...")
    arf_data = _run_arf()
    print("Running deep ensemble...")
    ens_data = _run_ensemble()
    print("Running MC Dropout...")
    mc_data = _run_mc_dropout()

    # Detect drift on each epistemic signal
    arf_det, arf_mu, arf_sig, arf_sm = _detect(arf_data)
    ens_det, ens_mu, ens_sig, ens_sm = _detect(ens_data)
    mc_det, mc_mu, mc_sig, mc_sm = _detect(mc_data)

    # ARF internal detector (for reference)
    arf_internal = first_arf_drift_after(arf_data["n_drifts_detected"], arf_data["step"], after=DRIFT_POS - 10)

    # Print summary
    def _fmt(step: int | None) -> str:
        if step is None:
            return "not detected"
        return f"step {step}  (latency +{step - DRIFT_POS})"

    print(f"\n{'Method':<18s} {'Baseline mu':>12s} {'sigma':>8s} {'Threshold':>10s} {'Detection':>30s}")
    print("-" * 82)
    print(
        f"{'ARF (epistemic)':<18s} {arf_mu:>12.4f} {arf_sig:>8.4f} {arf_mu + K_SIGMA * arf_sig:>10.4f} {_fmt(arf_det):>30s}"
    )
    print(
        f"{'Deep Ensemble':<18s} {ens_mu:>12.4f} {ens_sig:>8.4f} {ens_mu + K_SIGMA * ens_sig:>10.4f} {_fmt(ens_det):>30s}"
    )
    print(f"{'MC Dropout':<18s} {mc_mu:>12.4f} {mc_sig:>8.4f} {mc_mu + K_SIGMA * mc_sig:>10.4f} {_fmt(mc_det):>30s}")
    print(f"{'ARF internal':<18s} {'':>12s} {'':>8s} {'':>10s} {_fmt(arf_internal):>30s}")

    # Save raw data
    np.savez(
        RESULTS / "level5_deep_drift_detector.npz",
        arf_detect=arf_det if arf_det is not None else -1,
        ens_detect=ens_det if ens_det is not None else -1,
        mc_detect=mc_det if mc_det is not None else -1,
        arf_internal_detect=arf_internal if arf_internal is not None else -1,
        arf_smoothed=arf_sm,
        ens_smoothed=ens_sm,
        mc_smoothed=mc_sm,
        **{f"arf_{k}": v for k, v in arf_data.items()},
        **{f"ens_{k}": v for k, v in ens_data.items()},
        **{f"mc_{k}": v for k, v in mc_data.items()},
    )

    # Plot
    all_results = [
        ("ARF", arf_data, arf_det, arf_mu, arf_sig, arf_sm),
        ("Deep Ensemble", ens_data, ens_det, ens_mu, ens_sig, ens_sm),
        ("MC Dropout", mc_data, mc_det, mc_mu, mc_sig, mc_sm),
    ]
    path = _plot(all_results, arf_internal)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
