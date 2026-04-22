"""Level 4 - Deep-learning stream-response survey.

Recreates Level 1 using two deep-learning approaches instead of ARF:

* **Deep ensemble** -- N independent online MLPs; per-member
  ``predict_proba_one`` outputs are stacked into a second-order sample.
  Epistemic uncertainty comes from member disagreement.
* **MC Dropout** -- a single online MLP with dropout layers; N stochastic
  forward passes (dropout active) form the sample.  Epistemic uncertainty
  comes from parameter sensitivity via dropout noise.

Both are evaluated on the same three streams as Level 1 (stationary,
abrupt drift, harder drift) using an identical prequential protocol.

Usage::

    uv run python experiments/04_deep_stream_response.py
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from river_uncertainty import (
    RESULTS_DIR as RESULTS,
    DropoutMLP,
    OnlineClassifier,
    deep_ensemble_to_probly_sample,
    make_synthetic_stream,
    mc_dropout_to_probly_sample,
    rolling_mean,
    run_prequential_generic,
)
import torch

# ---- easy-to-tweak settings ------------------------------------------------
N_MEMBERS = 15  # ensemble size / MC forward passes (matches ARF n_models=15)
N_SAMPLES = 4_000
SEED = 42
LR = 0.01
HIDDEN_SIZES = (64, 32)
DROPOUT_RATE = 0.2
ROLLING_WINDOW = 150
STREAMS: tuple[tuple[str, str, int | None], ...] = (
    ("agrawal", "Stationary (Agrawal)", None),
    ("stagger_drift", "Abrupt drift (STAGGER 0->2)", N_SAMPLES // 2),
    ("sea_drift", "Harder drift (SEA 0->3)", N_SAMPLES // 2),
)
# ---------------------------------------------------------------------------


def _make_ensemble() -> list[OnlineClassifier]:
    return [
        OnlineClassifier(
            module_factory=partial(
                DropoutMLP,
                hidden_sizes=HIDDEN_SIZES,
                dropout_rate=0.0,
            ),
            optimizer_fn=torch.optim.Adam,
            lr=LR,
            seed=SEED + i,
        )
        for i in range(N_MEMBERS)
    ]


def _make_mc_dropout() -> OnlineClassifier:
    return OnlineClassifier(
        module_factory=partial(
            DropoutMLP,
            hidden_sizes=HIDDEN_SIZES,
            dropout_rate=DROPOUT_RATE,
        ),
        optimizer_fn=torch.optim.Adam,
        lr=LR,
        seed=SEED,
    )


def _run_ensemble(kind: str) -> dict[str, np.ndarray]:
    stream_list = list(make_synthetic_stream(kind, n_samples=N_SAMPLES, seed=SEED))
    classifiers = _make_ensemble()

    def learn_fn(x: dict[str, float], y: object) -> None:
        for clf in classifiers:
            clf.learn_one(x, y)

    def predict_fn(x: dict[str, float]):
        rep = deep_ensemble_to_probly_sample(classifiers, x)
        return rep.sample, rep.classes

    trace = run_prequential_generic(learn_fn, predict_fn, iter(stream_list), warmup=50)
    return trace.as_arrays()


def _run_mc_dropout(kind: str) -> dict[str, np.ndarray]:
    stream_list = list(make_synthetic_stream(kind, n_samples=N_SAMPLES, seed=SEED))
    clf = _make_mc_dropout()

    def learn_fn(x: dict[str, float], y: object) -> None:
        clf.learn_one(x, y)

    def predict_fn(x: dict[str, float]):
        rep = mc_dropout_to_probly_sample(clf, x, n_forward_passes=N_MEMBERS)
        return rep.sample, rep.classes

    trace = run_prequential_generic(learn_fn, predict_fn, iter(stream_list), warmup=50)
    return trace.as_arrays()


def _plot(
    ensemble_data: dict[str, dict[str, np.ndarray]],
    mc_data: dict[str, dict[str, np.ndarray]],
) -> Path:
    n_streams = len(STREAMS)
    fig, axes = plt.subplots(4, n_streams, figsize=(4.5 * n_streams, 12), sharex=True)
    if n_streams == 1:
        axes = axes.reshape(4, 1)

    methods = [
        ("Deep Ensemble", ensemble_data),
        ("MC Dropout", mc_data),
    ]

    for m_idx, (method_name, datasets) in enumerate(methods):
        row_acc = m_idx * 2
        row_unc = m_idx * 2 + 1

        for col, (kind, title, drift_pos) in enumerate(STREAMS):
            data = datasets[kind]
            steps = data["step"]

            # Accuracy row
            ax_acc = axes[row_acc, col]
            ax_acc.plot(steps, rolling_mean(data["correct"], ROLLING_WINDOW), color="black")
            ax_acc.set_ylim(0, 1.02)
            if m_idx == 0:
                ax_acc.set_title(title)
            if col == 0:
                ax_acc.set_ylabel(f"{method_name}\nrolling accuracy (w={ROLLING_WINDOW})")
            if drift_pos is not None:
                ax_acc.axvline(drift_pos, color="crimson", linestyle="--", alpha=0.7)

            # Uncertainty row
            ax_unc = axes[row_unc, col]
            ax_unc.plot(
                steps,
                rolling_mean(data["total_entropy"], ROLLING_WINDOW),
                label="total",
                color="tab:blue",
            )
            ax_unc.plot(
                steps,
                rolling_mean(data["aleatoric_entropy"], ROLLING_WINDOW),
                label="aleatoric",
                color="tab:green",
            )
            ax_unc.plot(
                steps,
                rolling_mean(data["epistemic_entropy"], ROLLING_WINDOW),
                label="epistemic",
                color="tab:orange",
            )
            if drift_pos is not None:
                ax_unc.axvline(drift_pos, color="crimson", linestyle="--", alpha=0.7)
            if col == 0:
                ax_unc.set_ylabel("entropy decomposition (nats)")
            if m_idx == 1:
                ax_unc.set_xlabel("step")
            if col == n_streams - 1 and m_idx == 0:
                ax_unc.legend(loc="upper right")

    fig.suptitle(
        f"Level 4: deep stream-response survey (N={N_MEMBERS}, lr={LR}, hidden={HIDDEN_SIZES}, seed={SEED})",
        fontsize=13,
    )
    fig.tight_layout()
    out = RESULTS / "level4_deep_stream_response.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _summarise(
    ensemble_data: dict[str, dict[str, np.ndarray]],
    mc_data: dict[str, dict[str, np.ndarray]],
) -> None:
    for method_name, datasets in [("Deep Ensemble", ensemble_data), ("MC Dropout", mc_data)]:
        print(f"\n--- {method_name} ---")
        print(f"{'stream':<22s} {'acc':>6s} {'total':>8s} {'alea':>8s} {'epi':>8s}")
        for kind, _, drift_pos in STREAMS:
            data = datasets[kind]
            steps = data["step"]
            if drift_pos is not None:
                pre = (steps >= drift_pos - 200) & (steps < drift_pos)
                post = (steps >= drift_pos) & (steps < drift_pos + 100)
                for label, mask in [("pre", pre), ("post", post)]:
                    if mask.any():
                        print(
                            f"{kind + f' ({label})':<22s} "
                            f"{data['correct'][mask].mean():>6.3f} "
                            f"{data['total_entropy'][mask].mean():>8.3f} "
                            f"{data['aleatoric_entropy'][mask].mean():>8.3f} "
                            f"{data['epistemic_entropy'][mask].mean():>8.3f}",
                        )
            else:
                mask = steps >= steps[-1] - 500
                print(
                    f"{kind + ' (tail)':<22s} "
                    f"{data['correct'][mask].mean():>6.3f} "
                    f"{data['total_entropy'][mask].mean():>8.3f} "
                    f"{data['aleatoric_entropy'][mask].mean():>8.3f} "
                    f"{data['epistemic_entropy'][mask].mean():>8.3f}",
                )


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    print("Running deep ensemble experiments...")
    ensemble_data = {kind: _run_ensemble(kind) for kind, _, _ in STREAMS}

    print("Running MC Dropout experiments...")
    mc_data = {kind: _run_mc_dropout(kind) for kind, _, _ in STREAMS}

    # Save raw data
    all_arrays: dict[str, np.ndarray] = {}
    for prefix, datasets in [("ens", ensemble_data), ("mc", mc_data)]:
        for kind, data in datasets.items():
            for key, arr in data.items():
                all_arrays[f"{prefix}_{kind}_{key}"] = arr
    np.savez(RESULTS / "level4_deep_stream_response.npz", **all_arrays)

    path = _plot(ensemble_data, mc_data)
    print(f"Wrote {path}")
    _summarise(ensemble_data, mc_data)


if __name__ == "__main__":
    main()
