"""Agrawal transition zoo: three classification-function transitions, three methods.

Showcases when EU-based detectors fire and when they don't:

* ``agrawal_drift_9to2`` -- pre-drift function 9 is so simple deep ensembles
  converge to perfect agreement (epi=0); the transition to function 2
  explodes member disagreement.
* ``agrawal_drift_7to4`` -- moderate pre-drift epi; post-drift inputs land
  in regions members disagree about. EU fires.
* ``agrawal_drift_4to0`` -- "confidently wrong" regime: accuracy collapses
  by ~0.44 yet members all track each other into wrongness, so EU stays
  flat (or even drops). Demonstrates that EU measures hypothesis
  *disagreement*, not predictive error.

3 methods x 3 transitions x 3 seeds, 3000 steps each.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import cast

os.environ.setdefault("RIVER_DATA", "/tmp/claude/river_data")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from river_uq.plotting import (  # noqa: E402
    METHOD_LABEL,
    METHOD_ORDER,
    PROBLY_COLOR,
    SMOOTH_WIN,
    TAILORED_COLOR,
)
from river_uq.prequential import run_prequential  # noqa: E402

TRANSITIONS = ["agrawal_drift_9to2", "agrawal_drift_7to4", "agrawal_drift_4to0"]
TRANSITION_LABEL = {
    "agrawal_drift_9to2": "9 -> 2  (EU fires)",
    "agrawal_drift_7to4": "7 -> 4  (EU fires)",
    "agrawal_drift_4to0": "4 -> 0  (confidently wrong)",
}
SEEDS = (0, 1, 2)
N = 3000


def _smooth(s: pd.Series, win: int = SMOOTH_WIN) -> pd.Series:
    return s.rolling(win, min_periods=1).mean()


def main() -> None:
    out_dir = Path("results/exploratory")
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    parts: list[pd.DataFrame] = []
    for method in METHOD_ORDER:
        for stream in TRANSITIONS:
            for seed in SEEDS:
                df = run_prequential(method=method, stream_name=stream, seed=seed, n_steps=N)
                parts.append(df)
    elapsed = time.time() - t0
    records: pd.DataFrame = pd.concat(parts, ignore_index=True)
    records.to_parquet(out_dir / "agrawal_transition_zoo.parquet")
    print(f"ran 27 prequential loops in {elapsed:.1f}s")

    fig, axes = plt.subplots(
        len(METHOD_ORDER), len(TRANSITIONS), figsize=(11.5, 7.5), squeeze=False, sharex=True
    )
    for j, stream in enumerate(TRANSITIONS):
        axes[0, j].set_title(TRANSITION_LABEL[stream], fontsize=11)
    for i, method in enumerate(METHOD_ORDER):
        axes[i, 0].set_ylabel(METHOD_LABEL[method], fontsize=11, fontweight="bold")
    for i, method in enumerate(METHOD_ORDER):
        for j, stream in enumerate(TRANSITIONS):
            ax = axes[i, j]
            sub = records[(records["method"] == method) & (records["stream"] == stream)]
            for _, sg in sub.groupby("seed"):
                ax.plot(sg["t"], _smooth(sg["epi"]), alpha=0.7, lw=1.2, color=PROBLY_COLOR)
            ax.axvline(2000, color="black", ls="--", alpha=0.4, lw=0.8)
            # alarm markers per seed
            ymax = ax.get_ylim()[1]
            for _, sg in sub.groupby("seed"):
                pa = sg.loc[sg["alarm_probly"], "t"]
                ta = sg.loc[sg["alarm_tailored"], "t"]
                if len(pa):
                    ax.scatter([pa.iloc[0]], [ymax * 0.95], marker="v", color=PROBLY_COLOR, s=55, zorder=5)
                if len(ta):
                    ax.scatter([ta.iloc[0]], [ymax * 0.85], marker="^", color=TAILORED_COLOR, s=55, zorder=5)
            ax.set_xlabel("step t" if i == len(METHOD_ORDER) - 1 else "")
            ax.grid(alpha=0.3)
    fig.suptitle(
        "Agrawal classification-function transitions: when does EU fire?\n"
        "v probly-UQ alarm    ^ tailored alarm    -- true drift @ t=2000",
        fontsize=10,
    )
    fig.tight_layout()
    out_pdf = out_dir / "agrawal_transition_zoo.pdf"
    out_png = out_dir / "agrawal_transition_zoo.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"wrote {out_pdf}")

    print("\n=== alarm summary ===")
    rows = []
    for key, g in records.groupby(["method", "stream", "seed"]):
        m, s, seed = cast("tuple[str, str, int]", key)
        pa = g.loc[g["alarm_probly"], "t"]
        ta = g.loc[g["alarm_tailored"], "t"]
        rows.append(
            {
                "method": m,
                "stream": s,
                "seed": seed,
                "probly_first": int(pa.iloc[0]) if len(pa) else None,
                "tailored_first": int(ta.iloc[0]) if len(ta) else None,
                "acc_pre": float(g.loc[(g["t"] >= 1500) & (g["t"] < 2000), "correct"].mean()),
                "acc_post": float(g.loc[(g["t"] >= 2000) & (g["t"] < 2200), "correct"].mean()),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
