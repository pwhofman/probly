"""Scan all 90 (pre_fn, post_fn) pairs in Agrawal on deep_ensemble.

Used to map which classification-function transitions yield an EU bump
versus which fall into the "confidently wrong" regime where members agree
on the wrong answer. See ``agrawal_transition_zoo.py`` for the curated
showcase that uses three transitions selected from this scan.

Output: ``/tmp/claude/agrawal_function_scan.csv`` with one row per
(pre_fn, post_fn) seed=0 run plus aggregate statistics.
"""

from __future__ import annotations

import itertools
import os
import time

os.environ.setdefault("RIVER_DATA", "/tmp/claude/river_data")

import numpy as np
import pandas as pd
from river.datasets import synth

from river_uq.models import build_model

DRIFT_T = 2000
N = 3000


def stream_pair(pre_fn: int, post_fn: int, seed: int):
    pre = synth.Agrawal(classification_function=pre_fn, seed=seed)
    post = synth.Agrawal(classification_function=post_fn, seed=seed + 1)
    return itertools.chain(
        itertools.islice(iter(pre), DRIFT_T),
        itertools.islice(iter(post), N - DRIFT_T),
    )


def run_one(method: str, pre_fn: int, post_fn: int, seed: int) -> dict:
    model = build_model(method, seed=seed)
    epi_arr = np.zeros(N)
    correct_arr = np.zeros(N, dtype=int)
    for t, (x, y) in enumerate(stream_pair(pre_fn, post_fn, seed)):
        y_pred = model.predict_one(x)
        decomp = model.epistemic_decomposition(x)
        correct_arr[t] = int(int(y_pred) == int(y))
        epi_arr[t] = float(decomp.epistemic)
        model.learn_one(x, y)
    return {
        "method": method,
        "pre_fn": pre_fn,
        "post_fn": post_fn,
        "seed": seed,
        "epi_pre": epi_arr[1500:2000].mean(),
        "epi_imm": epi_arr[2000:2200].mean(),
        "epi_late": epi_arr[2500:3000].mean(),
        "acc_pre": correct_arr[1500:2000].mean(),
        "acc_imm": correct_arr[2000:2200].mean(),
    }


def main() -> None:
    rows = []
    methods = ["deep_ensemble"]
    pairs = [p for p in itertools.product(range(10), repeat=2) if p[0] != p[1]]
    t0 = time.time()
    for method in methods:
        for pre_fn, post_fn in pairs:
            r = run_one(method, pre_fn, post_fn, seed=0)
            r["ratio"] = r["epi_imm"] / max(r["epi_pre"], 1e-9)
            rows.append(r)
            elapsed = time.time() - t0
            print(
                f"  [{elapsed:5.1f}s] {method} {pre_fn}->{post_fn}  "
                f"epi {r['epi_pre']:.3f}->{r['epi_imm']:.3f} ratio={r['ratio']:.2f}  "
                f"acc {r['acc_pre']:.2f}->{r['acc_imm']:.2f}"
            )
    df = pd.DataFrame(rows)
    out_csv = "/tmp/claude/agrawal_function_scan.csv"
    df.to_csv(out_csv, index=False)
    df["epi_bump"] = df["epi_imm"] - df["epi_pre"]
    df["acc_drop"] = df["acc_pre"] - df["acc_imm"]
    print("\n=== top 10 by absolute epi bump ===")
    print(df.sort_values("epi_bump", ascending=False).head(10).to_string(index=False))
    print("\n=== top 10 by accuracy drop ===")
    print(df.sort_values("acc_drop", ascending=False).head(10).to_string(index=False))
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
