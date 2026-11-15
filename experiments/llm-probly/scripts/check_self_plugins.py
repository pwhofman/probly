"""Check that the four SELF plugins reproduce the week 1 score files.

Runs every plugin on the old 50-question file (data/qwen0.5b/triviaqa_50.jsonl)
with the same setup as week 1 and compares the uncertainties with
results/qwen0.5b/<old name>.json, question by question.

Week 1 setup (read from the old notebooks/scripts):
- model loaded with load_model() and no dtype (the model's default precision)
- zero-shot and brainstorm ran on CPU (R1/R2 notebooks, no .to(device))
- empirical and verbalized ran on mps (scripts/r3_real_data.py)

Run from experiments/llm-probly:
    uv run python -m scripts.check_self_plugins
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from probly.representer.sampler.huggingface import load_model

from methods import p_true_brainstorm, p_true_empirical, p_true_zeroshot, verbalized_2s

ROOT = Path(__file__).resolve().parents[1]   # experiments/llm-probly
DATA = ROOT / "data" / "qwen0.5b" / "triviaqa_50.jsonl"
OLD = ROOT / "results" / "qwen0.5b"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TOLERANCE = 1e-4   # max allowed difference in uncertainty per question

# (plugin, old result file name, device used in week 1)
CHECKS = [
    (p_true_zeroshot, "p_true_zeroshot_norm", "cpu"),
    (p_true_brainstorm, "p_true_brainstorm", "cpu"),
    (p_true_empirical, "p_true_empirical", "mps"),
    (verbalized_2s, "verbalized_2s", "mps"),
]


def main() -> None:
    rows = [json.loads(line) for line in DATA.read_text().splitlines() if line.strip()]
    print(f"{len(rows)} questions from {DATA.name}")

    model, tok = load_model(MODEL)
    model.eval()
    print("model dtype:", model.dtype)

    all_ok = True
    for plugin, old_name, device in CHECKS:
        if device == "mps" and not torch.backends.mps.is_available():
            print(f"\n{plugin.NAME}: mps not available, skipped")
            all_ok = False
            continue
        model.to(device)
        print(f"\n{plugin.NAME} on {device} ...")

        new = plugin.score(rows, model, tok)
        old = json.loads((OLD / f"{old_name}.json").read_text())

        same_ids = [str(i) for i in new["ids"]] == [str(i) for i in old["ids"]]
        same_failed = new["failed"] == old["failed"]
        diff = np.abs(np.array(new["uncertainty"]) - np.array(old["uncertainty"]))
        ok = same_ids and same_failed and diff.max() <= TOLERANCE
        all_ok = all_ok and ok

        print(f"  ids equal: {same_ids}   failed equal: {same_failed}")
        print(f"  max |difference|: {diff.max():.2e}   questions above tolerance: "
              f"{int((diff > TOLERANCE).sum())}/{len(diff)}")
        print(f"  seconds per question: {new['seconds_per_question']:.2f} "
              f"(week 1: {old['seconds_per_question']:.2f})")
        print("  ->", "MATCH" if ok else "DIFFERENT")

    print("\nALL FOUR MATCH" if all_ok else "\nNOT ALL MATCH, see above")


if __name__ == "__main__":
    main()