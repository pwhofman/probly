"""Why do empirical p(True) and verbalized differ slightly from week 1 on mps?

1. Run verbalized twice on mps: if greedy decoding differs between two runs,
   mps itself is not deterministic.
2. Show the questions that differ, with the old and new raw replies.
3. Compare AUROC with the old and the new scores: does the difference matter?

Run from experiments/llm-probly:
    uv run python -m scripts.diagnose_mps
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from probly.representer.sampler.huggingface import load_model
from sklearn.metrics import roc_auc_score

from methods import p_true_empirical, verbalized_2s

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "qwen0.5b" / "triviaqa_50.jsonl"
OLD = ROOT / "results" / "qwen0.5b"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def auroc(uncertainty, correct):
    return roc_auc_score(~correct, uncertainty)   # error = positive class


def main() -> None:
    rows = [json.loads(line) for line in DATA.read_text().splitlines() if line.strip()]
    correct = np.array([bool(r["correct"]) for r in rows])

    model, tok = load_model(MODEL)
    model.eval()
    model.to("mps")

    # 1. Is greedy decoding on mps the same in two runs?
    run1 = verbalized_2s.score(rows, model, tok)
    run2 = verbalized_2s.score(rows, model, tok)
    n_diff = sum(a != b for a, b in zip(run1["replies"], run2["replies"]))
    print(f"1. verbalized, two runs on mps: {n_diff}/50 replies differ")

    # 2. Which questions differ from week 1?
    old_v = json.loads((OLD / "verbalized_2s.json").read_text())
    old_v_raw = {str(d["id"]): d["raw"] for d in json.loads((OLD / "verbalized_2s_details.json").read_text())}
    print("\n2. verbalized: questions that differ from week 1")
    for i, (u_new, u_old) in enumerate(zip(run1["uncertainty"], old_v["uncertainty"])):
        if abs(u_new - u_old) > 1e-4:
            qid = str(run1["ids"][i])
            print(f"   id {qid}: old reply {old_v_raw[qid]!r}  new reply {run1['replies'][i]!r}")

    new_e = p_true_empirical.score(rows, model, tok)
    old_e = json.loads((OLD / "p_true_empirical.json").read_text())
    print("\n   empirical: questions that differ from week 1 (p_true old -> new)")
    for i, (u_new, u_old) in enumerate(zip(new_e["uncertainty"], old_e["uncertainty"])):
        if abs(u_new - u_old) > 1e-4:
            print(f"   id {new_e['ids'][i]}: {1 - u_old:.1f} -> {1 - u_new:.1f}")

    # 3. Does it change the result?
    print("\n3. AUROC (week 1 -> now)")
    for name, old, new in [("verbalized_2s", old_v, run1), ("p_true_empirical", old_e, new_e)]:
        a_old = auroc(np.array(old["uncertainty"]), correct)
        a_new = auroc(np.array(new["uncertainty"]), correct)
        print(f"   {name:<18} {a_old:.3f} -> {a_new:.3f}   (difference {a_new - a_old:+.3f})")


if __name__ == "__main__":
    main()