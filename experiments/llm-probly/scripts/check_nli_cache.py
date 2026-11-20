"""SELF second task: test SAMPLE's NLI cache against probly's clusterer.

For every question, cluster the samples twice:
1. with probly's HFGreedySemanticClusterer (runs the NLI model), and
2. from the saved NLI cache (no model), with three equivalence rules:
   - probly:  no contradiction in either direction and not neutral-neutral (probly's rule)
   - strict:  entailment in both directions
   - lenient: no contradiction in either direction
Then report for how many questions each rule gives exactly the same clusters as probly.

Run from experiments/llm-probly:
    uv run python -m scripts.check_nli_cache                       # 5-question file (plan)
    uv run python -m scripts.check_nli_cache --n 200 --limit 30    # first 30 of the 200 file
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from probly.representer.semantic_clustering.huggingface import HFGreedySemanticClusterer

ROOT = Path(__file__).resolve().parents[1]
ENTAIL, NEUTRAL, CONTRADICT = 0, 1, 2   # order in the cache (labels = entail, neutral, contradict)


def same_meaning(mat, a, b, rule):
    """Are texts a and b equivalent, using the cached NLI labels in both directions?"""
    ab, ba = mat[a, b].argmax(), mat[b, a].argmax()
    if rule == "probly":
        return ab != CONTRADICT and ba != CONTRADICT and not (ab == NEUTRAL and ba == NEUTRAL)
    if rule == "strict":
        return ab == ENTAIL and ba == ENTAIL
    if rule == "lenient":
        return ab != CONTRADICT and ba != CONTRADICT
    raise ValueError(rule)


def clusters_from_cache(mat, indices, rule):
    """Greedy clustering like probly: each text joins the first cluster whose first member matches."""
    clusters = []
    for t in indices:
        for c in clusters:
            if same_meaning(mat, c[0], t, rule):
                c.append(t)
                break
        else:
            clusters.append([t])
    return clusters


def as_partition(clusters):
    """Clusters as a set of sets, so the order and the cluster ids do not matter."""
    return {frozenset(c) for c in clusters}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen0.5b")
    ap.add_argument("--n", type=int, default=5, help="which cache file: triviaqa_<n>")
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions")
    args = ap.parse_args()

    data = ROOT / "data" / args.model / f"triviaqa_{args.n}.jsonl"
    rows = [json.loads(line) for line in data.read_text().splitlines() if line.strip()]
    cache = np.load(data.with_suffix("").with_suffix(".nli.npz") if False else
                    ROOT / "data" / args.model / f"triviaqa_{args.n}.nli.npz")
    if args.limit:
        rows = rows[: args.limit]

    # 1. Is the cache itself consistent with the data file?
    ids_ok = [str(i) for i in cache["ids"][: len(rows)]] == [str(r["id"]) for r in rows]
    texts_ok = all(cache["n_texts"][i] == 1 + len(r["samples"]) for i, r in enumerate(rows))
    print(f"cache {cache['nli_model']}, use_question={cache['use_question']}")
    print(f"ids match: {ids_ok}   texts per question match: {texts_ok}")

    # 2. probly's clusterer (same NLI model as the cache)
    clusterer = HFGreedySemanticClusterer.from_model_name(str(cache["nli_model"]))
    rules = ["probly", "strict", "lenient"]
    matches = {rule: 0 for rule in rules}
    for i, r in enumerate(rows):
        samples = np.asarray(r["samples"], dtype=object)
        ids = clusterer._cluster_row(samples).tolist()   # one semantic id per sample
        probly_clusters = as_partition(
            [[k + 1 for k, s in enumerate(ids) if s == c] for c in sorted(set(ids))]
        )   # +1: in the cache, index 0 is the greedy answer and the samples are 1..K

        mat = cache["nli"][i]
        indices = range(1, len(r["samples"]) + 1)
        line = f"id {r['id']:>4}: probly {len(probly_clusters)} clusters"
        for rule in rules:
            cached = as_partition(clusters_from_cache(mat, indices, rule))
            same = cached == probly_clusters
            matches[rule] += same
            line += f" | {rule}: {len(cached)}{'' if same else ' DIFF'}"
        print(line)

    print(f"\nsame clusters as probly ({len(rows)} questions):")
    for rule in rules:
        print(f"  {rule:<8} {matches[rule]}/{len(rows)}")


if __name__ == "__main__":
    main()
