"""SELF: NLI comparison, deberta-base-mnli (ours) vs deberta-large-mnli (lm-polygraph's default).

For 0.5B and 1.5B (TriviaQA-200), using the two NLI caches and probly's clustering rule:
- how many questions get different clusters,
- AUROC of semantic entropy (SE) and discrete SE with each NLI model,
- NLI seconds per question (from the cache files).

Sanity check first: SE computed from the base cache must equal SAMPLE's SE results
(computed with probly's clusterer), otherwise the comparison is not trustworthy.

Run from experiments/llm-probly:
    uv run python -m scripts.compare_nli
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common.evaluate import bootstrap_auroc, evaluate
from scripts.check_nli_cache import as_partition, clusters_from_cache

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["qwen0.5b", "qwen1.5b"]
NLI = {"base": "triviaqa_200.nli.npz", "large": "triviaqa_200.nli.deberta-large.npz"}


def entropy(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def se_and_dse(clusters, log_likelihood):
    """clusters: lists of cache indices (samples are 1..K); log_likelihood: per sample (index 0 = sample 1)."""
    ll = np.asarray(log_likelihood, dtype=float)
    weights = np.exp(ll - ll.max())                       # likelihood of every sample, up to a constant
    se_p = np.array([weights[[k - 1 for k in c]].sum() for c in clusters])
    dse_p = np.array([len(c) for c in clusters], dtype=float)
    return entropy(se_p / se_p.sum()), entropy(dse_p / dse_p.sum())


def main() -> None:
    for model in MODELS:
        rows = [json.loads(line) for line in (ROOT / "data" / model / "triviaqa_200.jsonl").read_text().splitlines() if line.strip()]
        correct = np.array([bool(r["correct"]) for r in rows])
        print(f"\n===== {model} (accuracy {correct.mean():.3f}) =====")

        partitions, scores = {}, {}
        for name, file in NLI.items():
            cache = np.load(ROOT / "data" / model / file)
            assert [str(i) for i in cache["ids"]] == [str(r["id"]) for r in rows], f"{file}: ids differ"
            parts, se, dse = [], [], []
            for i, r in enumerate(rows):
                clusters = clusters_from_cache(cache["nli"][i], range(1, len(r["samples"]) + 1), "probly")
                parts.append(as_partition(clusters))
                s, d = se_and_dse(clusters, r["samples_ll_mean"])
                se.append(s)
                dse.append(d)
            partitions[name] = parts
            scores[name] = {"semantic_entropy": np.array(se), "discrete_semantic_entropy": np.array(dse)}
            n_clusters = np.mean([len(p) for p in parts])
            print(f"{name:<6} {str(cache['nli_model']):<28} mean clusters {n_clusters:.2f}   "
                  f"NLI {float(cache['seconds_per_question']):.2f} s/question")

        # Sanity check: our SE from the base cache vs SAMPLE's SE (probly's clusterer)
        for method in ["semantic_entropy", "discrete_semantic_entropy"]:
            path = ROOT / "results" / model / "triviaqa" / f"{method}.json"
            if path.exists():
                theirs = np.array(json.loads(path.read_text())["uncertainty"])
                diff = np.abs(theirs - scores["base"][method]).max()
                print(f"check vs SAMPLE's {method}: max |difference| {diff:.2e}  ->  {'OK' if diff < 1e-4 else 'DIFFERENT'}")

        changed = sum(a != b for a, b in zip(partitions["base"], partitions["large"]))
        print(f"questions with different clusters (base vs large): {changed}/{len(rows)}")

        print(f"{'method':<28} {'base AUROC':>20} {'large AUROC':>20}")
        for method in ["semantic_entropy", "discrete_semantic_entropy"]:
            cells = []
            for name in NLI:
                u = scores[name][method]
                lo, hi = bootstrap_auroc(u, correct)
                cells.append(f"{evaluate(u, correct)['auroc']:.3f} [{lo:.2f}-{hi:.2f}]")
            print(f"{method:<28} {cells[0]:>20} {cells[1]:>20}")


if __name__ == "__main__":
    main()


