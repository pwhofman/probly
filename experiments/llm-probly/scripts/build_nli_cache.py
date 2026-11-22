"""Build the NLI cache: pairwise NLI probabilities for every question.

For each question the texts are [greedy, sample_1, ..., sample_K]. The NLI
model is probly's clusterer (``HFSemanticClusterer.pairwise_nli``), the same
code path the semantic-entropy plugins use, so there is only one NLI
implementation. File format: see ``common/nli_matrix.py``.

Usage:
  uv run python -m scripts.build_nli_cache --cache data/qwen0.5b/triviaqa_200.jsonl
  # another NLI model:
  uv run python -m scripts.build_nli_cache --cache data/qwen0.5b/triviaqa_200.jsonl \
      --nli-model microsoft/deberta-large-mnli \
      --out data/qwen0.5b/triviaqa_200.nli.deberta-large.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from common.nli_matrix import compute_matrix, load_nli_cache, save_nli_cache  # noqa: F401  (load_nli_cache: re-export)
import numpy as np

from probly.representer.semantic_clustering.huggingface import HFGreedySemanticClusterer


def pick_device() -> str:
    """Return "cuda", "mps" or "cpu", whichever is available."""
    import torch  # noqa: PLC0415

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build(cache: str, nli_model: str, out: str, *, use_question: bool, batch_size: int, limit: int) -> None:
    """Compute and save the NLI matrices of one data cache."""
    with Path(cache).open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if limit:
        rows = rows[:limit]
    missing = [f for f in ("greedy_answer", "samples") if f not in rows[0]]
    if missing:
        msg = f"fields {missing} not in cache row; keys are {list(rows[0])}"
        raise KeyError(msg)

    clusterer = HFGreedySemanticClusterer.from_model_name(nli_model, batch_size=batch_size, max_length=256)
    clusterer.model.to(pick_device())

    matrices = []
    t0 = time.time()
    for i, row in enumerate(rows):
        matrices.append(compute_matrix(clusterer, row, use_question=use_question))
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(rows)}  {(time.time() - t0) / (i + 1):.2f}s/question")
    sec_per_q = (time.time() - t0) / len(rows)

    ids = [r.get("id", i) for i, r in enumerate(rows)]
    save_nli_cache(
        out,
        matrices,
        ids,
        nli_model=nli_model,
        use_question=use_question,
        seconds_per_question=sec_per_q,
    )
    pairs = np.mean([m.shape[0] * (m.shape[0] - 1) for m in matrices])
    print(f"saved {out}  {len(rows)} questions  {pairs:.0f} NLI pairs/question  {sec_per_q:.2f}s/question")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="data/<model>/<dataset>_<n>.jsonl")
    ap.add_argument("--nli-model", default="microsoft/deberta-base-mnli")
    ap.add_argument("--out", default=None, help="default: <cache>.nli.npz")
    ap.add_argument("--use-question", action="store_true")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="only first N rows (testing)")
    a = ap.parse_args()
    out = a.out or str(Path(a.cache).with_suffix("")) + ".nli.npz"
    build(a.cache, a.nli_model, out, use_question=a.use_question, batch_size=a.batch_size, limit=a.limit)
