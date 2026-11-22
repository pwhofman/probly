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

from common.nli_matrix import (  # noqa: F401  (load_nli_cache: re-export)
    QUESTIONS_PER_BATCH,
    compute_matrices,
    load_nli_cache,
    nli_pair_count,
    pick_device,
    resolve_dtype,
    save_nli_cache,
)
import numpy as np

from probly.representer.semantic_clustering.huggingface import HFGreedySemanticClusterer


def build(
    cache: str,
    nli_model: str,
    out: str,
    *,
    use_question: bool,
    deduplicate: bool,
    batch_size: int,
    limit: int,
    dtype: str = "float32",
    questions_per_batch: int = QUESTIONS_PER_BATCH,
) -> None:
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
    device = pick_device()
    clusterer.model.to(device=device, dtype=resolve_dtype(dtype, device))
    print(f"NLI on {device}, {dtype}, {questions_per_batch} questions per batch, batch size {batch_size}")

    matrices = []
    t0 = time.time()
    for start in range(0, len(rows), questions_per_batch):
        chunk = rows[start : start + questions_per_batch]
        matrices += compute_matrices(clusterer, chunk, use_question=use_question, deduplicate=deduplicate)
        print(f"{len(matrices)}/{len(rows)}  {(time.time() - t0) / len(matrices):.2f}s/question")
    sec_per_q = (time.time() - t0) / len(rows)

    ids = [r.get("id", i) for i, r in enumerate(rows)]
    save_nli_cache(
        out,
        matrices,
        ids,
        nli_model=nli_model,
        use_question=use_question,
        seconds_per_question=sec_per_q,
        deduplicate=deduplicate,
    )
    pairs = np.mean([nli_pair_count(row, deduplicate=deduplicate) for row in rows])
    print(f"saved {out}  {len(rows)} questions  {pairs:.0f} NLI pairs/question  {sec_per_q:.2f}s/question")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="data/<model>/<dataset>_<n>.jsonl")
    ap.add_argument("--nli-model", default="microsoft/deberta-base-mnli")
    ap.add_argument("--out", default=None, help="default: <cache>.nli.npz")
    ap.add_argument("--use-question", action="store_true")
    ap.add_argument("--no-dedup", action="store_true", help="also send identical texts to the NLI model (A2 off)")
    ap.add_argument("--dtype", choices=["float32", "float16"], default="float32", help="float16 only on a GPU")
    ap.add_argument(
        "--questions-per-batch", type=int, default=QUESTIONS_PER_BATCH, help="1 = no batching across questions"
    )
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="only first N rows (testing)")
    a = ap.parse_args()
    out = a.out or str(Path(a.cache).with_suffix("")) + ".nli.npz"
    build(
        a.cache,
        a.nli_model,
        out,
        use_question=a.use_question,
        deduplicate=not a.no_dedup,
        batch_size=a.batch_size,
        limit=a.limit,
        dtype=a.dtype,
        questions_per_batch=a.questions_per_batch,
    )
