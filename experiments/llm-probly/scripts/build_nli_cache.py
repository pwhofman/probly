"""Build the NLI cache: pairwise NLI probabilities for every question.

For each question the texts are [greedy, sample_1, ..., sample_K].
nli[i, a, b] = P(entail, neutral, contradict) with premise = text a,
hypothesis = text b (directional: NLI is not symmetric). Diagonal and padding = NaN.
Index 0 is the greedy answer, so the matrix holds both "all samples vs all samples"
and "greedy vs each sample".

Usage:
  uv run python -m scripts.build_nli_cache \
      --cache data/qwen2.5-0.5b/triviaqa_200.jsonl
  # another NLI model (SELF):
  uv run python -m scripts.build_nli_cache \
      --cache data/qwen2.5-0.5b/triviaqa_200.jsonl \
      --nli-model microsoft/deberta-large-mnli \
      --out data/qwen2.5-0.5b/triviaqa_200.nli.deberta-large.npz
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CANON = ["entail", "neutral", "contr"]  # fixed output order, matched by label prefix


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class NLI:
    def __init__(self, name, device=None, max_length=256):
        self.name = name
        self.device = device or pick_device()
        self.max_length = max_length
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(name).to(self.device).eval()
        )
        # Label order differs between models: read it from the config, never hardcode.
        id2label = {int(i): l.lower() for i, l in self.model.config.id2label.items()}
        self.order = []
        for prefix in CANON:
            ids = [i for i, l in id2label.items() if l.startswith(prefix)]
            assert len(ids) == 1, f"cannot map labels {id2label} to {CANON}"
            self.order.append(ids[0])

    @torch.no_grad()
    def probs(self, premises, hypotheses, batch_size=32):
        out = []
        for s in range(0, len(premises), batch_size):
            enc = self.tok(
                premises[s:s + batch_size],
                hypotheses[s:s + batch_size],
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            p = torch.softmax(self.model(**enc).logits, dim=-1)[:, self.order]
            out.append(p.float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 3), dtype=np.float32)


def row_matrix(nli, texts, question=None, batch_size=32):
    """(n, n, 3) matrix, premise = row index, hypothesis = column index."""
    n = len(texts)
    pairs = [(a, b) for a in range(n) for b in range(n) if a != b]
    pre = [texts[a] for a, _ in pairs]
    hyp = [texts[b] for _, b in pairs]
    if question is not None:  # setting "question + answer"
        pre = [f"{question} {t}" for t in pre]
        hyp = [f"{question} {t}" for t in hyp]
    p = nli.probs(pre, hyp, batch_size)
    mat = np.full((n, n, 3), np.nan, dtype=np.float32)
    for (a, b), v in zip(pairs, p):
        mat[a, b] = v
    return mat


def build(cache_path, nli_model, out_path, greedy_field, samples_field,
          question_field, id_field, use_question, batch_size, limit):
    rows = [json.loads(l) for l in open(cache_path) if l.strip()]
    if limit:
        rows = rows[:limit]
    missing = [f for f in (greedy_field, samples_field) if f not in rows[0]]
    if missing:
        raise KeyError(f"fields {missing} not in cache row; keys are {list(rows[0])}")

    nli = NLI(nli_model)
    K = max(len(r[samples_field]) for r in rows)
    N = len(rows)
    big = np.full((N, K + 1, K + 1, 3), np.nan, dtype=np.float32)
    n_texts = np.zeros(N, dtype=np.int16)

    t0 = time.time()
    for i, r in enumerate(rows):
        texts = [r[greedy_field]] + list(r[samples_field])
        n = len(texts)
        n_texts[i] = n
        q = r[question_field] if use_question else None
        big[i, :n, :n] = row_matrix(nli, texts, q, batch_size)
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{N}  {(time.time() - t0) / (i + 1):.2f}s/question")
    sec_per_q = (time.time() - t0) / N

    ids = np.array([r.get(id_field, i) for i, r in enumerate(rows)])
    np.savez_compressed(
        out_path,
        nli=big,                       # (N, K+1, K+1, 3)
        n_texts=n_texts,               # how many texts per question (1 greedy + samples)
        ids=ids,
        labels=np.array(["entail", "neutral", "contradict"]),
        nli_model=np.array(nli_model),
        use_question=np.array(use_question),
        seconds_per_question=np.array(sec_per_q),
    )
    print(f"saved {out_path}  shape={big.shape}  {sec_per_q:.2f}s/question")


# ---- helpers for readers (SELF, SE plugins) --------------------------------
def load_nli_cache(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def clusters_from_matrix(mat, n, strict=True):
    """Sequential clustering like semantic entropy: a text joins the first cluster
    whose first member it matches. strict = entail in both directions (argmax);
    lenient = neither direction is a contradiction. Compare with probly's clusterer."""
    def same(a, b):
        ab, ba = mat[a, b], mat[b, a]
        if strict:
            return ab.argmax() == 0 and ba.argmax() == 0
        return ab.argmax() != 2 and ba.argmax() != 2

    clusters = []
    for t in range(n):
        for c in clusters:
            if same(c[0], t):
                c.append(t)
                break
        else:
            clusters.append([t])
    return clusters


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="data/<model>/<dataset>_<n>.jsonl")
    ap.add_argument("--nli-model", default="microsoft/deberta-base-mnli")
    ap.add_argument("--out", default=None, help="default: <cache>.nli.npz")
    ap.add_argument("--greedy-field", default="greedy_answer")
    ap.add_argument("--samples-field", default="samples")
    ap.add_argument("--question-field", default="question")
    ap.add_argument("--id-field", default="id")
    ap.add_argument("--use-question", action="store_true")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="only first N rows (testing)")
    a = ap.parse_args()
    out = a.out or str(Path(a.cache).with_suffix("")) + ".nli.npz"
    build(a.cache, a.nli_model, out, a.greedy_field, a.samples_field,
          a.question_field, a.id_field, a.use_question, a.batch_size, a.limit)
