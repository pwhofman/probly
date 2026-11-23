"""E: the optimized NLI paths (A1 shared matrix, A2 dedup, A3 cross-question batches) equal the old path.

The fast tests use a rule-based stand-in for the NLI model that goes through probly's real code
(tokenizer and model replaced, everything after the logits unchanged): asymmetric labels,
neutral-neutral pairs and several model batches per call. The last test runs a real NLI model on
real answers and is opt-in (``LLM_PROBLY_REAL_NLI=1``; downloads deberta-base on the first run).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
import re
from typing import TYPE_CHECKING

from common.nli_matrix import compute_matrices, nli_pair_count, question_texts
import pytest
from scripts.check_nli_equivalence import PATHS, SE_ATOL, compare_matrices, run_paths, summarize
import torch

from probly.representer.semantic_clustering.huggingface import HFGreedySemanticClusterer

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONTRADICTION, NEUTRAL, ENTAILMENT = 0, 1, 2


def _norm(text: str) -> str:
    return re.sub(r"[^a-z ]", "", str(text).lower()).replace("the ", "").strip()


def fake_label(premise: str, hypothesis: str) -> int:
    """Rule-based NLI label of one ordered pair.

    Same words -> entailment; premise contains hypothesis -> entailment; hypothesis contains
    premise -> neutral (asymmetric); same first letter -> neutral both ways; else contradiction.
    """
    p, h = _norm(premise), _norm(hypothesis)
    if p == h or (h and h in p):
        return ENTAILMENT
    if p and p in h:
        return NEUTRAL
    if p[:1] == h[:1]:
        return NEUTRAL
    return CONTRADICTION


class RuleClusterer(HFGreedySemanticClusterer):
    """probly's greedy clusterer with the model replaced by :func:`fake_label` (small batches)."""

    def __init__(self, batch_size: int = 5) -> None:
        """Set what probly's code reads; no tokenizer or model."""
        self.batch_size = batch_size
        self.max_length = None
        self.truncation = True
        self._label_id_to_canonical = {0: CONTRADICTION, 1: NEUTRAL, 2: ENTAILMENT}
        self.batches: list[int] = []

    def _iter_pair_logits(self, statements: np.ndarray, pairs: torch.Tensor) -> Iterator[torch.Tensor]:
        for start in range(0, pairs.shape[0], self.batch_size):
            batch = pairs[start : start + self.batch_size]
            self.batches.append(int(batch.shape[0]))
            logits = torch.zeros((batch.shape[0], 3))
            for i, (a, b) in enumerate(batch.tolist()):
                logits[i, fake_label(statements[a], statements[b])] = 4.0
            yield logits


ANSWERS = ["Paris", "paris", "The Paris", "Paris, France", "Lyon", "lyon.", "London", "Lille", "Berlin", "Bern"]


def make_rows(n: int, seed: int = 0) -> list[dict]:
    """Questions with repeated, nested and unrelated answers (10 samples each, like the benchmark)."""
    rng = random.Random(seed)  # noqa: S311  (test data, not crypto)
    rows = []
    for i in range(n):
        pool = rng.sample(ANSWERS, k=rng.randint(1, 5))
        samples = [rng.choice(pool) for _ in range(10)]
        rows.append(
            {
                "id": i,
                "greedy_answer": rng.choice(pool),
                "samples": samples,
                "samples_ll_mean": [rng.uniform(-3.0, -0.01) for _ in samples],
            }
        )
    return rows


def test_fake_model_is_asymmetric_and_has_neutral_pairs() -> None:
    """The stand-in really exercises the hard cases of the greedy rule."""
    assert fake_label("Paris, France", "Paris") == ENTAILMENT
    assert fake_label("Paris", "Paris, France") == NEUTRAL
    assert fake_label("London", "Lille") == fake_label("Lille", "London") == NEUTRAL
    assert fake_label("Paris", "Berlin") == CONTRADICTION


def test_all_paths_give_the_old_clusters_and_se() -> None:
    rows = make_rows(40)
    results = run_paths(RuleClusterer(), rows)
    for row in summarize(results, len(rows)):
        assert row["same_clusters"] == len(rows), row["path"]
        assert row["max_se_diff"] <= SE_ATOL, row["path"]
        assert row["max_dse_diff"] <= SE_ATOL, row["path"]


def test_pair_counts_of_each_path() -> None:
    rows = make_rows(40)
    results = run_paths(RuleClusterer(), rows)
    full = sum(len(question_texts(r)) * (len(question_texts(r)) - 1) for r in rows)
    dedup = sum(nli_pair_count(r) for r in rows)
    assert results["A1"]["pairs"] == full
    assert results["A1+A2"]["pairs"] == results["A1-A3"]["pairs"] == dedup < full


def test_cross_question_batches_are_fuller() -> None:
    """A3: with many small questions, one shared call fills the model batches."""
    rows = make_rows(16)
    one_by_one, shared = RuleClusterer(batch_size=32), RuleClusterer(batch_size=32)
    for row in rows:
        compute_matrices(one_by_one, [row])
    compute_matrices(shared, rows)
    assert sum(one_by_one.batches) == sum(shared.batches) == sum(nli_pair_count(r) for r in rows)
    assert all(size == 32 for size in shared.batches[:-1])  # only the last batch is not full
    assert len(shared.batches) < len(one_by_one.batches)


def test_matrix_paths_are_identical_with_a_deterministic_model() -> None:
    rows = make_rows(20)
    results = run_paths(RuleClusterer(), rows)
    groups = [question_texts(r) for r in rows]
    diff = compare_matrices(results["A1"]["matrices"], results["A1-A3"]["matrices"], groups)
    assert diff["pairs"] > 0
    assert diff["identical_pairs"] > 0
    assert diff["max_prob_diff"] == 0.0
    assert diff["label_flips"] == diff["identical_not_entailment"] == 0


def test_compare_matrices_counts_flips() -> None:
    a = torch.full((2, 2, 3), float("nan"))
    b = a.clone()
    a[0, 1] = torch.tensor([0.1, 0.2, 0.7])
    b[0, 1] = torch.tensor([0.5, 0.2, 0.3])
    a[1, 0] = b[1, 0] = torch.tensor([0.0, 0.0, 1.0])
    diff = compare_matrices([a], [b], [["Paris", "Lyon"]])
    assert diff["max_prob_diff"] == pytest.approx(0.4)
    assert (diff["label_flips"], diff["pairs"], diff["identical_pairs"]) == (1, 2, 0)


def test_compare_matrices_checks_the_dedup_shortcut() -> None:
    """Identical texts: A1 asked the model, A2 wrote entailment; a non-entailment answer is counted."""
    a = torch.full((2, 2, 3), float("nan"))
    b = a.clone()
    a[0, 1] = torch.tensor([0.1, 0.6, 0.3])  # the model says neutral for "Paris" vs "Paris"
    a[1, 0] = torch.tensor([0.0, 0.0, 1.0])
    b[0, 1] = b[1, 0] = torch.tensor([0.0, 0.0, 1.0])
    diff = compare_matrices([a], [b], [["Paris", "Paris"]])
    assert (diff["pairs"], diff["identical_pairs"], diff["identical_not_entailment"]) == (0, 2, 1)


@pytest.mark.skipif(os.environ.get("LLM_PROBLY_REAL_NLI") != "1", reason="set LLM_PROBLY_REAL_NLI=1 (real model)")
def test_real_nli_model_on_real_answers() -> None:
    """Same NLI model, 10 real questions of qwen0.5b: every path gives the old clusters and SE."""
    rows = [json.loads(line) for line in (ROOT / "data/qwen0.5b/triviaqa_200.jsonl").open(encoding="utf-8")][:10]
    clusterer = HFGreedySemanticClusterer.from_model_name(
        os.environ.get("LLM_PROBLY_NLI_MODEL", "microsoft/deberta-base-mnli")
    )
    clusterer.model.eval()
    results = run_paths(clusterer, rows)
    table = {row["path"]: row for row in summarize(results, len(rows))}
    assert set(table) == set(PATHS)
    for row in table.values():
        assert row["same_clusters"] == len(rows), row
        assert row["max_se_diff"] <= SE_ATOL, row
    groups = [question_texts(r) for r in rows]
    diff = compare_matrices(results["A1"]["matrices"], results["A1-A3"]["matrices"], groups)
    assert diff["label_flips"] == diff["identical_not_entailment"] == 0
    assert diff["max_prob_diff"] < 1e-3
