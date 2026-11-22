"""Tests for the shared NLI matrix (A1): equivalence with the old SE path, file format, sharing."""

from __future__ import annotations

import json
from pathlib import Path

from common import nli_matrix
from common.io import validate_scores
from common.nli_matrix import load_nli_cache, save_nli_cache, se_and_dse
from methods import _se_common, discrete_semantic_entropy, semantic_entropy
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("model", ["qwen0.5b", "qwen1.5b"])
def test_se_from_cached_matrix_equals_old_results(model):
    """Equivalence: SE / DSE from the week-2 NLI cache = the stored week-2 results (all 200 questions)."""
    cache = load_nli_cache(ROOT / f"data/{model}/triviaqa_200.nli.npz")
    rows = [json.loads(line) for line in (ROOT / f"data/{model}/triviaqa_200.jsonl").open()]
    results = ROOT / f"results/{model}/triviaqa_200"
    old_se = json.loads((results / "semantic_entropy.json").read_text())
    old_dse = json.loads((results / "discrete_semantic_entropy.json").read_text())

    assert cache["ids"] == [r["id"] for r in rows] == old_se["ids"]
    new = np.array([se_and_dse(m, r["samples_ll_mean"]) for m, r in zip(cache["matrices"], rows, strict=True)])

    np.testing.assert_allclose(new[:, 0], old_se["uncertainty"], atol=1e-5)
    np.testing.assert_allclose(new[:, 1], old_dse["uncertainty"], atol=1e-5)


def test_load_converts_file_order_to_probly_order(tmp_path):
    entail, _, contradict = np.eye(3, dtype=np.float32)
    file_matrix = np.full((1, 2, 2, 3), np.nan, dtype=np.float32)
    file_matrix[0, 0, 1] = entail
    file_matrix[0, 1, 0] = contradict
    path = tmp_path / "old.nli.npz"
    np.savez(
        path,
        nli=file_matrix,
        n_texts=np.array([2], dtype=np.int16),
        ids=np.array([7]),
        labels=np.array(["entail", "neutral", "contradict"]),
        nli_model=np.array("m"),
        use_question=np.array(False),
        seconds_per_question=np.array(1.0),
    )

    loaded = load_nli_cache(path)

    matrix = loaded["matrices"][0]
    assert matrix[0, 1].argmax().item() == 2  # probly: entailment is last
    assert matrix[1, 0].argmax().item() == 0  # probly: contradiction is first
    assert loaded["ids"] == [7]


def test_save_then_load_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    matrices = []
    for n in (3, 2):
        m = torch.from_numpy(rng.dirichlet(np.ones(3), size=(n, n)).astype(np.float32))
        m[torch.arange(n), torch.arange(n)] = float("nan")
        matrices.append(m)
    path = tmp_path / "x.nli.npz"

    save_nli_cache(path, matrices, [1, 2], nli_model="m", use_question=True, seconds_per_question=0.5)
    loaded = load_nli_cache(path)

    for before, after in zip(matrices, loaded["matrices"], strict=True):
        torch.testing.assert_close(after, before, equal_nan=True)
    assert loaded["use_question"] is True
    with np.load(path) as z:
        assert z["labels"].tolist() == ["entail", "neutral", "contradict"]
        assert int(z["format_version"]) == nli_matrix.FORMAT_VERSION


class CountingClusterer:
    """Stands in for the NLI clusterer: everything entails everything, calls are counted."""

    def __init__(self) -> None:
        """Start with zero calls."""
        self.calls = 0

    def pairwise_nli(self, statements: list[str], context: str | None = None) -> torch.Tensor:
        """Return an all-entailment matrix and count the call."""
        self.calls += 1
        n = len(statements)
        matrix = torch.zeros((n, n, 3))
        matrix[..., 2] = 1.0
        return matrix


def test_se_and_dse_plugins_share_one_nli_pass(monkeypatch):
    fake = CountingClusterer()
    monkeypatch.setattr(_se_common, "_clusterer", fake)
    monkeypatch.setattr(_se_common, "_matrices", {})
    rows = [
        {"id": i, "model": "m", "greedy_answer": "Paris", "samples": ["Paris"] * 3, "samples_ll_mean": [-0.1] * 3}
        for i in range(2)
    ]
    rows[1]["samples"] = ["Paris", "paris", "PARIS"]

    se = semantic_entropy.score(rows)
    dse = discrete_semantic_entropy.score(rows)

    assert fake.calls == 2  # one per question, not one per question and method
    assert se["nli_pairs_per_question"] == 12  # 4 texts -> 4 * 3 ordered pairs per question
    assert dse["nli_pairs_per_question"] == 0
    assert se["uncertainty"] == pytest.approx([0.0, 0.0])
    assert dse["uncertainty"] == pytest.approx([0.0, 0.0])
    assert se["failed"] == dse["failed"] == [False, False]
    assert validate_scores(se, expected_ids=[0, 1]) == []
    assert validate_scores(dse, expected_ids=[0, 1]) == []
