"""Shared code for the semantic-entropy plugins (SE and discrete SE).

Both plugins read the same per-question NLI matrix (A1). The matrix is computed
once per question and kept in memory, so when the runner calls
``semantic_entropy`` and then ``discrete_semantic_entropy`` in the same process,
the NLI model runs only for the first one.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from common.nli_matrix import compute_matrix, nli_pair_count, question_texts, se_and_dse

from probly.representer.semantic_clustering.huggingface import HFGreedySemanticClusterer

if TYPE_CHECKING:
    import torch

NLI_MODEL = "microsoft/deberta-base-mnli"

_clusterer: HFGreedySemanticClusterer | None = None
_matrices: dict[tuple[str, ...], torch.Tensor] = {}
nli_pairs_scored = 0
"""Number of NLI pairs sent to the model in this process (for before / after reports)."""


def get_clusterer() -> HFGreedySemanticClusterer:
    """Load the NLI clusterer once per process."""
    global _clusterer  # noqa: PLW0603
    if _clusterer is None:
        _clusterer = HFGreedySemanticClusterer.from_model_name(NLI_MODEL)
    return _clusterer


def get_matrix(row: dict) -> torch.Tensor:
    """Return the NLI matrix of one question, computing it only the first time."""
    global nli_pairs_scored  # noqa: PLW0603
    key = tuple(question_texts(row))
    if key not in _matrices:
        _matrices[key] = compute_matrix(get_clusterer(), row)
        nli_pairs_scored += nli_pair_count(row)
    return _matrices[key]


def compute_se_and_dse(row: dict) -> tuple[float, float]:
    """Semantic entropy and discrete semantic entropy of one question."""
    return se_and_dse(get_matrix(row), row["samples_ll_mean"])


def score_rows(rows: list[dict], name: str, which: int) -> dict:
    """Score record for SE (``which=0``) or discrete SE (``which=1``), see CONTRACT.md."""
    ids, unc, failed = [], [], []
    pairs_before = nli_pairs_scored
    t0 = time.time()
    for i, r in enumerate(rows):
        ids.append(r.get("id", i))
        try:
            unc.append(compute_se_and_dse(r)[which])
            failed.append(False)
        except Exception:  # noqa: BLE001
            unc.append(1.0)
            failed.append(True)
    return {
        "method": name,
        "model": rows[0].get("model"),
        "answer": "greedy",
        "ids": ids,
        "uncertainty": unc,
        "failed": failed,
        "seconds_per_question": (time.time() - t0) / len(rows),
        "nli_pairs_per_question": (nli_pairs_scored - pairs_before) / len(rows),
    }
