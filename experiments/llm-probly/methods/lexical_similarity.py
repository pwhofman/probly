"""Lexical similarity between the sampled answers (BIG).

uncertainty = -mean ROUGE-L F1 over all pairs of the 10 sampled answers.
If the samples use the same words, the model is consistent (low uncertainty).
Text only: no model, no NLI. Follows lm-polygraph's LexicalSimilarity
(estimators/lexical_similarity.py, metric "rougeL"); ROUGE-L is computed here
with the rouge_score tokenisation (lowercase, alphanumeric words) so no extra
package is needed. lm-polygraph is MIT-licensed like probly.
"""

from __future__ import annotations

from itertools import combinations
import re
import time
from typing import TYPE_CHECKING, Any

from methods._cache_baselines import record

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

NAME = "lexical_similarity"
FAMILY = "agreement"
NEEDS_MODEL = False


def tokens(text: str) -> list[str]:
    """Lowercase words of letters and digits, as rouge_score tokenises."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).split()


def lcs_length(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence of two token lists."""
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge_l(a: str, b: str) -> float:
    """ROUGE-L F1 of two texts (0 if either has no words)."""
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    lcs = lcs_length(ta, tb)
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(tb), lcs / len(ta)
    return 2 * precision * recall / (precision + recall)


def score(
    rows: list[dict[str, Any]],
    model: PreTrainedModel | None = None,
    tok: PreTrainedTokenizerBase | None = None,
) -> dict[str, Any]:
    """Uncertainty = minus the mean pairwise ROUGE-L of the samples."""
    start = time.perf_counter()
    values = []
    for r in rows:
        pairs = list(combinations(r["samples"], 2))
        values.append(-sum(rouge_l(a, b) for a, b in pairs) / len(pairs) if pairs else None)
    seconds = (time.perf_counter() - start) / len(rows)
    return record(NAME, rows, values, seconds)
