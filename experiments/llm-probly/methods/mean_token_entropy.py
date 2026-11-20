"""Mean token entropy baseline (BIG).

uncertainty = mean entropy of the model's next-token distribution over the
tokens of the greedy answer (cache v2 field greedy_token_entropies).
Same as lm-polygraph's MeanTokenEntropy. Reads the cache only.
Unlike MSP it looks at the whole distribution at every step, not only at the
token that was chosen.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from methods._cache_baselines import record

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

NAME = "mean_token_entropy"
FAMILY = "baseline"
NEEDS_MODEL = False


def score(
    rows: list[dict[str, Any]],
    model: PreTrainedModel | None = None,
    tok: PreTrainedTokenizerBase | None = None,
) -> dict[str, Any]:
    """Uncertainty = mean token entropy of the greedy answer."""
    values = []
    for r in rows:
        ents = r["greedy_token_entropies"]
        values.append(sum(ents) / len(ents) if ents else None)
    return record(NAME, rows, values)
