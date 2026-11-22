"""Maximum sequence probability (MSP) baseline (BIG).

uncertainty = -sum of the token log probabilities of the greedy answer
            = -log P(greedy answer | question).
Same as lm-polygraph's MaximumSequenceProbability. Reads the cache only.
Unlike likelihood_mean it is not length-normalised, so longer answers look
more uncertain.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from methods._cache_baselines import record

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

NAME = "msp"


def score(
    rows: list[dict[str, Any]],
    model: PreTrainedModel | None = None,
    tok: PreTrainedTokenizerBase | None = None,
) -> dict[str, Any]:
    """Uncertainty = minus the log probability of the whole greedy answer."""
    values = [-sum(r["greedy_token_logprobs"]) if r["greedy_token_logprobs"] else None for r in rows]
    return record(NAME, rows, values)
