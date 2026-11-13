"""Answer generation with token-level statistics (cache v2).

probly's ``HFTextGenerationSampler`` returns the decoded text and one log
likelihood per answer. For cache v2 we also need, for every generated token,
its log probability and the entropy of the model's next-token distribution
(used by the MSP / perplexity / mean token entropy baselines). This module
runs ``model.generate`` once and returns all of it.

It follows probly's sampler step by step so that the numbers are the same:

* the prompt goes through the tokenizer's chat template as a user message,
* token log probabilities come from ``compute_transition_scores`` with
  ``normalize_logits=True``,
* only tokens before the first stop token are scored,
* the text is the decoded generation with special tokens removed.
"""

from __future__ import annotations

from collections.abc import Iterable
import copy
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizerBase


def stop_at_newline_config(model: PreTrainedModel, tok: PreTrainedTokenizerBase) -> GenerationConfig:
    """Copy the model's generation config and make every newline token a stop token.

    Small models often answer and then keep talking on the next line. Stopping at
    the first line break (as LM-Polygraph does with its newline stop string) keeps
    only the answer.
    """
    config = copy.deepcopy(model.generation_config)
    eos = config.eos_token_id
    eos_ids = set(eos) if isinstance(eos, list) else ({eos} if eos is not None else set())
    vocab = tok.batch_decode([[i] for i in range(len(tok))])
    newline_ids = {i for i, piece in enumerate(vocab) if "\n" in piece}
    config.eos_token_id = sorted(eos_ids | newline_ids)
    return config


def stop_token_ids(config: GenerationConfig, tok: PreTrainedTokenizerBase) -> set[int]:
    """Token ids that end an answer: EOS ids of the config plus the pad id (as in probly)."""
    ids: set[int] = set()
    for value in (config.eos_token_id, config.pad_token_id, tok.pad_token_id):
        if value is None:
            continue
        ids.update(value if isinstance(value, Iterable) else [value])
    return ids


def generate_answers(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    prompt: str,
    *,
    num_return_sequences: int,
    do_sample: bool,
    max_new_tokens: int,
    generation_config: GenerationConfig,
    temperature: float | None = None,
) -> list[dict]:
    """Generate ``num_return_sequences`` answers to one prompt.

    Returns:
        One dict per answer with ``text``, ``token_logprobs``, ``token_entropies``,
        ``ll_sum`` and ``ll_mean``. The token lists only cover the scored tokens
        (those before the first stop token), so ``sum(token_logprobs) == ll_sum``.
        Entropies are in nats and computed from the same (processed) next-token
        distribution as the log probabilities.
    """
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tok([text], return_tensors="pt", padding=True).to(model.device)

    config = copy.copy(generation_config)
    config.do_sample = do_sample
    config.max_new_tokens = max_new_tokens
    config.num_return_sequences = num_return_sequences
    config.return_dict_in_generate = True
    config.output_scores = True
    if temperature is not None:
        config.temperature = temperature
    if tok.pad_token_id is not None:
        config.pad_token_id = tok.pad_token_id

    with torch.inference_mode():
        out = model.generate(**inputs, generation_config=config)
        logprobs = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True).float()
        # scores: one (num_return_sequences, vocab) tensor per generated step
        step_logp = torch.stack([torch.log_softmax(s.float(), dim=-1) for s in out.scores], dim=1)
        entropies = -(step_logp.exp() * step_logp.nan_to_num(neginf=0.0)).sum(dim=-1)

    generated = out.sequences[:, inputs["input_ids"].shape[-1] :]
    stops = torch.tensor(sorted(stop_token_ids(config, tok)), device=generated.device)
    scored = torch.cumsum(torch.isin(generated, stops).long(), dim=-1) == 0
    texts = tok.batch_decode(generated.tolist(), skip_special_tokens=True)

    answers = []
    for i, answer in enumerate(texts):
        n = int(scored[i].sum())
        lp = logprobs[i, :n].tolist()
        ll_sum = float(sum(lp))
        answers.append(
            {
                "text": answer.strip(),
                "token_logprobs": lp,
                "token_entropies": entropies[i, :n].tolist(),
                "ll_sum": ll_sum,
                "ll_mean": ll_sum / n if n > 0 else 0.0,
            },
        )
    return answers
