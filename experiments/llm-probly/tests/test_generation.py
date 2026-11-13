"""Tests for cache-v2 generation: same numbers as probly's sampler, plus token statistics."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from common.generation import generate_answers, stop_at_newline_config  # noqa: E402
from probly.representer.sampler.huggingface import HFTextGenerationSampler  # noqa: E402

WORDS = ["<pad>", "<eos>", "\n", ".\n", "a", "b", "c", "Paris", "1822", "the", "Question", ":", "user", "assistant"]
PROMPT = "Question : the Paris"


def tiny_model(seed: int) -> tuple:
    """A random 2-layer causal LM with a word-level tokenizer, built offline."""
    vocab = {w: i for i, w in enumerate(WORDS)}
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab=vocab, unk_token="a"))  # noqa: S106
    backend.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    tok = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",  # noqa: S106
        eos_token="<eos>",  # noqa: S106
    )
    tok.padding_side = "left"
    tok.chat_template = (
        "{% for m in messages %}{{ m['role'] }} : {{ m['content'] }} {% endfor %}"
        "{% if add_generation_prompt %}assistant : {% endif %}"
    )
    torch.manual_seed(seed)
    config = transformers.LlamaConfig(
        vocab_size=len(WORDS),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        pad_token_id=0,
        eos_token_id=1,
    )
    model = transformers.LlamaForCausalLM(config).eval()
    model.generation_config = transformers.GenerationConfig(eos_token_id=1, pad_token_id=0, repetition_penalty=1.05)
    return model, tok


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_greedy_matches_probly_sampler(seed: int) -> None:
    model, tok = tiny_model(seed)
    cfg = stop_at_newline_config(model, tok)
    ours = generate_answers(
        model, tok, PROMPT, num_return_sequences=1, do_sample=False, max_new_tokens=8, generation_config=cfg
    )[0]
    kwargs = {"num_samples": 1, "do_sample": False, "max_new_tokens": 8, "generation_config": cfg}
    summed = HFTextGenerationSampler(model, tok, length_normalization=False, **kwargs)([PROMPT]).tensor
    mean = HFTextGenerationSampler(model, tok, length_normalization=True, **kwargs)([PROMPT]).tensor

    assert ours["text"] == str(summed.text[0, 0]).strip()
    assert ours["ll_sum"] == pytest.approx(float(summed.log_likelihood[0, 0]), abs=1e-5)
    assert ours["ll_mean"] == pytest.approx(float(mean.log_likelihood[0, 0]), abs=1e-5)


def test_samples_match_probly_sampler_with_same_seed() -> None:
    model, tok = tiny_model(1)
    cfg = stop_at_newline_config(model, tok)
    torch.manual_seed(7)
    ours = generate_answers(
        model,
        tok,
        PROMPT,
        num_return_sequences=10,
        do_sample=True,
        temperature=0.7,
        max_new_tokens=8,
        generation_config=cfg,
    )
    torch.manual_seed(7)
    sampler = HFTextGenerationSampler(
        model, tok, num_samples=10, temperature=0.7, max_new_tokens=8, generation_config=cfg
    )
    theirs = sampler([PROMPT]).tensor

    assert [a["text"] for a in ours] == [str(t).strip() for t in theirs.text[0]]
    for a, ll in zip(ours, theirs.log_likelihood[0], strict=True):
        assert a["ll_mean"] == pytest.approx(float(ll), abs=1e-5)


def test_token_statistics_are_consistent() -> None:
    model, tok = tiny_model(1)
    cfg = stop_at_newline_config(model, tok)
    answer = generate_answers(
        model, tok, PROMPT, num_return_sequences=1, do_sample=False, max_new_tokens=8, generation_config=cfg
    )[0]

    assert len(answer["token_logprobs"]) == len(answer["token_entropies"]) > 0
    assert sum(answer["token_logprobs"]) == pytest.approx(answer["ll_sum"], abs=1e-5)
    # entropy of a distribution over the vocabulary lies between 0 and log(vocab size)
    assert all(0.0 <= h <= math.log(len(WORDS)) + 1e-5 for h in answer["token_entropies"])


def test_newline_tokens_become_stop_tokens() -> None:
    model, tok = tiny_model(0)
    cfg = stop_at_newline_config(model, tok)
    assert {1, WORDS.index("\n"), WORDS.index(".\n")} <= set(cfg.eos_token_id)
