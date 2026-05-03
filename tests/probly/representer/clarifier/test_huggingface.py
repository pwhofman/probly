"""Tests for Hugging Face question clarification representers."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
import torch

from probly.representer.clarifier.huggingface import CLARIFICATION_PROMPT, HFQuestionClarifier


class FakeTokenizer:
    """Fake tokenizer for text-generation clarifier tests."""

    eos_token = "<eos>"  # noqa: S105

    def __init__(self) -> None:
        """Initialize the fake tokenizer."""
        self.padding_side = "left"
        self.pad_token = "<pad>"  # noqa: S105
        self.chat_contents: list[str] = []
        self.add_generation_prompt_calls: list[bool] = []
        self.raw_prompts: list[str] = []

    @property
    def pad_token_id(self) -> int:
        """Return the pad token id."""
        return 0

    @property
    def eos_token_id(self) -> int:
        """Return the EOS token id."""
        return 0

    def apply_chat_template(
        self,
        interaction: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        """Render a chat interaction into a prompt string."""
        assert tokenize is False
        content = interaction[-1]["content"]
        self.chat_contents.append(content)
        self.add_generation_prompt_calls.append(add_generation_prompt)
        return content

    def __call__(self, prompts: list[str], *, return_tensors: str, padding: bool) -> dict[str, torch.Tensor]:
        """Tokenize prompts into deterministic token ids."""
        assert return_tensors == "pt"
        assert padding is True
        self.raw_prompts.extend(prompts)
        encoded = [[10 + index for index, _word in enumerate(prompt.split())] for prompt in prompts]
        max_len = max(len(ids) for ids in encoded)
        input_ids = torch.tensor([[0] * (max_len - len(ids)) + ids for ids in encoded], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": input_ids.ne(0).to(dtype=torch.long)}

    def batch_decode(
        self,
        sequences: list[list[int]],
        *,
        skip_special_tokens: bool = True,
        **_kwargs: object,
    ) -> list[str]:
        """Decode generated token ids into deterministic text."""
        return [
            " ".join(f"tok{token}" for token in sequence if not skip_special_tokens or token != 0)
            for sequence in sequences
        ]


@dataclass
class FakeGenerationOutput:
    """Fake transformers generation output."""

    sequences: torch.Tensor
    scores: tuple[torch.Tensor, ...]
    beam_indices: torch.Tensor | None = None


class FakeModel:
    """Fake text-generation model."""

    device = torch.device("cpu")
    generation_config = SimpleNamespace(name="default")
    config = SimpleNamespace(is_encoder_decoder=False)

    def __init__(self) -> None:
        """Initialize the fake model."""
        self.generate_calls: list[dict[str, object]] = []
        self.eval_calls = 0

    def eval(self) -> FakeModel:
        """Record eval calls."""
        self.eval_calls += 1
        return self

    def generate(self, **kwargs: object) -> FakeGenerationOutput:
        """Return deterministic generated token ids."""
        input_ids = kwargs["input_ids"]
        if not isinstance(input_ids, torch.Tensor):
            msg = "input_ids must be a torch tensor."
            raise TypeError(msg)
        self.generate_calls.append(kwargs)
        num_return_sequences = kwargs.get("num_return_sequences", 1)
        if not isinstance(num_return_sequences, int):
            msg = "num_return_sequences must be an int."
            raise TypeError(msg)
        input_ids = input_ids.repeat_interleave(num_return_sequences, dim=0)
        generated = torch.full((input_ids.shape[0], 1), 100, dtype=torch.long)
        sequences = torch.cat((input_ids, generated), dim=1)
        scores = (torch.zeros((input_ids.shape[0], 128)),)
        return FakeGenerationOutput(sequences=sequences, scores=scores)

    def compute_transition_scores(
        self,
        sequences: torch.Tensor,
        scores: tuple[torch.Tensor, ...],
        *,
        beam_indices: torch.Tensor | None = None,
        normalize_logits: bool = True,
    ) -> torch.Tensor:
        """Return deterministic transition scores."""
        assert beam_indices is None
        assert normalize_logits is True
        return torch.full((sequences.shape[0], len(scores)), -0.5)


def test_clarifier_applies_default_prompt_as_chat_interaction() -> None:
    """The default prompt is used before chat templating."""
    tokenizer = FakeTokenizer()
    clarifier = HFQuestionClarifier(model=FakeModel(), tokenizer=tokenizer, num_samples=1)

    sample = clarifier.represent(["What is a bank?"])

    assert sample.shape == (1, 1)
    assert sample.sample_dim == 1
    assert sample.tensor.text.tolist() == [["tok100"]]
    assert tokenizer.chat_contents == [CLARIFICATION_PROMPT.format(question="What is a bank?")]
    assert tokenizer.add_generation_prompt_calls == [True]


def test_clarifier_honors_custom_prompt_without_chat_template() -> None:
    """A custom prompt can be passed during construction."""
    tokenizer = FakeTokenizer()
    clarifier = HFQuestionClarifier(
        model=FakeModel(),
        tokenizer=tokenizer,
        num_samples=2,
        apply_chat_template=False,
        clarification_prompt="Clarify: {question}",
    )

    sample = clarifier.represent(["Why?"])

    assert sample.shape == (1, 2)
    assert tokenizer.raw_prompts == ["Clarify: Why?"]
    assert tokenizer.chat_contents == []


def test_clarifier_rejects_chat_interaction_inputs() -> None:
    """The clarifier accepts question strings, not preformatted chat interactions."""
    clarifier = HFQuestionClarifier(model=FakeModel(), tokenizer=FakeTokenizer(), num_samples=1)

    with pytest.raises(TypeError, match="question strings"):
        clarifier.represent([[{"role": "user", "content": "Why?"}]])


def test_clarifier_from_model_name_loads_model_and_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    """from_model_name forwards loading options and stores the prompt."""
    model = FakeModel()
    tokenizer = FakeTokenizer()
    calls: list[dict[str, object]] = []

    def fake_load_model(
        model_name: str,
        *,
        cache_dir: str | None = None,
        force_download: bool = False,
        model_kwargs: dict[str, object] | None = None,
        tokenizer_kwargs: dict[str, object] | None = None,
    ) -> tuple[FakeModel, FakeTokenizer]:
        calls.append(
            {
                "model_name": model_name,
                "cache_dir": cache_dir,
                "force_download": force_download,
                "model_kwargs": model_kwargs,
                "tokenizer_kwargs": tokenizer_kwargs,
            }
        )
        return model, tokenizer

    monkeypatch.setattr("probly.representer.clarifier.huggingface.load_model", fake_load_model)

    clarifier = HFQuestionClarifier.from_model_name(
        "fake/model",
        num_samples=3,
        cache_dir="probly-cache",
        force_download=True,
        model_kwargs={"dtype": "auto"},
        tokenizer_kwargs={"use_fast": True},
        batch_size=1,
        apply_chat_template=False,
        add_generation_prompt=False,
        strip_inputs=False,
        do_sample=False,
        temperature=0.7,
        max_new_tokens=8,
        top_k=5,
        with_log_likelihood=False,
        length_normalization=False,
        clarification_prompt="Rephrase {question}",
    )

    assert clarifier.model is model
    assert clarifier.tokenizer is tokenizer
    assert model.eval_calls == 1
    assert clarifier.num_samples == 3
    assert clarifier.batch_size == 1
    assert clarifier.apply_chat_template is False
    assert clarifier.add_generation_prompt is False
    assert clarifier.strip_inputs is False
    assert clarifier.do_sample is False
    assert clarifier.temperature == 0.7
    assert clarifier.max_new_tokens == 8
    assert clarifier.top_k == 5
    assert clarifier.with_log_likelihood is False
    assert clarifier.length_normalization is False
    assert clarifier.clarification_prompt == "Rephrase {question}"
    assert calls == [
        {
            "model_name": "fake/model",
            "cache_dir": "probly-cache",
            "force_download": True,
            "model_kwargs": {"dtype": "auto"},
            "tokenizer_kwargs": {"use_fast": True},
        }
    ]
