"""Tests for transformers text generation sampling."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

pytest.importorskip("transformers")
from transformers import PreTrainedModel, PreTrainedTokenizerBase

pytest.importorskip("torch")
import torch

from probly.representation.text_generation.torch import (
    TorchTextGeneration,
    TorchTextGenerationSample,
    TorchTextGenerationSampleSample,
)
from probly.representer.sampler.huggingface import HFTextGenerationSampler, load_model


class FakeTokenizer:
    eos_token = "<eos>"  # noqa: S105

    def __init__(self, padding_side: str = "left", pad_token: str | None = "<pad>") -> None:  # noqa: S107
        """Initialize the fake tokenizer."""
        self.padding_side = padding_side
        self.pad_token = pad_token
        self.chat_calls = 0
        self.add_generation_prompt_calls: list[bool] = []

    @property
    def pad_token_id(self) -> int | None:
        return 0 if self.pad_token is not None else None

    @property
    def eos_token_id(self) -> int:
        return 0

    def apply_chat_template(
        self,
        interaction: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        self.chat_calls += 1
        self.add_generation_prompt_calls.append(add_generation_prompt)
        assert tokenize is False
        return interaction[-1]["content"]

    def __call__(self, prompts: list[str], *, return_tensors: str, padding: bool) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        assert padding is True
        encoded = [[10 + idx for idx, _word in enumerate(prompt.split())] for prompt in prompts]
        max_len = max(len(ids) for ids in encoded)
        padded = []
        for ids in encoded:
            pad = [0] * (max_len - len(ids))
            padded.append([*pad, *ids] if self.padding_side == "left" else [*ids, *pad])
        input_ids = torch.tensor(padded, dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": input_ids.ne(0).to(dtype=torch.long)}

    def batch_decode(
        self,
        sequences: list[list[int]],
        *,
        skip_special_tokens: bool = True,
        **_kwargs: object,
    ) -> list[str]:
        return [
            " ".join(f"tok{token}" for token in sequence if not skip_special_tokens or token != 0)
            for sequence in sequences
        ]


@dataclass
class FakeGenerationOutput:
    sequences: torch.Tensor
    scores: tuple[torch.Tensor, ...]
    beam_indices: torch.Tensor | None = None


class FakeModel:
    device = torch.device("cpu")
    generation_config = SimpleNamespace(name="default")
    config = SimpleNamespace(is_encoder_decoder=False)

    def __init__(self) -> None:
        """Initialize the fake model."""
        self.generate_calls: list[dict[str, object]] = []
        self.transition_score_calls = 0
        self.eval_calls = 0

    def eval(self) -> FakeModel:
        self.eval_calls += 1
        return self

    def generate(self, **kwargs: object) -> FakeGenerationOutput:
        input_ids = kwargs["input_ids"]
        if not isinstance(input_ids, torch.Tensor):
            msg = "input_ids must be a torch tensor."
            raise TypeError(msg)
        call_idx = len(self.generate_calls)
        self.generate_calls.append(kwargs)
        generation_config = kwargs.get("generation_config")
        num_return_sequences = kwargs.get(
            "num_return_sequences",
            getattr(generation_config, "num_return_sequences", 1),
        )
        if not isinstance(num_return_sequences, int):
            msg = "num_return_sequences must be an int."
            raise TypeError(msg)
        input_ids = input_ids.repeat_interleave(num_return_sequences, dim=0)
        batch_size = input_ids.shape[0]
        first = torch.full((batch_size, 1), 100 + call_idx, dtype=torch.long)
        second = torch.arange(batch_size, dtype=torch.long).reshape(batch_size, 1) + 200
        sequences = torch.cat((input_ids, first, second), dim=1)
        scores = (torch.zeros((batch_size, 256)), torch.zeros((batch_size, 256)))
        return FakeGenerationOutput(sequences=sequences, scores=scores)

    def compute_transition_scores(
        self,
        sequences: torch.Tensor,
        scores: tuple[torch.Tensor, ...],
        *,
        beam_indices: torch.Tensor | None = None,
        normalize_logits: bool = True,
    ) -> torch.Tensor:
        self.transition_score_calls += 1
        assert beam_indices is None
        assert normalize_logits is True
        return torch.full((sequences.shape[0], len(scores)), -0.5)


class FakePostEOSScoreModel(FakeModel):
    def generate(self, **kwargs: object) -> FakeGenerationOutput:
        input_ids = kwargs["input_ids"]
        if not isinstance(input_ids, torch.Tensor):
            msg = "input_ids must be a torch tensor."
            raise TypeError(msg)
        self.generate_calls.append(kwargs)
        generated = torch.tensor([[100, 0, 0], [0, 0, 0]], dtype=torch.long)
        sequences = torch.cat((input_ids, generated[: input_ids.shape[0]]), dim=1)
        scores = (torch.zeros((input_ids.shape[0], 256)),) * 3
        return FakeGenerationOutput(sequences=sequences, scores=scores)

    def compute_transition_scores(
        self,
        sequences: torch.Tensor,
        scores: tuple[torch.Tensor, ...],
        *,
        beam_indices: torch.Tensor | None = None,
        normalize_logits: bool = True,
    ) -> torch.Tensor:
        assert beam_indices is None
        assert normalize_logits is True
        assert len(scores) == 3
        return torch.tensor(
            [
                [-0.25, -0.5, -torch.inf],
                [-torch.inf, -torch.inf, -torch.inf],
            ],
            dtype=torch.float32,
        )[: sequences.shape[0]]


class FakeGemmaTokenizer(FakeTokenizer):
    @property
    def eos_token_id(self) -> int:
        return 1


class FakeGemmaStopScoreModel(FakeModel):
    generation_config = SimpleNamespace(eos_token_id=[1, 106, 50], pad_token_id=0)

    def generate(self, **kwargs: object) -> FakeGenerationOutput:
        input_ids = kwargs["input_ids"]
        if not isinstance(input_ids, torch.Tensor):
            msg = "input_ids must be a torch tensor."
            raise TypeError(msg)
        self.generate_calls.append(kwargs)
        generated = torch.tensor([[100, 106, 0], [101, 102, 103]], dtype=torch.long)
        sequences = torch.cat((input_ids, generated[: input_ids.shape[0]]), dim=1)
        scores = (torch.zeros((input_ids.shape[0], 256)),) * 3
        return FakeGenerationOutput(sequences=sequences, scores=scores)

    def compute_transition_scores(
        self,
        sequences: torch.Tensor,
        scores: tuple[torch.Tensor, ...],
        *,
        beam_indices: torch.Tensor | None = None,
        normalize_logits: bool = True,
    ) -> torch.Tensor:
        assert beam_indices is None
        assert normalize_logits is True
        assert len(scores) == 3
        return torch.tensor(
            [
                [-0.2, -0.4, -torch.inf],
                [-0.3, -0.6, -0.9],
            ],
            dtype=torch.float32,
        )[: sequences.shape[0]]


class FakeEncoderDecoderModel(FakeModel):
    config = SimpleNamespace(is_encoder_decoder=True)


class FakeHFTokenizer(FakeTokenizer, PreTrainedTokenizerBase):
    """Fake tokenizer that satisfies Hugging Face tokenizer isinstance checks."""

    def __init__(self, padding_side: str = "left", pad_token: str | None = "<pad>") -> None:  # noqa: S107
        """Initialize Hugging Face-compatible fake tokenizer internals."""
        object.__setattr__(self, "verbose", False)
        object.__setattr__(self, "_special_tokens_map", {"eos_token": FakeTokenizer.eos_token})
        super().__init__(padding_side=padding_side, pad_token=pad_token)


class FakeHFModel(FakeModel, PreTrainedModel):
    """Fake model that satisfies Hugging Face model isinstance checks."""


class FakeHFEncoderDecoderModel(FakeEncoderDecoderModel, PreTrainedModel):
    """Fake encoder-decoder model that satisfies Hugging Face model isinstance checks."""


def test_sampler_chunks_samples_and_returns_text_generation_sample() -> None:
    model = FakeModel()
    tokenizer = FakeTokenizer()
    sampler = HFTextGenerationSampler(
        model=model,
        tokenizer=tokenizer,
        num_samples=3,
        batch_size=2,
        apply_chat_template=False,
        temperature=0.8,
        max_new_tokens=8,
        top_k=20,
    )

    sample = sampler.represent(["first question", "second question"])

    assert sample.sample_dim == 1
    assert sample.shape == (2, 3)
    assert sample.tensor.text.shape == (2, 3)
    assert torch.equal(sample.tensor.log_likelihood, torch.full((2, 3), -0.5))
    assert len(model.generate_calls) == 2
    assert cast("torch.Tensor", model.generate_calls[0]["input_ids"]).shape[0] == 2
    assert cast("torch.Tensor", model.generate_calls[1]["input_ids"]).shape[0] == 2
    assert model.generate_calls[0]["num_return_sequences"] == 2
    assert model.generate_calls[1]["num_return_sequences"] == 1
    assert model.generate_calls[0]["do_sample"] is True
    assert model.generate_calls[0]["temperature"] == 0.8
    assert model.generate_calls[0]["max_new_tokens"] == 8
    assert model.generate_calls[0]["top_k"] == 20
    assert "generation_config" not in model.generate_calls[0]
    assert len(set(sample.tensor.text[0].tolist())) == 3
    assert tokenizer.padding_side == "left"
    assert tokenizer.pad_token == "<pad>"  # noqa: S105


def test_sampler_accepts_string_ndarrays_and_restores_shape() -> None:
    sample = HFTextGenerationSampler(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        num_samples=2,
        apply_chat_template=False,
    ).represent(np.asarray([["first", "second"], ["third", "fourth"]]))

    assert sample.shape == (2, 2, 2)
    assert sample.sample_dim == 2
    assert sample.tensor.text.shape == (2, 2, 2)
    assert sample.tensor.log_likelihood.shape == (2, 2, 2)
    assert sample.tensor.text[0, 0].tolist() == ["tok100 tok200", "tok100 tok201"]


def test_sampler_accepts_object_ndarrays_and_rejects_non_strings() -> None:
    sampler = HFTextGenerationSampler(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        num_samples=1,
        apply_chat_template=False,
    )

    sample = sampler.represent(np.asarray(["first", "second"], dtype=object))

    assert sample.shape == (2, 1)
    assert sample.sample_dim == 1

    with pytest.raises(TypeError, match="only strings"):
        sampler.represent(np.asarray(["first", 1], dtype=object))


def test_sampler_accepts_text_generation_and_restores_shape() -> None:
    generation = TorchTextGeneration(
        text=np.asarray([["first", "second"], ["third", "fourth"]], dtype=object),
        log_likelihood=torch.zeros((2, 2)),
    )

    sample = HFTextGenerationSampler(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        num_samples=2,
        apply_chat_template=False,
    ).represent(generation)

    assert sample.shape == (2, 2, 2)
    assert sample.sample_dim == 2
    assert sample.tensor.text.shape == (2, 2, 2)
    assert sample.tensor.log_likelihood.shape == (2, 2, 2)


def test_sampler_preserves_existing_sample_axis_when_sampling_text_sample() -> None:
    generation = TorchTextGeneration(
        text=np.asarray(
            [
                [["a", "b", "c", "d"], ["e", "f", "g", "h"], ["i", "j", "k", "l"]],
                [["m", "n", "o", "p"], ["q", "r", "s", "t"], ["u", "v", "w", "x"]],
            ],
            dtype=object,
        ),
        log_likelihood=torch.zeros((2, 3, 4)),
    )
    input_sample = TorchTextGenerationSample(tensor=generation, sample_dim=1)

    output_sample = HFTextGenerationSampler(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        num_samples=2,
        apply_chat_template=False,
    ).represent(input_sample)

    assert output_sample.shape == (2, 3, 4, 2)
    assert output_sample.sample_dim == 1
    assert isinstance(output_sample, TorchTextGenerationSampleSample)
    assert isinstance(output_sample.tensor, TorchTextGenerationSample)
    assert output_sample.tensor.shape == (2, 3, 4, 2)
    assert output_sample.tensor.sample_dim == 3
    assert output_sample.tensor.tensor.text.shape == (2, 3, 4, 2)


def test_sampler_preserves_nested_sample_axes_when_sampling_nested_text_sample() -> None:
    generation = TorchTextGeneration(
        text=np.asarray([["a", "b", "c"], ["d", "e", "f"]], dtype=object),
        log_likelihood=torch.zeros((2, 3)),
    )
    inner_sample = TorchTextGenerationSample(tensor=generation, sample_dim=1)
    outer_sample = TorchTextGenerationSampleSample(tensor=inner_sample, sample_dim=0)

    output_sample = HFTextGenerationSampler(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        num_samples=2,
        apply_chat_template=False,
    ).represent(outer_sample)

    assert output_sample.shape == (2, 3, 2)
    assert output_sample.sample_dim == 0
    assert isinstance(output_sample, TorchTextGenerationSampleSample)
    assert isinstance(output_sample.tensor, TorchTextGenerationSampleSample)
    assert output_sample.tensor.sample_dim == 1
    assert isinstance(output_sample.tensor.tensor, TorchTextGenerationSample)
    assert output_sample.tensor.tensor.sample_dim == 2
    assert output_sample.tensor.tensor.tensor.text.shape == (2, 3, 2)


def test_sampler_can_skip_log_likelihood_scoring() -> None:
    model = FakeModel()
    sample = HFTextGenerationSampler(
        model=model,
        tokenizer=FakeTokenizer(),
        num_samples=1,
        apply_chat_template=False,
        with_log_likelihood=False,
    ).represent(["prompt words"])

    assert torch.equal(sample.tensor.log_likelihood, torch.zeros((1, 1)))
    assert model.transition_score_calls == 0
    assert model.generate_calls[0]["output_scores"] is False


def test_sampler_uses_gemma_style_non_eos_mean_log_likelihood() -> None:
    sample = HFTextGenerationSampler(
        model=FakePostEOSScoreModel(),
        tokenizer=FakeTokenizer(),
        num_samples=1,
        apply_chat_template=False,
    ).represent(["first question", "second question"])

    assert sample.tensor.text.tolist() == [["tok100"], [""]]
    assert torch.isfinite(sample.tensor.log_likelihood).all()
    assert torch.equal(sample.tensor.log_likelihood, torch.tensor([[-0.25], [0.0]]))


def test_sampler_can_return_summed_log_likelihood() -> None:
    sample = HFTextGenerationSampler(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        num_samples=1,
        apply_chat_template=False,
        length_normalization=False,
    ).represent(["prompt words"])

    assert torch.equal(sample.tensor.log_likelihood, torch.tensor([[-1.0]]))


def test_sampler_uses_generation_config_stop_ids_for_log_likelihood() -> None:
    sample = HFTextGenerationSampler(
        model=FakeGemmaStopScoreModel(),
        tokenizer=FakeGemmaTokenizer(),
        num_samples=1,
        apply_chat_template=False,
    ).represent(["first question", "second question"])

    assert sample.tensor.text.tolist() == [["tok100 tok106"], ["tok101 tok102 tok103"]]
    assert torch.isfinite(sample.tensor.log_likelihood).all()
    assert torch.allclose(sample.tensor.log_likelihood, torch.tensor([[-0.2], [-0.6]]))


def test_sampler_merges_options_into_explicit_generation_config() -> None:
    model = FakeModel()
    generation_config = SimpleNamespace(name="custom")
    sample = HFTextGenerationSampler(
        model=model,
        tokenizer=FakeTokenizer(),
        num_samples=2,
        apply_chat_template=False,
        temperature=0.9,
        max_new_tokens=4,
        top_k=5,
        generation_config=generation_config,
    ).represent(["prompt words"])

    assert sample.shape == (1, 2)
    call = model.generate_calls[0]
    config = cast("Any", call["generation_config"])
    assert config is not generation_config
    assert config.name == "custom"
    assert config.do_sample is True
    assert config.return_dict_in_generate is True
    assert config.output_scores is True
    assert config.temperature == 0.9
    assert config.max_new_tokens == 4
    assert config.top_k == 5
    assert config.pad_token_id == 0
    assert config.num_return_sequences == 2
    assert "temperature" not in call
    assert "max_new_tokens" not in call
    assert "top_k" not in call
    assert "num_return_sequences" not in call
    assert not hasattr(generation_config, "temperature")


def test_sampler_strip_inputs_controls_decoded_prefix() -> None:
    model = FakeModel()
    tokenizer = FakeTokenizer()
    stripped = HFTextGenerationSampler(
        model=model,
        tokenizer=tokenizer,
        num_samples=1,
        apply_chat_template=False,
        strip_inputs=True,
    ).represent(["prompt words"])

    unstripped = HFTextGenerationSampler(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        num_samples=1,
        apply_chat_template=False,
        strip_inputs=False,
    ).represent(["prompt words"])

    assert stripped.tensor.text[0, 0].startswith("tok100")
    assert "tok10" not in stripped.tensor.text[0, 0].split()
    assert unstripped.tensor.text[0, 0].startswith("tok10 tok11")


def test_sampler_applies_chat_template() -> None:
    tokenizer = FakeTokenizer()
    sampler = HFTextGenerationSampler(model=FakeModel(), tokenizer=tokenizer, num_samples=1)

    sample = sampler.represent([[{"role": "user", "content": "hello there"}]])

    assert sample.shape == (1, 1)
    assert tokenizer.chat_calls == 1
    assert tokenizer.add_generation_prompt_calls == [True]


def test_sampler_wraps_string_inputs_as_chat_messages() -> None:
    tokenizer = FakeTokenizer()
    sampler = HFTextGenerationSampler(model=FakeModel(), tokenizer=tokenizer, num_samples=1)

    sample = sampler.represent(["hello there"])

    assert sample.shape == (1, 1)
    assert tokenizer.chat_calls == 1
    assert tokenizer.add_generation_prompt_calls == [True]


def test_sampler_honors_add_generation_prompt() -> None:
    tokenizer = FakeTokenizer()
    sampler = HFTextGenerationSampler(
        model=FakeModel(),
        tokenizer=tokenizer,
        num_samples=1,
        add_generation_prompt=False,
    )

    sample = sampler.represent([[{"role": "user", "content": "hello there"}]])

    assert sample.shape == (1, 1)
    assert tokenizer.add_generation_prompt_calls == [False]


def test_decoder_only_sampler_rejects_right_padding() -> None:
    sampler = HFTextGenerationSampler(
        model=FakeModel(),
        tokenizer=FakeTokenizer(padding_side="right"),
        num_samples=1,
        apply_chat_template=False,
    )

    with pytest.raises(ValueError, match="left padding"):
        sampler.represent(["prompt words"])


def test_decoder_only_sampler_rejects_missing_pad_token() -> None:
    sampler = HFTextGenerationSampler(
        model=FakeModel(),
        tokenizer=FakeTokenizer(pad_token=None),
        num_samples=1,
        apply_chat_template=False,
    )

    with pytest.raises(ValueError, match="pad token"):
        sampler.represent(["prompt words"])


def test_encoder_decoder_sampler_allows_right_padding_and_does_not_strip_input_width() -> None:
    sample = HFTextGenerationSampler(
        model=FakeEncoderDecoderModel(),
        tokenizer=FakeTokenizer(padding_side="right"),
        num_samples=1,
        apply_chat_template=False,
        strip_inputs=True,
    ).represent(["prompt words"])

    assert sample.tensor.text[0, 0].startswith("tok10 tok11")


def test_deterministic_multi_sample_generation_keeps_repeat_prompt_behavior() -> None:
    model = FakeModel()
    sample = HFTextGenerationSampler(
        model=model,
        tokenizer=FakeTokenizer(),
        num_samples=3,
        batch_size=2,
        apply_chat_template=False,
        do_sample=False,
    ).represent(["first question", "second question"])

    assert sample.shape == (2, 3)
    assert len(model.generate_calls) == 2
    assert cast("torch.Tensor", model.generate_calls[0]["input_ids"]).shape[0] == 4
    assert cast("torch.Tensor", model.generate_calls[1]["input_ids"]).shape[0] == 2
    assert "num_return_sequences" not in model.generate_calls[0]
    assert model.generate_calls[1]["num_return_sequences"] == 1


def test_sampler_from_model_name_loads_model_and_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
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

    monkeypatch.setattr("probly.representer.sampler.huggingface.load_model", fake_load_model)

    sampler = HFTextGenerationSampler.from_model_name(
        "fake/model",
        num_samples=2,
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
        max_new_tokens=4,
        top_k=5,
        with_log_likelihood=False,
        length_normalization=False,
    )

    assert sampler.model is model
    assert sampler.tokenizer is tokenizer
    assert model.eval_calls == 1
    assert sampler.num_samples == 2
    assert sampler.batch_size == 1
    assert sampler.apply_chat_template is False
    assert sampler.add_generation_prompt is False
    assert sampler.strip_inputs is False
    assert sampler.do_sample is False
    assert sampler.temperature == 0.7
    assert sampler.max_new_tokens == 4
    assert sampler.top_k == 5
    assert sampler.with_log_likelihood is False
    assert sampler.length_normalization is False
    assert calls == [
        {
            "model_name": "fake/model",
            "cache_dir": "probly-cache",
            "force_download": True,
            "model_kwargs": {"dtype": "auto"},
            "tokenizer_kwargs": {"use_fast": True},
        }
    ]


def test_load_model_forwards_download_options_and_configures_decoder_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transformers = pytest.importorskip("transformers")
    tokenizer = FakeHFTokenizer(padding_side="right", pad_token=None)
    model = FakeHFModel()
    tokenizer_calls: list[dict[str, object]] = []
    model_calls: list[dict[str, object]] = []

    def tokenizer_from_pretrained(model_name: str, **kwargs: object) -> FakeHFTokenizer:
        tokenizer_calls.append({"model_name": model_name, **kwargs})
        return tokenizer

    def model_from_pretrained(model_name: str, **kwargs: object) -> FakeHFModel:
        model_calls.append({"model_name": model_name, **kwargs})
        return model

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenizer_from_pretrained)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", model_from_pretrained)

    loaded_model, loaded_tokenizer = load_model(
        "fake/model",
        cache_dir="probly-cache",
        force_download=True,
        model_kwargs={"revision": "main"},
        tokenizer_kwargs={"trust_remote_code": True},
    )

    assert loaded_model is model
    assert loaded_tokenizer is tokenizer
    assert tokenizer_calls == [
        {
            "model_name": "fake/model",
            "cache_dir": "probly-cache",
            "force_download": True,
            "trust_remote_code": True,
        }
    ]
    assert model_calls == [
        {
            "model_name": "fake/model",
            "cache_dir": "probly-cache",
            "force_download": True,
            "revision": "main",
        }
    ]
    assert tokenizer.padding_side == "left"
    assert tokenizer.pad_token == tokenizer.eos_token


def test_load_model_leaves_encoder_decoder_tokenizer_padding_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    transformers = pytest.importorskip("transformers")
    tokenizer = FakeHFTokenizer(padding_side="right", pad_token=None)

    def tokenizer_from_pretrained(_model_name: str, **_kwargs: object) -> FakeHFTokenizer:
        return tokenizer

    def model_from_pretrained(_model_name: str, **_kwargs: object) -> FakeHFEncoderDecoderModel:
        return FakeHFEncoderDecoderModel()

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenizer_from_pretrained)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", model_from_pretrained)

    _model, loaded_tokenizer = load_model("fake/model")

    assert loaded_tokenizer is tokenizer
    assert tokenizer.padding_side == "right"
    assert tokenizer.pad_token is None
