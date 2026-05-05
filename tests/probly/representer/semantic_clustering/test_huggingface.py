"""Tests for Hugging Face semantic clustering representers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("transformers")
from transformers import PreTrainedModel, PreTrainedTokenizerBase

pytest.importorskip("torch")
import torch

from probly.representation.distribution.torch_sparse_log_categorical import (
    TorchSparseLogCategoricalDistribution,
    TorchSparseLogCategoricalDistributionSample,
)
from probly.representation.text_generation.torch import (
    TorchTextGeneration,
    TorchTextGenerationSample,
    TorchTextGenerationSampleSample,
)
from probly.representer.semantic_clustering.huggingface import DEFAULT_NLI_MODEL, HFGreedySemanticClusterer


class FakeTokenizer:
    def __init__(self) -> None:
        """Initialize the fake tokenizer."""
        self.pair_calls: list[list[tuple[str, str]]] = []
        self._text_to_id: dict[str, int] = {}

    def id_for_text(self, text: str) -> int:
        if text not in self._text_to_id:
            self._text_to_id[text] = len(self._text_to_id) + 1
        return self._text_to_id[text]

    def __call__(
        self,
        statements: list[str],
        hypotheses: list[str],
        **kwargs: object,
    ) -> dict[str, torch.Tensor]:
        assert kwargs["padding"] is True
        assert kwargs["return_tensors"] == "pt"
        self.pair_calls.append(list(zip(statements, hypotheses, strict=True)))
        pair_ids = [
            (self.id_for_text(premise), self.id_for_text(hypothesis))
            for premise, hypothesis in zip(statements, hypotheses, strict=True)
        ]
        return {
            "input_ids": torch.tensor([[101, premise, 102, 102, hypothesis, 102] for premise, hypothesis in pair_ids]),
            "attention_mask": torch.ones((len(pair_ids), 6), dtype=torch.long),
            "pair_ids": torch.tensor(pair_ids, dtype=torch.long),
        }


class FakeNLIModel:
    device = torch.device("cpu")
    config = SimpleNamespace(label2id={"CONTRADICTION": 0, "NEUTRAL": 1, "ENTAILMENT": 2})

    def __init__(self, labels: dict[tuple[int, int], int] | None = None) -> None:
        """Initialize the fake NLI model."""
        self.labels = labels or {}
        self.calls: list[list[tuple[int, int]]] = []
        self.eval_calls = 0

    def eval(self) -> FakeNLIModel:
        self.eval_calls += 1
        return self

    def __call__(self, **kwargs: object) -> SimpleNamespace:
        pair_ids = kwargs["pair_ids"]
        if not isinstance(pair_ids, torch.Tensor):
            msg = "pair_ids must be a torch tensor."
            raise TypeError(msg)
        pairs = [tuple(pair) for pair in pair_ids.tolist()]
        self.calls.append(pairs)
        logits = torch.zeros((len(pairs), 3), dtype=torch.float32)
        for index, pair in enumerate(pairs):
            logits[index, self.labels.get(pair, 0)] = 10.0
        return SimpleNamespace(logits=logits)


class FakeHFTokenizer(FakeTokenizer, PreTrainedTokenizerBase):
    """Fake tokenizer that satisfies Hugging Face tokenizer isinstance checks."""

    def __init__(self) -> None:
        """Initialize Hugging Face-compatible fake tokenizer internals."""
        object.__setattr__(self, "verbose", False)
        object.__setattr__(self, "_special_tokens_map", {})
        super().__init__()


class FakeHFNLIModel(FakeNLIModel, PreTrainedModel):
    """Fake model that satisfies Hugging Face model isinstance checks."""


def _labels_for(tokenizer: FakeTokenizer, pairs: dict[tuple[str, str], int]) -> dict[tuple[int, int], int]:
    return {
        (tokenizer.id_for_text(premise), tokenizer.id_for_text(hypothesis)): label
        for (premise, hypothesis), label in pairs.items()
    }


def test_greedy_clusterer_clusters_sample_with_batched_nli_calls() -> None:
    tokenizer = FakeTokenizer()
    labels = _labels_for(
        tokenizer,
        {
            ("a", "b"): 2,
            ("b", "a"): 1,
            ("d", "e"): 0,
            ("e", "d"): 0,
            ("d", "f"): 0,
            ("f", "d"): 0,
            ("e", "f"): 2,
            ("f", "e"): 2,
        },
    )
    model = FakeNLIModel(labels)
    generation = TorchTextGeneration(
        text=np.asarray([["a", "b", "c"], ["d", "e", "f"]], dtype=object),
        log_likelihood=torch.tensor([[-0.1, -0.2, -0.3], [-0.4, -0.5, -0.6]]),
    )
    sample = TorchTextGenerationSample(tensor=generation, sample_dim=1)

    clustered = HFGreedySemanticClusterer(model, tokenizer, batch_size=2)(sample)

    assert isinstance(clustered, TorchSparseLogCategoricalDistribution)
    assert torch.equal(clustered.entry_logits, generation.log_likelihood)
    assert torch.equal(clustered.group_ids, torch.tensor([[0, 0, 1], [0, 1, 1]]))
    assert tokenizer.pair_calls == [
        [("a", "b"), ("b", "a")],
        [("a", "c"), ("c", "a")],
        [("d", "e"), ("e", "d")],
        [("d", "f"), ("f", "d")],
        [("e", "f"), ("f", "e")],
    ]
    assert all(len(call) <= 2 for call in tokenizer.pair_calls)
    assert all(len(call) <= 2 for call in model.calls)


def test_greedy_clusterer_preserves_outer_sample_axis_for_nested_samples() -> None:
    tokenizer = FakeTokenizer()
    labels = _labels_for(
        tokenizer,
        {
            ("a", "b"): 2,
            ("b", "a"): 2,
            ("j", "k"): 2,
            ("k", "j"): 2,
        },
    )
    model = FakeNLIModel(labels)
    generation = TorchTextGeneration(
        text=np.asarray(
            [
                [["a", "b", "c"], ["d", "e", "f"]],
                [["g", "h", "i"], ["j", "k", "l"]],
            ],
            dtype=object,
        ),
        log_likelihood=torch.zeros((2, 2, 3)),
    )
    inner_sample = TorchTextGenerationSample(tensor=generation, sample_dim=2)
    outer_sample = TorchTextGenerationSampleSample(
        tensor=inner_sample,
        sample_dim=1,
        weights=torch.tensor([0.25, 0.75]),
    )

    clustered = HFGreedySemanticClusterer(model, tokenizer, batch_size=3)(outer_sample)

    assert isinstance(clustered, TorchSparseLogCategoricalDistributionSample)
    assert clustered.shape == (2, 2)
    assert clustered.sample_dim == 1
    assert torch.equal(clustered.weights, torch.tensor([0.25, 0.75]))
    assert isinstance(clustered.tensor, TorchSparseLogCategoricalDistribution)
    assert torch.equal(
        clustered.tensor.group_ids,
        torch.tensor(
            [
                [[0, 0, 1], [0, 1, 2]],
                [[0, 1, 2], [0, 0, 1]],
            ]
        ),
    )
    assert torch.equal(clustered.tensor.entry_logits, generation.log_likelihood)


def test_raw_text_generation_requires_axis_and_returns_raw_semantic_generation() -> None:
    tokenizer = FakeTokenizer()
    model = FakeNLIModel(_labels_for(tokenizer, {("x", "y"): 1, ("y", "x"): 1}))
    generation = TorchTextGeneration(
        text=np.asarray([["x", "y"]], dtype=object),
        log_likelihood=torch.tensor([[-0.1, -0.2]]),
    )
    clusterer = HFGreedySemanticClusterer(model, tokenizer)

    with pytest.raises(ValueError, match="axis"):
        clusterer(generation)

    clustered = clusterer(generation, axis=1)

    assert isinstance(clustered, TorchSparseLogCategoricalDistribution)
    assert torch.equal(clustered.group_ids, torch.tensor([[0, 1]]))
    assert torch.equal(clustered.entry_logits, generation.log_likelihood)


def test_single_generation_assigns_cluster_zero_without_nli_model_call() -> None:
    tokenizer = FakeTokenizer()
    model = FakeNLIModel()
    generation = TorchTextGeneration(
        text=np.asarray(["only"], dtype=object),
        log_likelihood=torch.tensor([-0.1]),
    )

    clustered = HFGreedySemanticClusterer(model, tokenizer)(generation, axis=0)

    assert torch.equal(clustered.group_ids, torch.tensor([0]))
    assert model.calls == []
    assert tokenizer.pair_calls == []


def test_from_model_name_loads_hf_sequence_classifier(monkeypatch: pytest.MonkeyPatch) -> None:
    transformers = pytest.importorskip("transformers")
    tokenizer = FakeHFTokenizer()
    model = FakeHFNLIModel()
    tokenizer_calls: list[dict[str, object]] = []
    model_calls: list[dict[str, object]] = []

    def tokenizer_from_pretrained(model_name: str, **kwargs: object) -> FakeHFTokenizer:
        tokenizer_calls.append({"model_name": model_name, **kwargs})
        return tokenizer

    def model_from_pretrained(model_name: str, **kwargs: object) -> FakeHFNLIModel:
        model_calls.append({"model_name": model_name, **kwargs})
        return model

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenizer_from_pretrained)
    monkeypatch.setattr(transformers.AutoModelForSequenceClassification, "from_pretrained", model_from_pretrained)

    clusterer = HFGreedySemanticClusterer.from_model_name(
        cache_dir="probly-cache",
        force_download=True,
        model_kwargs={"dtype": "auto"},
        tokenizer_kwargs={"use_fast": True},
        batch_size=7,
        max_length=128,
    )

    assert clusterer.model is model
    assert clusterer.tokenizer is tokenizer
    assert clusterer.batch_size == 7
    assert clusterer.max_length == 128
    assert model.eval_calls == 1
    assert tokenizer_calls == [
        {
            "model_name": DEFAULT_NLI_MODEL,
            "cache_dir": "probly-cache",
            "force_download": True,
            "use_fast": True,
        }
    ]
    assert model_calls == [
        {
            "model_name": DEFAULT_NLI_MODEL,
            "cache_dir": "probly-cache",
            "force_download": True,
            "dtype": "auto",
        }
    ]
