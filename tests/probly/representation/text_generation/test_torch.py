"""Tests for torch text generation representations."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from probly.representation.sample import create_sample
from probly.representation.text_generation import (
    TorchTextGeneration,
    TorchTextGenerationSample,
    TorchTokenGeneration,
)


class FakeTokenizer:
    def batch_decode(
        self,
        sequences: list[list[int]],
        *,
        skip_special_tokens: bool = True,
        **_kwargs: object,
    ) -> list[str]:
        return [
            " ".join(str(token) for token in sequence if not skip_special_tokens or token != 0)
            for sequence in sequences
        ]


def test_token_generation_to_text_decodes_and_sums_log_likelihoods() -> None:
    generation = TorchTokenGeneration(
        sequences=torch.tensor([[1, 2, 3], [4, 0, 5]]),
        transition_scores=torch.tensor([[-0.1, -0.2], [-0.3, -0.4]]),
    )

    text = generation.to_text(FakeTokenizer())

    assert isinstance(text, TorchTextGeneration)
    assert text.text.dtype == object
    assert text.text.tolist() == ["1 2 3", "4 5"]
    assert torch.allclose(text.log_likelihood, torch.tensor([-0.3, -0.7]))


def test_text_generation_validates_object_text_shape() -> None:
    with pytest.raises(TypeError, match="dtype=object"):
        TorchTextGeneration(log_likelihood=torch.zeros(2), text=np.array(["a", "b"]))

    with pytest.raises(ValueError, match="identical shapes"):
        TorchTextGeneration(log_likelihood=torch.zeros(2), text=np.asarray([["a", "b"]], dtype=object))


def test_text_generation_index_to_and_detach_preserve_text() -> None:
    generation = TorchTextGeneration(
        log_likelihood=torch.tensor([1.0, 2.0], requires_grad=True),
        text=np.asarray(["a", "b"], dtype=object),
    )

    indexed = generation[1]
    assert isinstance(indexed, TorchTextGeneration)
    assert indexed.text.shape == ()
    assert indexed.text.item() == "b"
    assert torch.equal(indexed.log_likelihood, torch.tensor(2.0))

    converted = generation.to(dtype=torch.float64)
    assert converted.text is generation.text
    assert converted.log_likelihood.dtype == torch.float64

    detached = generation.detach()
    assert detached.text is generation.text
    assert not detached.log_likelihood.requires_grad


def test_create_sample_stacks_text_with_numpy_and_likelihoods_with_torch() -> None:
    left = TorchTextGeneration(
        log_likelihood=torch.tensor([1.0, 2.0]),
        text=np.asarray(["a", "b"], dtype=object),
    )
    right = TorchTextGeneration(
        log_likelihood=torch.tensor([3.0, 4.0]),
        text=np.asarray(["c", "d"], dtype=object),
    )

    sample = create_sample([left, right], sample_axis=1)

    assert isinstance(sample, TorchTextGenerationSample)
    assert sample.sample_dim == 1
    assert sample.shape == (2, 2)
    assert sample.tensor.text.tolist() == [["a", "c"], ["b", "d"]]
    assert torch.equal(sample.tensor.log_likelihood, torch.tensor([[1.0, 3.0], [2.0, 4.0]]))


def test_text_generation_sample_move_and_concat() -> None:
    tensor = TorchTextGeneration(
        log_likelihood=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        text=np.asarray([["a", "b"], ["c", "d"]], dtype=object),
    )
    sample = TorchTextGenerationSample(tensor=tensor, sample_dim=1)

    moved = sample.move_sample_dim(0)
    assert moved.sample_dim == 0
    assert moved.tensor.text.tolist() == [["a", "c"], ["b", "d"]]

    concatenated = sample.concat(sample)
    assert concatenated.sample_dim == 1
    assert concatenated.shape == (2, 4)
    assert concatenated.tensor.text.tolist() == [["a", "b", "a", "b"], ["c", "d", "c", "d"]]
