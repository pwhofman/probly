"""Torch-backed text generation representations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, ClassVar, Self, cast, override

import numpy as np
import torch

from probly.representation._protected_axis.torch import TorchAxisProtected
from probly.representation.sample import Sample, create_sample
from probly.representation.sample.torch import TorchSample

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from transformers import PreTrainedTokenizerBase

    from probly.representation.array_like import ToIndices


@dataclass(frozen=True, slots=True, weakref_slot=True)
class TorchTextGeneration(TorchAxisProtected[torch.Tensor]):
    """Decoded text generations with sequence log-likelihoods.

    Shape: ``batch_shape``. The text is stored as a NumPy object array because
    PyTorch does not support string tensors, while log-likelihoods remain torch tensors.
    """

    log_likelihood: torch.Tensor
    text: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"log_likelihood": 0, "text": 0}

    def __post_init__(self) -> None:
        """Validate text and log-likelihood fields."""
        if not isinstance(self.log_likelihood, torch.Tensor):
            msg = "log_likelihood must be a torch tensor."
            raise TypeError(msg)
        if not isinstance(self.text, np.ndarray):
            msg = "text must be a numpy ndarray."
            raise TypeError(msg)
        if self.text.dtype != object:
            msg = "text must be a numpy ndarray with dtype=object."
            raise TypeError(msg)
        if self.text.shape != tuple(self.log_likelihood.shape):
            msg = "text and log_likelihood must have identical shapes."
            raise ValueError(msg)

    @override
    def __getitem__(self, index: ToIndices, /) -> Self:
        text = self.text[index]
        if not isinstance(text, np.ndarray):
            text = np.asarray(text, dtype=object)

        log_likelihood = self.log_likelihood[index]  # ty:ignore[invalid-argument-type]
        if not isinstance(log_likelihood, torch.Tensor):
            log_likelihood = torch.as_tensor(log_likelihood, device=self.log_likelihood.device)

        return type(self)(log_likelihood=log_likelihood, text=text)

    @override
    def __setitem__(self, index: ToIndices, value: object, /) -> None:
        if isinstance(value, TorchTextGeneration):
            self.text[index] = value.text
            self.log_likelihood[index] = value.log_likelihood  # ty:ignore[invalid-assignment]
            return
        if isinstance(value, tuple) and len(value) == 2:
            log_likelihood, text = value
            self.log_likelihood[index] = log_likelihood  # ty:ignore[invalid-assignment]
            self.text[index] = text
            return
        msg = "Assignment requires a TorchTextGeneration or (log_likelihood, text) tuple."
        raise TypeError(msg)

    @override
    @property
    def mT(self) -> Self:
        if self.ndim < 2:
            msg = "mT requires at least 2 batch dimensions."
            raise ValueError(msg)
        return type(self)(
            log_likelihood=torch.transpose(self.log_likelihood, -2, -1),
            text=np.swapaxes(self.text, -2, -1),
        )

    @override
    @property
    def mH(self) -> Self:
        if self.ndim < 2:
            msg = "mH requires at least 2 batch dimensions."
            raise ValueError(msg)
        return type(self)(
            log_likelihood=torch.adjoint(self.log_likelihood),
            text=np.swapaxes(self.text, -2, -1),
        )

    @override
    def to(self, *args: Any, **kwargs: Any) -> Self:
        log_likelihood = self.log_likelihood.to(*args, **kwargs)
        if log_likelihood is self.log_likelihood:
            return self
        return replace(self, log_likelihood=log_likelihood)

    @override
    def detach(self) -> Self:
        log_likelihood = self.log_likelihood.detach()
        if log_likelihood is self.log_likelihood:
            return self
        return replace(self, log_likelihood=log_likelihood)


@dataclass(frozen=True, slots=True, weakref_slot=True)
class TorchTokenGeneration(TorchAxisProtected[torch.Tensor]):
    """Generated token sequences and token transition scores.

    Shape: ``batch_shape``. ``sequences`` stores token ids with a protected trailing
    sequence axis. ``transition_scores`` stores generated-token log probabilities
    with a protected trailing generated-token axis.
    """

    sequences: torch.Tensor
    transition_scores: torch.Tensor
    protected_axes: ClassVar[dict[str, int]] = {"sequences": 1, "transition_scores": 1}

    def __post_init__(self) -> None:
        """Validate sequence and transition-score fields."""
        if not isinstance(self.sequences, torch.Tensor):
            msg = "sequences must be a torch tensor."
            raise TypeError(msg)
        if not isinstance(self.transition_scores, torch.Tensor):
            msg = "transition_scores must be a torch tensor."
            raise TypeError(msg)
        if self.sequences.ndim < 1:
            msg = "sequences must have at least one dimension."
            raise ValueError(msg)
        if self.transition_scores.ndim < 1:
            msg = "transition_scores must have at least one dimension."
            raise ValueError(msg)
        self.protected_values()

    def to_text(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        skip_special_tokens: bool = True,
        **decode_kwargs: Any,  # noqa: ANN401
    ) -> TorchTextGeneration:
        """Decode token sequences and sum transition scores into log-likelihoods.

        Args:
            tokenizer: Tokenizer exposing ``batch_decode``.
            skip_special_tokens: Whether special tokens should be omitted while decoding.
            **decode_kwargs: Additional keyword arguments forwarded to ``batch_decode``.

        Returns:
            The decoded text generation representation.
        """
        batch_shape = self.shape
        flat_sequences = self.sequences.reshape((-1, self.sequences.shape[-1]))
        decoded = tokenizer.batch_decode(
            flat_sequences.detach().cpu().tolist(),
            skip_special_tokens=skip_special_tokens,
            **decode_kwargs,
        )
        text = np.asarray(decoded, dtype=object).reshape(batch_shape)
        log_likelihood = torch.sum(self.transition_scores, dim=-1)
        return TorchTextGeneration(log_likelihood=log_likelihood, text=text)


class TorchTextGenerationSample(TorchSample[TorchTextGeneration]):
    """A torch sample of decoded text generations."""

    sample_space: ClassVar[type[TorchTextGeneration]] = TorchTextGeneration

    @classmethod
    def from_sample(
        cls,
        sample: Sample[TorchTextGeneration] | Iterable[TorchTextGeneration] | TorchTextGeneration,
        **kwargs: Any,  # noqa: ANN401
    ) -> Self:
        """Create a text-generation sample from individual generations.

        Args:
            sample: Individual text-generation representations to stack, or an already stacked representation.
            **kwargs: Optional ``weights``, ``sample_dim``, and ``sample_axis`` arguments.

        Returns:
            The created text-generation sample.
        """
        weights = kwargs.pop("weights", None)
        sample_dim = kwargs.pop("sample_dim", None)
        sample_axis = kwargs.pop("sample_axis", "auto")
        if kwargs:
            msg = f"Unexpected sample arguments: {', '.join(kwargs)}."
            raise TypeError(msg)

        if sample_dim is None:
            if sample_axis is None:
                msg = "Either sample_dim or sample_axis must be not None."
                raise ValueError(msg)
            sample_dim = sample_axis
        elif sample_axis is not None and sample_axis != "auto":
            msg = "Cannot specify both sample_dim and sample_axis."
            raise ValueError(msg)

        if isinstance(sample, TorchTextGeneration):
            if sample_dim == "auto":
                if sample.ndim == 0:
                    msg = "Cannot infer sample_dim for 0-dimensional text generation."
                    raise ValueError(msg)
                sample_dim = -1
            return cls(
                tensor=sample,
                sample_dim=sample_dim,
                weights=torch.as_tensor(weights, device=sample.device) if weights is not None else None,
            )

        sample_iterable = sample.samples if isinstance(sample, Sample) else sample
        sample_list = list(sample_iterable)
        if len(sample_list) == 0:
            msg = "Cannot create a text-generation sample from an empty iterable."
            raise ValueError(msg)
        if not all(isinstance(item, TorchTextGeneration) for item in sample_list):
            msg = "All samples must be TorchTextGeneration instances."
            raise TypeError(msg)
        text_generations = cast("list[TorchTextGeneration]", sample_list)
        if sample_dim == "auto":
            sample_dim = -1

        text = np.stack([generation.text for generation in text_generations], axis=sample_dim)
        log_likelihood = torch.stack([generation.log_likelihood for generation in text_generations], dim=sample_dim)
        return cls(
            tensor=TorchTextGeneration(log_likelihood=log_likelihood, text=text),
            sample_dim=sample_dim,
            weights=torch.as_tensor(weights, device=log_likelihood.device) if weights is not None else None,
        )

    @property
    @override
    def samples(self) -> Sequence[TorchTextGeneration]:
        return [self.tensor[index] for index in range(self.sample_size)]

    @override
    def concat(self, other: Sample[TorchTextGeneration]) -> Self:
        if isinstance(other, TorchTextGenerationSample):
            other_tensor = other.move_sample_dim(self.sample_dim).tensor
            other_weights = other.weights
        else:
            other_sample = type(self).from_sample(other.samples, sample_dim=self.sample_dim)
            other_tensor = other_sample.tensor
            other_weights = other_sample.weights

        text = np.concatenate((self.tensor.text, other_tensor.text), axis=self.sample_dim)
        log_likelihood = torch.cat((self.tensor.log_likelihood, other_tensor.log_likelihood), dim=self.sample_dim)

        weights = self.weights
        if weights is not None or other_weights is not None:
            if weights is None:
                weights = torch.ones(self.sample_size, device=self.tensor.device)
            other_weights = (
                torch.ones(other_tensor.shape[self.sample_dim], device=other_tensor.device)
                if other_weights is None
                else torch.as_tensor(other_weights, device=other_tensor.device)
            )
            weights = torch.cat((weights, other_weights), dim=0)

        return type(self)(
            tensor=TorchTextGeneration(log_likelihood=log_likelihood, text=text),
            sample_dim=self.sample_dim,
            weights=weights,
        )

    @override
    def move_sample_dim(self, new_sample_dim: int) -> Self:
        text = np.moveaxis(self.tensor.text, self.sample_dim, new_sample_dim)
        log_likelihood = torch.moveaxis(self.tensor.log_likelihood, self.sample_dim, new_sample_dim)
        return type(self)(
            tensor=TorchTextGeneration(log_likelihood=log_likelihood, text=text),
            sample_dim=new_sample_dim,
            weights=self.weights,
        )

    @override
    def move_sample_axis(self, new_sample_axis: int) -> Self:
        return self.move_sample_dim(new_sample_axis)


create_sample.register(TorchTextGeneration, TorchTextGenerationSample.from_sample)
