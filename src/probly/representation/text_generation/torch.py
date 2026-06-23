"""Torch-backed text generation representations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, override

import numpy as np
import torch

from probly.representation._protected_axis.torch import TorchAxisProtected
from probly.representation.sample import RepresentationSample
from probly.representation.sample.torch import TorchSample

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True, slots=True, weakref_slot=True)
class TorchTextGeneration(TorchAxisProtected[Any]):
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

    def _text_log_likelihood(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        length_normalization: bool,
        stop_token_ids: set[int] | None = None,
    ) -> torch.Tensor:
        """Compute sequence log likelihoods for decoded generated text.

        Args:
            tokenizer: Tokenizer providing fallback EOS and padding token ids.
            length_normalization: Whether to average over scored tokens instead of summing.
            stop_token_ids: Explicit stop token ids. If omitted, tokenizer EOS and padding ids are used.

        Returns:
            Per-sequence log likelihoods with shape ``batch_shape``.
        """
        if self.sequences.shape[-1] < self.transition_scores.shape[-1]:
            msg = "Generated sequences cannot be shorter than transition scores."
            raise ValueError(msg)

        scores = self.transition_scores.float()
        if scores.shape[-1] == 0:
            return torch.zeros(self.shape, dtype=scores.dtype, device=scores.device)

        generated_tokens = self.sequences[..., -scores.shape[-1] :].to(device=scores.device)
        if stop_token_ids is None:
            stop_token_ids: set[int | None] = {
                getattr(tokenizer, "eos_token_id", None),
                getattr(tokenizer, "pad_token_id", None),
            }
            stop_token_ids: set[int] = {token_id for token_id in stop_token_ids if token_id is not None}

        if len(stop_token_ids) == 0:
            scored_mask = torch.ones_like(scores, dtype=torch.bool)
        else:
            stop_tokens = torch.tensor(
                sorted(stop_token_ids), dtype=generated_tokens.dtype, device=generated_tokens.device
            )
            stop_mask = torch.isin(generated_tokens, stop_tokens)
            scored_mask = torch.cumsum(stop_mask.to(dtype=torch.long), dim=-1) == 0

        summed = torch.where(scored_mask, scores, torch.zeros_like(scores)).sum(dim=-1)
        if not length_normalization:
            return summed

        lengths = scored_mask.sum(dim=-1)
        return torch.where(
            lengths > 0,
            summed / lengths.clamp_min(1).to(dtype=scores.dtype),
            torch.zeros_like(summed),
        )

    def to_text(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        length_normalization: bool = False,
        stop_token_ids: set[int] | None = None,
        skip_special_tokens: bool = True,
        **decode_kwargs: Any,  # noqa: ANN401
    ) -> TorchTextGeneration:
        """Decode token sequences and sum transition scores into log-likelihoods.

        Args:
            tokenizer: Tokenizer exposing ``batch_decode``.
            length_normalization: Whether log likelihoods should be averaged over scored tokens.
            stop_token_ids: Explicit stop token ids. If omitted, tokenizer EOS and padding ids are used.
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
        log_likelihood = self._text_log_likelihood(
            tokenizer,
            length_normalization=length_normalization,
            stop_token_ids=stop_token_ids,
        )
        return TorchTextGeneration(log_likelihood=log_likelihood, text=text)


class TorchTextGenerationSample(  # ty:ignore[conflicting-metaclass]
    RepresentationSample[TorchTextGeneration],
    TorchSample[TorchTextGeneration],
):
    """A torch sample of decoded text generations."""

    sample_space: ClassVar[type[TorchTextGeneration]] = TorchTextGeneration

    @override
    @classmethod
    def __instancehook__(cls, instance: object) -> bool:
        return super().__instancehook__(instance)


class TorchTextGenerationSampleSample(  # ty:ignore[conflicting-metaclass]
    RepresentationSample[TorchTextGenerationSample],
    TorchSample[Any],
):
    """A torch sample of decoded text generation samples."""

    sample_space: ClassVar[type[TorchTextGenerationSample]] = TorchTextGenerationSample

    @override
    @classmethod
    def __instancehook__(cls, instance: object) -> bool:
        return super().__instancehook__(instance)
