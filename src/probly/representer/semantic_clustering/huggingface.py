"""Semantic clustering representers backed by Hugging Face NLI models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Self, override

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from probly.representation.distribution.torch_sparse_log_categorical import (
    TorchSparseLogCategoricalDistribution,
    TorchSparseLogCategoricalDistributionSample,
)
from probly.representation.text_generation import (
    TorchTextGeneration,
    TorchTextGenerationSample,
    TorchTextGenerationSampleSample,
)
from probly.representer._representer import Representer

if TYPE_CHECKING:
    from os import PathLike


DEFAULT_NLI_MODEL = "microsoft/deberta-base-mnli"

type SemanticClusterInput = TorchTextGeneration | TorchTextGenerationSample | TorchTextGenerationSampleSample
type SemanticClusterOutput = TorchSparseLogCategoricalDistribution | TorchSparseLogCategoricalDistributionSample

_CONTRADICTION = 0
_NEUTRAL = 1
_ENTAILMENT = 2


def _canonical_label_name(label: object) -> int | None:
    normalized = str(label).lower().replace("_", " ").replace("-", " ")
    if "contradiction" in normalized:
        return _CONTRADICTION
    if "neutral" in normalized:
        return _NEUTRAL
    if "entail" in normalized:
        return _ENTAILMENT
    return None


def _label_id_to_canonical(model: PreTrainedModel) -> dict[int, int]:
    config = getattr(model, "config", None)
    mapping: dict[int, int] = {}

    label2id = getattr(config, "label2id", None)
    if isinstance(label2id, Mapping):
        for label, label_id in label2id.items():
            canonical = _canonical_label_name(label)
            if canonical is not None:
                mapping[int(label_id)] = canonical

    id2label = getattr(config, "id2label", None)
    if isinstance(id2label, Mapping):
        for label_id, label in id2label.items():
            canonical = _canonical_label_name(label)
            if canonical is not None:
                mapping[int(label_id)] = canonical

    if {_CONTRADICTION, _NEUTRAL, _ENTAILMENT}.issubset(set(mapping.values())):
        return mapping

    return {_CONTRADICTION: _CONTRADICTION, _NEUTRAL: _NEUTRAL, _ENTAILMENT: _ENTAILMENT}


class HFSemanticClusterer(Representer[Any, Any, torch.Tensor, SemanticClusterOutput], ABC):
    """Base semantic clusterer using a Hugging Face NLI model."""

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    batch_size: int
    max_length: int | None
    truncation: bool
    _label_id_to_canonical: dict[int, int]

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        *,
        batch_size: int = 32,
        max_length: int | None = None,
        truncation: bool = True,
    ) -> None:
        """Initialize the semantic clusterer.

        Args:
            model: Hugging Face NLI sequence classification model.
            tokenizer: Tokenizer associated with ``model``.
            batch_size: Number of NLI pairs to score in one model call.
            max_length: Optional maximum pair sequence length passed to the tokenizer.
            truncation: Whether pair inputs should be truncated to ``max_length`` or model defaults.
        """
        if batch_size <= 0:
            msg = "batch_size must be positive."
            raise ValueError(msg)
        if max_length is not None and max_length <= 0:
            msg = "max_length must be positive when provided."
            raise ValueError(msg)

        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.truncation = truncation
        self._label_id_to_canonical = _label_id_to_canonical(model)

    @classmethod
    def from_model_name(
        cls,
        model_name: str | None = None,
        *,
        cache_dir: str | PathLike[str] | None = None,
        force_download: bool = False,
        model_kwargs: Mapping[str, object] | None = None,
        tokenizer_kwargs: Mapping[str, object] | None = None,
        batch_size: int = 32,
        max_length: int | None = None,
        truncation: bool = True,
    ) -> Self:
        """Load an NLI model by name and initialize the clusterer.

        Args:
            model_name: Hugging Face model name or local path. Defaults to ``DEFAULT_NLI_MODEL``.
            cache_dir: Optional Hugging Face cache directory.
            force_download: Whether Hugging Face should re-download files even if cached.
            model_kwargs: Additional keyword arguments forwarded to ``AutoModelForSequenceClassification``.
            tokenizer_kwargs: Additional keyword arguments forwarded to ``AutoTokenizer``.
            batch_size: Number of NLI pairs to score in one model call.
            max_length: Optional maximum pair sequence length passed to the tokenizer.
            truncation: Whether pair inputs should be truncated to ``max_length`` or model defaults.

        Returns:
            A semantic clusterer backed by the loaded NLI model and tokenizer.
        """
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: PLC0415

        resolved_model_name = DEFAULT_NLI_MODEL if model_name is None else model_name
        tokenizer = AutoTokenizer.from_pretrained(
            resolved_model_name,
            cache_dir=cache_dir,
            force_download=force_download,
            **dict(tokenizer_kwargs or {}),
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            resolved_model_name,
            cache_dir=cache_dir,
            force_download=force_download,
            **dict(model_kwargs or {}),
        )

        if not isinstance(tokenizer, PreTrainedTokenizerBase):
            msg = f"Expected a PreTrainedTokenizerBase tokenizer, got {type(tokenizer)}."
            raise TypeError(msg)
        if not isinstance(model, PreTrainedModel):
            msg = f"Expected a PreTrainedModel model, got {type(model)}."
            raise TypeError(msg)

        model.eval()
        return cls(
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            truncation=truncation,
        )

    @property
    def predictor(self) -> PreTrainedModel:
        """The underlying NLI model used for pairwise entailment."""
        return self.model

    def _normalize_axis(self, axis: int, ndim: int) -> int:
        normalized = axis + ndim if axis < 0 else axis
        if normalized < 0 or normalized >= ndim:
            msg = f"axis {axis} out of bounds for generation with ndim {ndim}."
            raise ValueError(msg)
        return normalized

    def _move_tokenized_to_model_device(self, tokenized: Mapping[str, object]) -> dict[str, object]:

        if not isinstance(tokenized, Mapping):
            msg = "Tokenizer must return a mapping of model inputs."
            raise TypeError(msg)

        device = torch.device(self.model.device)
        return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in tokenized.items()}

    def _tokenize_pairs(
        self,
        statements: np.ndarray,
        pairs: torch.Tensor,
    ) -> dict[str, object]:
        pair_indices = pairs.detach().cpu().numpy()
        premises = statements[pair_indices[:, 0]].astype(str, copy=False).tolist()
        hypotheses = statements[pair_indices[:, 1]].astype(str, copy=False).tolist()
        kwargs: dict[str, object] = {
            "padding": True,
            "return_tensors": "pt",
            "truncation": self.truncation,
        }
        if self.max_length is not None:
            kwargs["max_length"] = self.max_length

        tokenized = self.tokenizer(premises, hypotheses, **kwargs)
        return self._move_tokenized_to_model_device(tokenized)

    def _canonicalize_label_ids(self, label_ids: torch.Tensor, num_labels: int) -> torch.Tensor:
        labels: list[int] = []
        for label_id in label_ids.detach().cpu().tolist():
            raw_label = int(label_id)
            canonical = self._label_id_to_canonical.get(raw_label)
            if canonical is None and 0 <= raw_label < min(num_labels, 3):
                canonical = raw_label
            if canonical not in {_CONTRADICTION, _NEUTRAL, _ENTAILMENT}:
                msg = f"Cannot map NLI label id {raw_label} to contradiction/neutral/entailment."
                raise ValueError(msg)
            labels.append(canonical)
        return torch.tensor(labels, dtype=torch.long)

    def _predict_pair_labels(
        self,
        statements: np.ndarray,
        pairs: torch.Tensor,
    ) -> torch.Tensor:
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            msg = "pairs must have shape (num_pairs, 2)."
            raise ValueError(msg)
        if pairs.numel() == 0:
            return torch.empty(0, dtype=torch.long)

        labels: list[torch.Tensor] = []
        for start in range(0, pairs.shape[0], self.batch_size):
            batch_pairs = pairs[start : start + self.batch_size]
            inputs = self._tokenize_pairs(statements, batch_pairs)
            with torch.inference_mode():
                outputs = self.model(**inputs)

            logits = getattr(outputs, "logits", None)
            if logits is None and isinstance(outputs, Mapping):
                logits = outputs.get("logits")
            if logits is None and isinstance(outputs, (tuple, list)) and len(outputs) > 0:
                logits = outputs[0]
            if not isinstance(logits, torch.Tensor):
                msg = "NLI model output must include a logits tensor."
                raise TypeError(msg)
            if logits.ndim != 2:
                msg = "NLI model logits must have shape (batch_size, num_labels)."
                raise ValueError(msg)

            labels.append(self._canonicalize_label_ids(torch.argmax(logits, dim=-1), logits.shape[-1]))

        return torch.cat(labels, dim=0)

    @abstractmethod
    def _cluster_row(self, statements: np.ndarray) -> torch.Tensor:
        """Cluster one row of generated statements."""

    def _cluster(self, generation: TorchTextGeneration, axis: int) -> TorchSparseLogCategoricalDistribution:
        axis = self._normalize_axis(axis, generation.ndim)
        comparison_size = generation.shape[axis]
        if comparison_size == 0:
            msg = "Cannot semantically cluster an empty comparison axis."
            raise ValueError(msg)

        text = np.moveaxis(generation.text, axis, -1)
        logits = torch.moveaxis(generation.log_likelihood, axis, -1)
        row_shape = text.shape[:-1]
        flat_text = text.reshape((int(np.prod(row_shape, dtype=np.int64)), comparison_size))

        cluster_rows: list[torch.Tensor] = []
        for row_idx in range(flat_text.shape[0]):
            row_text = flat_text[row_idx]
            cluster_row = self._cluster_row(row_text).to(dtype=torch.long)
            if cluster_row.shape != (comparison_size,):
                msg = f"Cluster row must have shape ({comparison_size},), got {tuple(cluster_row.shape)}."
                raise ValueError(msg)
            cluster_rows.append(cluster_row)

        group_ids = torch.stack(cluster_rows, dim=0).reshape((*row_shape, comparison_size))
        group_ids = group_ids.to(device=logits.device)
        return TorchSparseLogCategoricalDistribution(group_ids=group_ids, logits=logits)

    def _with_sample_weights(
        self,
        distribution: TorchSparseLogCategoricalDistribution,
        weights: object,
    ) -> TorchSparseLogCategoricalDistribution:
        if weights is None:
            return distribution

        weights_tensor = torch.as_tensor(weights, dtype=distribution.logits.dtype, device=distribution.logits.device)
        if torch.any(weights_tensor < 0):
            msg = "sample weights must be non-negative."
            raise ValueError(msg)
        log_weights = torch.log(weights_tensor).reshape(
            (*((1,) * (distribution.logits.ndim - 1)), weights_tensor.shape[0])
        )
        return TorchSparseLogCategoricalDistribution(
            group_ids=distribution.group_ids,
            logits=distribution.logits + log_weights,
        )

    def _sample_weights_tensor(self, weights: object, *, device: torch.device) -> torch.Tensor | None:
        if weights is None:
            return None

        weights_tensor = torch.as_tensor(weights, dtype=torch.float32, device=device)
        if torch.any(weights_tensor < 0):
            msg = "sample weights must be non-negative."
            raise ValueError(msg)
        return weights_tensor

    def represent(
        self,
        generation: SemanticClusterInput,
        *,
        axis: int | None = None,
    ) -> SemanticClusterOutput:
        """Cluster text generations into semantic equivalence classes.

        Args:
            generation: Text generation representation or sample.
            axis: Comparison axis for raw ``TorchTextGeneration`` inputs. Ignored for samples.

        Returns:
            Sparse grouped logits whose final axis contains semantic cluster assignments.
        """
        if isinstance(generation, TorchTextGenerationSampleSample):
            clustered = self.represent(generation.tensor)
            if not isinstance(clustered, TorchSparseLogCategoricalDistribution):
                msg = "Nested text generation samples must contain a single inner sample axis."
                raise TypeError(msg)

            return TorchSparseLogCategoricalDistributionSample(
                tensor=clustered,
                sample_dim=generation.sample_dim,
                weights=self._sample_weights_tensor(generation.weights, device=clustered.logits.device),
            )

        if isinstance(generation, TorchTextGenerationSample):
            sample_dim = generation.sample_dim
            clustered = self._cluster(generation.tensor, sample_dim)
            return self._with_sample_weights(clustered, generation.weights)

        if not isinstance(generation, TorchTextGeneration):
            msg = (
                "generation must be a TorchTextGeneration, TorchTextGenerationSample, or nested text generation sample."
            )
            raise TypeError(msg)
        if axis is None:
            msg = "axis must be provided when clustering a TorchTextGeneration."
            raise ValueError(msg)
        return self._cluster(generation, axis)


class GreedyHFSemanticClusterer(HFSemanticClusterer):
    """Greedy semantic clustering via bidirectional NLI labels."""

    @staticmethod
    def _semantic_equivalence_mask(label_pairs: torch.Tensor) -> torch.Tensor:
        if label_pairs.ndim != 2 or label_pairs.shape[1] != 2:
            msg = "label_pairs must have shape (num_pairs, 2)."
            raise ValueError(msg)
        has_no_contradiction = torch.all(label_pairs != _CONTRADICTION, dim=1)
        is_not_neutral_neutral = ~torch.all(label_pairs == _NEUTRAL, dim=1)
        return has_no_contradiction & is_not_neutral_neutral

    @override
    def _cluster_row(self, statements: np.ndarray) -> torch.Tensor:
        if statements.ndim != 1:
            msg = "statements must be a one-dimensional numpy array."
            raise ValueError(msg)

        semantic_ids = torch.full((statements.shape[0],), -1, dtype=torch.long)
        next_id = 0

        for i in range(statements.shape[0]):
            if semantic_ids[i].item() != -1:
                continue

            semantic_ids[i] = next_id
            candidates = torch.nonzero(semantic_ids[i + 1 :] == -1, as_tuple=False).flatten() + i + 1
            if candidates.numel() > 0:
                pairs = torch.empty((candidates.numel() * 2, 2), dtype=torch.long)
                pairs[0::2, 0] = i
                pairs[0::2, 1] = candidates
                pairs[1::2, 0] = candidates
                pairs[1::2, 1] = i
                labels = self._predict_pair_labels(statements, pairs)
                equivalent = self._semantic_equivalence_mask(labels.reshape(-1, 2))
                semantic_ids[candidates[equivalent]] = next_id

            next_id += 1

        return semantic_ids
