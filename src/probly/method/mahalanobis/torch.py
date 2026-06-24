"""Torch implementation of the Mahalanobis OOD transformation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch
from torch import nn
import torch.nn.functional as F

from probly.layers.torch import MahalanobisHead
from probly.representation._protected_axis.torch import TorchAxisProtected
from probly.representation.distribution.torch_categorical import (
    TorchCategoricalDistribution,
    TorchProbabilityCategoricalDistribution,
)
from probly.traverse_nn import nn_compose, nn_traverser
from pytraverse import TRAVERSE_REVERSED, GlobalVariable, State, singledispatch_traverser, traverse_with_state

from ._common import (
    MahalanobisPredictor,
    MahalanobisRepresentation,
    combine_layer_scores,
    create_mahalanobis_representation,
    mahalanobis_generator,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from torch import Tensor


@create_mahalanobis_representation.register(TorchCategoricalDistribution)
@dataclass(frozen=True, slots=True, weakref_slot=True)
class TorchMahalanobisRepresentation(MahalanobisRepresentation, TorchAxisProtected):
    """Mahalanobis representation backed by torch tensors.

    ``weight`` and ``bias`` are shared (non per-sample) combiner parameters and
    are therefore left out of ``protected_axes`` so they ride along unchanged
    through indexing and batching.
    """

    softmax: TorchCategoricalDistribution
    layer_scores: Tensor
    weight: Tensor
    bias: Tensor
    protected_axes: ClassVar[dict[str, int]] = {"softmax": 0, "layer_scores": 1}


@combine_layer_scores.register(torch.Tensor)
def torch_combine_layer_scores(layer_scores: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Combine per-layer Mahalanobis confidences into a single OOD score.

    The combination is the logistic-regression logit ``s @ w + b``. With the
    default weights (``-1`` per layer and zero bias) this is the negated sum of
    the per-layer confidences, so a far-from-centroid (out-of-distribution) input
    yields a high score.
    """
    return layer_scores @ weight + bias


HEAD_MODULE: GlobalVariable[nn.Module | None] = GlobalVariable("MAHALANOBIS_HEAD_MODULE", default=None)


@singledispatch_traverser
def head_strip_traverser(obj: nn.Module, state: State) -> tuple[nn.Module, State]:
    """Default handler: return module unchanged."""
    return obj, state


@head_strip_traverser.register
def _(obj: nn.Linear, state: State) -> tuple[nn.Module, State]:
    """Replace the last Linear layer (the classification head) with Identity.

    With ``TRAVERSE_REVERSED`` the final Linear is encountered first; it is
    stored in ``HEAD_MODULE`` and replaced with ``nn.Identity()`` so the
    remaining model is a pure feature encoder.
    """
    if state[HEAD_MODULE] is None:
        state[HEAD_MODULE] = obj
        return nn.Identity(), state
    return obj, state


@mahalanobis_generator.register(nn.Module)
class TorchMahalanobisPredictor(nn.Module, MahalanobisPredictor[[torch.Tensor], TorchMahalanobisRepresentation]):
    """Torch Mahalanobis OOD predictor.

    The final ``nn.Linear`` head is replaced with ``nn.Identity()`` to expose the
    penultimate features as the encoder output; the original head is kept for
    classification.  One :class:`MahalanobisHead` is fitted per feature layer
    (any user-provided intermediate modules plus the penultimate features), and
    the per-layer Mahalanobis confidences are combined into a single OOD score.

    After training, call ``fit_mahalanobis_heads(features, labels)`` to estimate
    the Gaussian parameters; optionally call ``fit_combiner(id, ood)`` to
    calibrate the multi-layer combination weights on in- vs out-of-distribution
    data.

    Attributes:
        encoder: The feature encoder (head replaced with Identity).
        classification_head: The original final Linear layer.
        mahalanobis_heads: One Mahalanobis head per feature layer (populated by
            ``fit_mahalanobis_heads``).
        combiner_weight: Per-layer combination weights of shape ``(num_layers,)``,
            initialised to ``-1`` and calibrated by ``fit_combiner``.
        combiner_bias: Scalar combination bias, initialised to ``0`` and
            calibrated by ``fit_combiner``.
    """

    encoder: nn.Module
    classification_head: nn.Linear
    mahalanobis_heads: nn.ModuleList
    combiner_weight: torch.Tensor
    combiner_bias: torch.Tensor

    def __init__(
        self,
        model: nn.Module,
        feature_nodes: Sequence[str] | None = None,
        input_preprocessing_eps: float = 0.0,
    ) -> None:
        """Build the Mahalanobis predictor from a base classifier.

        Args:
            model: Base classification model to be transformed.
            feature_nodes: Optional names of intermediate submodules (as returned
                by ``named_modules``) whose outputs provide additional feature
                layers. When ``None`` only the penultimate features are used.
            input_preprocessing_eps: Magnitude of the FGSM-style input
                perturbation applied at inference. ``0`` disables it.
        """
        super().__init__()
        encoder, state = traverse_with_state(
            model,
            nn_compose(head_strip_traverser, nn_traverser=nn_traverser),
            init={HEAD_MODULE: None, TRAVERSE_REVERSED: True},
        )
        head: nn.Linear | None = state[HEAD_MODULE]  # ty:ignore[invalid-assignment]
        if head is None:
            msg = "No nn.Linear layer found in the model; cannot identify a classification head."
            raise ValueError(msg)

        self.encoder = encoder
        self.classification_head = head
        self.input_preprocessing_eps = input_preprocessing_eps
        self._num_classes = head.out_features

        self._feature_nodes = list(feature_nodes) if feature_nodes is not None else []
        self.mahalanobis_heads = nn.ModuleList()
        num_layers = len(self._feature_nodes) + 1
        # Default combiner: negated sum of per-layer confidences (high score => out-of-distribution).
        self.register_buffer("combiner_weight", -torch.ones(num_layers))
        self.register_buffer("combiner_bias", torch.zeros(()))

        # Hook the requested intermediate nodes on the rebuilt encoder (the
        # traversal preserves submodule names but rebuilds the module objects).
        self._captured: list[torch.Tensor] = []
        for name in self._feature_nodes:
            self.encoder.get_submodule(name).register_forward_hook(self._capture_hook())

    def _capture_hook(self) -> Callable[[nn.Module, object, torch.Tensor], None]:
        """Build a forward hook that records an intermediate module's output."""

        def hook(_module: nn.Module, _inputs: object, output: torch.Tensor) -> None:
            self._captured.append(output)

        return hook

    @staticmethod
    def _pool(features: torch.Tensor) -> torch.Tensor:
        """Global-average-pool any trailing spatial axes to shape ``(N, C)``."""
        if features.dim() <= 2:
            return features
        return features.mean(dim=tuple(range(2, features.dim())))

    def _forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run the encoder and collect the penultimate plus pooled intermediate features."""
        self._captured = []
        penultimate = self.encoder(x)
        feats = [self._pool(captured) for captured in self._captured]
        feats.append(self._pool(penultimate))
        return penultimate, feats

    def _layer_scores(self, feats: list[torch.Tensor]) -> torch.Tensor:
        """Per-layer Mahalanobis confidences (max over classes) stacked to ``(N, num_layers)``."""
        scores = [head(feat).max(dim=-1).values for head, feat in zip(self.mahalanobis_heads, feats, strict=True)]
        return torch.stack(scores, dim=-1)

    def _preprocessed_layer_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Per-layer scores after the FGSM-style input preprocessing of Lee et al. 2018.

        For each feature layer the input is independently nudged along the
        gradient that raises that layer's Mahalanobis confidence
        (``x + eps * sign(grad)``), then re-encoded and re-scored. This widens the
        gap between in- and out-of-distribution inputs.

        Args:
            x: Raw input tensor.

        Returns:
            Per-layer scores of shape ``(N, num_layers)``.
        """
        scores = []
        for layer_index, head in enumerate(self.mahalanobis_heads):
            with torch.enable_grad():
                x_var = x.clone().detach().requires_grad_(True)
                _, feats = self._forward_features(x_var)
                confidence = head(feats[layer_index]).max(dim=-1).values
                grad = torch.autograd.grad(confidence.sum(), x_var)[0]
            # Move the input to raise its confidence, then re-score (no grad).
            x_perturbed = (x_var + self.input_preprocessing_eps * grad.sign()).detach()
            with torch.no_grad():
                _, feats_perturbed = self._forward_features(x_perturbed)
                scores.append(head(feats_perturbed[layer_index]).max(dim=-1).values)
        return torch.stack(scores, dim=-1)

    def _input_layer_scores(self, x: torch.Tensor, feats: list[torch.Tensor]) -> torch.Tensor:
        """Per-layer Mahalanobis scores for the raw input ``x``.

        Applies the FGSM-style input preprocessing when
        ``input_preprocessing_eps`` is positive (recomputing the features from
        ``x``); otherwise returns the plain confidences from the already-extracted
        ``feats``. Both calibration (``fit_combiner``) and inference
        (``predict_representation``) route through this method so the combiner is
        trained on the same scores it is later applied to, as in Lee et al. 2018.

        Args:
            x: Raw input tensor.
            feats: Plain features already extracted from ``x``; used only when
                preprocessing is disabled.

        Returns:
            Per-layer scores of shape ``(N, num_layers)``.
        """
        if self.input_preprocessing_eps > 0:
            return self._preprocessed_layer_scores(x)
        return self._layer_scores(feats)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode features, classify, and score the raw per-layer Mahalanobis confidence.

        The returned scores are the plain confidences without FGSM input
        preprocessing; the preprocessing-aware scores used for OOD detection are
        produced by :meth:`predict_representation`.

        Args:
            x: Input tensor passed to the encoder.

        Returns:
            A 2-tuple of ``(logits, layer_scores)`` of shapes ``(N, num_classes)``
            and ``(N, num_layers)``.
        """
        penultimate, feats = self._forward_features(x)
        logits = self.classification_head(penultimate)
        return logits, self._layer_scores(feats)

    def fit_mahalanobis_heads(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        """Fit one Mahalanobis head per feature layer on the given inputs.

        Args:
            features: Input tensor (e.g. the training inputs) fed to the encoder.
            labels: Integer class labels of shape ``(N,)``.
        """
        encoder_state = self.encoder.training
        self.encoder.eval()
        with torch.no_grad():
            _, feats = self._forward_features(features)
        heads = [MahalanobisHead(self._num_classes, feat.shape[-1]).to(feat.device) for feat in feats]
        for head, feat in zip(heads, feats, strict=True):
            head.fit(feat, labels)
        self.mahalanobis_heads = nn.ModuleList(heads)
        self.encoder.train(encoder_state)

    def fit_combiner(
        self,
        id_features: torch.Tensor,
        ood_features: torch.Tensor,
        steps: int = 1000,
        lr: float = 0.05,
    ) -> None:
        """Calibrate the multi-layer combination weights by logistic regression.

        Fits weights and a bias over the per-layer Mahalanobis scores so that
        out-of-distribution inputs (label 1) score higher than in-distribution
        inputs (label 0), reproducing the feature-ensemble step of
        :cite:`leeSimpleUnifiedFramework2018` with a torch-native logistic
        regression.

        Args:
            id_features: In-distribution inputs fed to the encoder.
            ood_features: Out-of-distribution inputs fed to the encoder.
            steps: Number of optimisation steps.
            lr: Learning rate for the Adam optimiser.
        """
        encoder_state = self.encoder.training
        self.encoder.eval()
        with torch.no_grad():
            _, id_feats = self._forward_features(id_features)
            _, ood_feats = self._forward_features(ood_features)
        # Score through the same path as inference (FGSM preprocessing included
        # when enabled) so the combiner is calibrated on what it later sees.
        id_scores = self._input_layer_scores(id_features, id_feats)
        ood_scores = self._input_layer_scores(ood_features, ood_feats)
        self.encoder.train(encoder_state)

        scores = torch.cat([id_scores, ood_scores], dim=0).detach()
        targets = torch.cat([torch.zeros(len(id_scores)), torch.ones(len(ood_scores))]).to(scores.device)

        weight = torch.zeros(scores.shape[-1], device=scores.device, requires_grad=True)
        bias = torch.zeros((), device=scores.device, requires_grad=True)
        optimizer = torch.optim.Adam([weight, bias], lr=lr)
        for _ in range(steps):
            optimizer.zero_grad()
            logits = scores @ weight + bias
            loss = F.binary_cross_entropy_with_logits(logits, targets)
            loss.backward()
            optimizer.step()

        self.combiner_weight.copy_(weight.detach())
        self.combiner_bias.copy_(bias.detach())

    def predict_representation(self, x: torch.Tensor) -> TorchMahalanobisRepresentation:
        """Predict the Mahalanobis representation (softmax and per-layer scores)."""
        penultimate, feats = self._forward_features(x)
        logits = self.classification_head(penultimate)
        layer_scores = self._input_layer_scores(x, feats)

        return TorchMahalanobisRepresentation(
            TorchProbabilityCategoricalDistribution(torch.softmax(logits, dim=-1)),
            layer_scores,
            self.combiner_weight,
            self.combiner_bias,
        )
