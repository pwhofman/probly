"""Flax implementation of the Mahalanobis OOD transformation."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, cast

from flax import nnx
import jax
from jax import numpy as jnp
import optax

from probly.layers.flax import Identity, MahalanobisHead
from probly.representation._protected_axis.jax import JaxAxisProtected
from probly.representation.distribution.jax_categorical import (
    JaxCategoricalDistribution,
    JaxProbabilityCategoricalDistribution,
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
    from collections.abc import Iterator, Sequence

    from jax import Array


@create_mahalanobis_representation.register(JaxCategoricalDistribution)
@dataclass(frozen=True, slots=True, weakref_slot=True)
class JaxMahalanobisRepresentation(MahalanobisRepresentation, JaxAxisProtected[jax.Array]):
    """Mahalanobis representation backed by jax arrays.

    ``weight`` and ``bias`` are shared (non per-sample) combiner parameters and
    are therefore left out of ``protected_axes`` so they ride along unchanged
    through indexing and batching.
    """

    softmax: JaxCategoricalDistribution
    layer_scores: Array
    weight: Array
    bias: Array
    protected_axes: ClassVar[dict[str, int]] = {"softmax": 0, "layer_scores": 1}


@combine_layer_scores.register(jax.Array)
def jax_combine_layer_scores(layer_scores: jax.Array, weight: jax.Array, bias: jax.Array) -> jax.Array:
    """Combine per-layer Mahalanobis confidences into a single OOD score.

    The combination is the logistic-regression logit ``s @ w + b``. With the
    default weights (``-1`` per layer and zero bias) this is the negated sum of
    the per-layer confidences, so a far-from-centroid (out-of-distribution) input
    yields a high score.
    """
    return layer_scores @ weight + bias


HEAD_MODULE: GlobalVariable[nnx.Module | None] = GlobalVariable("MAHALANOBIS_HEAD_MODULE", default=None)


@singledispatch_traverser
def head_strip_traverser(obj: nnx.Module, state: State) -> tuple[nnx.Module, State]:
    """Default handler: return module unchanged."""
    return obj, state


@head_strip_traverser.register
def _(obj: nnx.Linear, state: State) -> tuple[nnx.Module, State]:
    """Replace the last Linear layer (the classification head) with an identity.

    With ``TRAVERSE_REVERSED`` the final Linear is encountered first; it is
    stored in ``HEAD_MODULE`` and replaced with :class:`probly.layers.flax.Identity`
    so the remaining model is a pure feature encoder.
    """
    if state[HEAD_MODULE] is None:
        state[HEAD_MODULE] = obj
        return Identity(), state
    return obj, state


class FeatureCapture(nnx.Module):
    """Wrap a submodule and record its output for Mahalanobis feature extraction.

    ``flax.nnx`` has no equivalent of torch's forward hooks, so intermediate
    feature layers are tapped by substituting the module of interest with this
    wrapper. The output is stored in an :class:`flax.nnx.Intermediate` variable
    rather than a plain python attribute so that it is visible to
    ``nnx.split``/``nnx.merge`` and survives ``nnx.jit``.

    Attributes:
        inner: The wrapped module, called unchanged.
        captured: The most recent output of ``inner``, or ``None`` before the
            first call.
    """

    def __init__(self, inner: nnx.Module) -> None:
        """Wrap ``inner`` without altering its behaviour.

        Args:
            inner: The module whose output should be recorded.
        """
        super().__init__()
        self.inner = inner
        self.captured = nnx.Intermediate(None)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """Call the wrapped module and record its output.

        Args:
            *args: Positional arguments forwarded to the wrapped module.
            **kwargs: Keyword arguments forwarded to the wrapped module.

        Returns:
            Whatever the wrapped module returns.
        """
        output = self.inner(*args, **kwargs)
        self.captured.set_value(output)
        return output


def _module_path(path: tuple[Any, ...]) -> str:
    """Render an ``nnx.iter_modules`` path tuple as a dotted name."""
    return ".".join(str(part) for part in path)


def _get_child(container: Any, key: Any) -> Any:  # noqa: ANN401
    """Return the child of ``container`` at ``key``."""
    return container[key] if isinstance(key, int) else getattr(container, key)


def _set_child(container: Any, key: Any, value: nnx.Module) -> None:  # noqa: ANN401
    """Replace the child of ``container`` at ``key``."""
    if isinstance(key, int):
        container[key] = value
    else:
        setattr(container, key, value)


def _resolve_container(root: nnx.Module, path: tuple[Any, ...]) -> tuple[Any, Any]:
    """Return the ``(container, key)`` pair that addresses ``path`` inside ``root``."""
    node: Any = root
    for part in path[:-1]:
        node = _get_child(node, part)
    return node, path[-1]


def _wrap_feature_nodes(encoder: nnx.Module, feature_nodes: Sequence[str]) -> nnx.List:
    """Substitute each requested submodule with a :class:`FeatureCapture` wrapper.

    Args:
        encoder: The feature encoder to instrument, modified in place.
        feature_nodes: Dotted submodule paths as produced by ``nnx.iter_modules``
            (for example ``"layers.0"`` inside an ``nnx.Sequential``).

    Returns:
        The capture wrappers, in the order the names were given.

    Raises:
        ValueError: If a name does not address a submodule of ``encoder``.
    """
    paths = {_module_path(path): path for path, _ in nnx.iter_modules(encoder) if path}
    # Resolve every container up front: wrapping a module rebinds it in its parent,
    # which would invalidate paths resolved afterwards.
    targets = []
    for name in feature_nodes:
        path = paths.get(name)
        if path is None:
            available = ", ".join(sorted(paths)) or "<none>"
            msg = f"Feature node {name!r} is not a submodule of the encoder. Available nodes: {available}."
            raise ValueError(msg)
        targets.append(_resolve_container(encoder, path))

    captures = nnx.List()
    for container, key in targets:
        capture = FeatureCapture(_get_child(container, key))
        _set_child(container, key, capture)
        captures.append(capture)
    return captures


# Flags that ``nnx.Module.train``/``eval`` toggle; snapshotted to restore the exact
# per-submodule state afterwards, which a single boolean (as in torch) cannot do.
_TRAINING_FLAGS = ("deterministic", "use_running_average")


@contextmanager
def _eval_mode(module: nnx.Module) -> Iterator[None]:
    """Put ``module`` in evaluation mode, restoring every submodule flag on exit."""
    snapshot = [
        (submodule, {name: getattr(submodule, name) for name in _TRAINING_FLAGS if hasattr(submodule, name)})
        for _, submodule in nnx.iter_modules(module)
    ]
    module.eval()
    try:
        yield
    finally:
        for submodule, flags in snapshot:
            for name, value in flags.items():
                setattr(submodule, name, value)


class _PredictorMeta(type(nnx.Module), type(MahalanobisPredictor)):
    """Reconcile the ``nnx`` pytree metaclass with the flextype protocol metaclass."""


@mahalanobis_generator.register(nnx.Module)
class FlaxMahalanobisPredictor(
    nnx.Module,
    MahalanobisPredictor[[jax.Array], JaxMahalanobisRepresentation],
    metaclass=_PredictorMeta,
):
    """Flax Mahalanobis OOD predictor.

    The final ``nnx.Linear`` head is replaced with :class:`probly.layers.flax.Identity`
    to expose the penultimate features as the encoder output; the original head is
    kept for classification. One :class:`~probly.layers.flax.MahalanobisHead` is
    fitted per feature layer (any user-provided intermediate modules plus the
    penultimate features), and the per-layer Mahalanobis confidences are combined
    into a single OOD score.

    After training, call ``fit_mahalanobis_heads(features, labels)`` to estimate
    the Gaussian parameters; optionally call ``fit_combiner(id, ood)`` to
    calibrate the multi-layer combination weights on in- vs out-of-distribution
    data.

    Attributes:
        encoder: The feature encoder (head replaced with an identity module).
        classification_head: The original final Linear layer.
        mahalanobis_heads: One Mahalanobis head per feature layer (populated by
            ``fit_mahalanobis_heads``).
        combiner_weight: Per-layer combination weights of shape ``(num_layers,)``,
            initialized to ``-1`` and calibrated by ``fit_combiner``.
        combiner_bias: Scalar combination bias, initialized to ``0`` and
            calibrated by ``fit_combiner``.
    """

    encoder: nnx.Module
    classification_head: nnx.Linear
    mahalanobis_heads: nnx.List
    combiner_weight: jax.Array
    combiner_bias: jax.Array

    def __init__(
        self,
        model: nnx.Module,
        feature_nodes: Sequence[str] | None = None,
        input_preprocessing_eps: float = 0.0,
    ) -> None:
        """Build the Mahalanobis predictor from a base classifier.

        Args:
            model: Base classification model to be transformed.
            feature_nodes: Optional dotted paths of intermediate submodules (as
                produced by ``nnx.iter_modules``, e.g. ``"layers.0"``) whose
                outputs provide additional feature layers. Their features are
                returned in the order given here. When ``None`` only the
                penultimate features are used.
            input_preprocessing_eps: Magnitude of the FGSM-style input
                perturbation applied at inference. ``0`` disables it.

        Raises:
            ValueError: If the model contains no ``nnx.Linear`` layer to use as
                the classification head.
        """
        super().__init__()
        encoder, state = traverse_with_state(
            model,
            nn_compose(head_strip_traverser, nn_traverser=nn_traverser),
            init={HEAD_MODULE: None, TRAVERSE_REVERSED: True},
        )
        head: nnx.Linear | None = state[HEAD_MODULE]  # ty:ignore[invalid-assignment]
        if head is None:
            msg = "No nnx.Linear layer found in the model; cannot identify a classification head."
            raise ValueError(msg)

        self.encoder = encoder
        self.classification_head = head
        self.input_preprocessing_eps = input_preprocessing_eps
        self._num_classes = head.out_features

        self._feature_nodes = list(feature_nodes) if feature_nodes is not None else []
        self.mahalanobis_heads = nnx.List()
        num_layers = len(self._feature_nodes) + 1
        # Default combiner: negated sum of per-layer confidences (high score => out-of-distribution).
        self.combiner_weight = -jnp.ones(num_layers)
        self.combiner_bias = jnp.zeros(())

        # Tap the requested intermediate nodes on the rebuilt encoder (the
        # traversal preserves submodule names but rebuilds the module objects).
        self._captures = _wrap_feature_nodes(self.encoder, self._feature_nodes)

    @staticmethod
    def _pool(features: jax.Array) -> jax.Array:
        """Global-average-pool the spatial axes of a feature map to shape ``(N, C)``.

        Flax convolutions are channels-last, so the axes to reduce are the ones
        between the batch and channel axis (unlike torch, where they trail).
        """
        if features.ndim <= 2:
            return features
        return features.mean(axis=tuple(range(1, features.ndim - 1)))

    def _forward_features(self, x: jax.Array) -> tuple[jax.Array, list[jax.Array]]:
        """Run the encoder and collect the pooled intermediate plus penultimate features."""
        penultimate = self.encoder(x)
        feats = [self._pool(capture.captured.get_value()) for capture in self._captures]
        feats.append(self._pool(penultimate))
        return penultimate, feats

    def _layer_scores(self, feats: list[jax.Array]) -> jax.Array:
        """Per-layer Mahalanobis confidences (max over classes) stacked to ``(N, num_layers)``."""
        scores = [head(feat).max(axis=-1) for head, feat in zip(self.mahalanobis_heads, feats, strict=True)]
        return jnp.stack(scores, axis=-1)

    def _confidence_gradient(self, x: jax.Array, layer_index: int) -> jax.Array:
        """Gradient of one layer's Mahalanobis confidence with respect to the input.

        The model is split and re-merged inside the differentiated function so
        that the tracers produced by ``jax.grad`` land on a throwaway copy
        instead of leaking into this predictor's captured intermediates.

        Args:
            x: Raw input array.
            layer_index: Index of the feature layer whose confidence is raised.

        Returns:
            The gradient, of the same shape as ``x``.
        """
        graphdef, state = nnx.split(self)

        def confidence(inputs: jax.Array) -> jax.Array:
            model = nnx.merge(graphdef, state)
            _, feats = model._forward_features(inputs)  # noqa: SLF001
            return model.mahalanobis_heads[layer_index](feats[layer_index]).max(axis=-1).sum()

        return jax.grad(confidence)(x)

    def _preprocessed_layer_scores(self, x: jax.Array) -> jax.Array:
        """Per-layer scores after the FGSM-style input preprocessing of Lee et al. 2018.

        For each feature layer the input is independently nudged along the
        gradient that raises that layer's Mahalanobis confidence
        (``x + eps * sign(grad)``), then re-encoded and re-scored. This widens the
        gap between in- and out-of-distribution inputs.

        Args:
            x: Raw input array.

        Returns:
            Per-layer scores of shape ``(N, num_layers)``.
        """
        scores = []
        for layer_index, head in enumerate(self.mahalanobis_heads):
            gradient = self._confidence_gradient(x, layer_index)
            # Move the input to raise its confidence, then re-score.
            x_perturbed = x + self.input_preprocessing_eps * jnp.sign(gradient)
            _, feats_perturbed = self._forward_features(x_perturbed)
            scores.append(head(feats_perturbed[layer_index]).max(axis=-1))
        return jnp.stack(scores, axis=-1)

    def _input_layer_scores(self, x: jax.Array, feats: list[jax.Array]) -> jax.Array:
        """Per-layer Mahalanobis scores for the raw input ``x``.

        Applies the FGSM-style input preprocessing when
        ``input_preprocessing_eps`` is positive (recomputing the features from
        ``x``); otherwise returns the plain confidences from the already-extracted
        ``feats``. Both calibration (``fit_combiner``) and inference
        (``predict_representation``) route through this method so the combiner is
        trained on the same scores it is later applied to, as in Lee et al. 2018.

        Args:
            x: Raw input array.
            feats: Plain features already extracted from ``x``; used only when
                preprocessing is disabled.

        Returns:
            Per-layer scores of shape ``(N, num_layers)``.
        """
        if self.input_preprocessing_eps > 0:
            return self._preprocessed_layer_scores(x)
        return self._layer_scores(feats)

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Encode features, classify, and score the raw per-layer Mahalanobis confidence.

        The returned scores are the plain confidences without FGSM input
        preprocessing; the preprocessing-aware scores used for OOD detection are
        produced by :meth:`predict_representation`.

        Args:
            x: Input array passed to the encoder.

        Returns:
            A 2-tuple of ``(logits, layer_scores)`` of shapes ``(N, num_classes)``
            and ``(N, num_layers)``.
        """
        penultimate, feats = self._forward_features(x)
        logits = self.classification_head(penultimate)
        return logits, self._layer_scores(feats)

    def fit_mahalanobis_heads(self, features: jax.Array, labels: jax.Array) -> None:
        """Fit one Mahalanobis head per feature layer on the given inputs.

        Args:
            features: Input array (e.g. the training inputs) fed to the encoder.
            labels: Integer class labels of shape ``(N,)``.
        """
        with _eval_mode(self.encoder):
            _, feats = self._forward_features(features)
        heads = [MahalanobisHead(self._num_classes, feat.shape[-1]) for feat in feats]
        for head, feat in zip(heads, feats, strict=True):
            head.fit(feat, labels)
        self.mahalanobis_heads = nnx.List(heads)

    def fit_combiner(
        self,
        id_features: jax.Array,
        ood_features: jax.Array,
        steps: int = 1000,
        lr: float = 0.05,
    ) -> None:
        """Calibrate the multi-layer combination weights by logistic regression.

        Fits weights and a bias over the per-layer Mahalanobis scores so that
        out-of-distribution inputs (label 1) score higher than in-distribution
        inputs (label 0), reproducing the feature-ensemble step of
        :cite:`leeSimpleUnifiedFramework2018` with an optax logistic regression.

        Args:
            id_features: In-distribution inputs fed to the encoder.
            ood_features: Out-of-distribution inputs fed to the encoder.
            steps: Number of optimisation steps.
            lr: Learning rate for the Adam optimiser.
        """
        with _eval_mode(self.encoder):
            _, id_feats = self._forward_features(id_features)
            _, ood_feats = self._forward_features(ood_features)
            # Score through the same path as inference (FGSM preprocessing included
            # when enabled) so the combiner is calibrated on what it later sees.
            id_scores = self._input_layer_scores(id_features, id_feats)
            ood_scores = self._input_layer_scores(ood_features, ood_feats)

        scores = jnp.concat([id_scores, ood_scores], axis=0)
        targets = jnp.concat([jnp.zeros(len(id_scores)), jnp.ones(len(ood_scores))])

        def loss(params: tuple[jax.Array, jax.Array]) -> jax.Array:
            weight, bias = params
            return optax.sigmoid_binary_cross_entropy(scores @ weight + bias, targets).mean()

        optimizer = optax.adam(lr)
        params = (jnp.zeros(scores.shape[-1]), jnp.zeros(()))
        opt_state = optimizer.init(params)

        @jax.jit
        def step(
            params: tuple[jax.Array, jax.Array],
            opt_state: optax.OptState,
        ) -> tuple[tuple[jax.Array, jax.Array], optax.OptState]:
            updates, opt_state = optimizer.update(jax.grad(loss)(params), opt_state, params)
            # apply_updates is typed over the generic optax.Params pytree.
            return cast("tuple[jax.Array, jax.Array]", optax.apply_updates(params, updates)), opt_state

        for _ in range(steps):
            params, opt_state = step(params, opt_state)

        self.combiner_weight, self.combiner_bias = params

    def predict_representation(self, x: jax.Array) -> JaxMahalanobisRepresentation:
        """Predict the Mahalanobis representation (softmax and per-layer scores)."""
        penultimate, feats = self._forward_features(x)
        logits = self.classification_head(penultimate)
        layer_scores = self._input_layer_scores(x, feats)

        return JaxMahalanobisRepresentation(
            JaxProbabilityCategoricalDistribution(jax.nn.softmax(logits, axis=-1)),
            layer_scores,
            self.combiner_weight,
            self.combiner_bias,
        )
