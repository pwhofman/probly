"""Per-method model construction for benchmarks.

Most methods are built by :func:`_default_builder`, which just calls
``method_fn(base, **params)``. Methods that need values derived from the
dataset or from the base model (e.g. Posterior Network, which needs
``class_counts`` from the training set and ``dim`` from the encoder)
register a custom builder in :data:`BUILDERS`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import inspect
import logging
from typing import TYPE_CHECKING, Any
import warnings

import torch
from torch import nn
from torch.utils.data import Subset

from probly.method.bayesian import bayesian
from probly.method.credal_ensembling import credal_ensembling
from probly.method.credal_relative_likelihood import credal_relative_likelihood
from probly.method.credal_wrapper import credal_wrapper
from probly.method.ddu import ddu
from probly.method.dropconnect import dropconnect
from probly.method.dropout import dropout
from probly.method.efficient_credal_prediction import efficient_credal_prediction
from probly.method.ensemble import ensemble
from probly.method.posterior_network import posterior_network
from probly.method.subensemble import subensemble
from probly_benchmark import models

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


logger = logging.getLogger(__name__)

METHODS = {
    "bayesian": bayesian,
    "ddu": ddu,
    "dropout": dropout,
    "dropconnect": dropconnect,
    "posterior_network": posterior_network,
    "ensemble": ensemble,
    "credal_ensembling": credal_ensembling,
    "credal_relative_likelihood": credal_relative_likelihood,
    "credal_wrapper": credal_wrapper,
    "efficient_credal_prediction": efficient_credal_prediction,
    "subensemble": subensemble,
}


def get_method(name: str) -> Callable[..., nn.Module]:
    """Look up a method transformation callable by name.

    Args:
        name: Method name as it appears in the ``method.name`` config field.

    Raises:
        ValueError: If the method name is not registered in :data:`METHODS`.
    """
    key = name.lower()
    if key not in METHODS:
        msg = f"Unknown method: {name}"
        raise ValueError(msg)
    return METHODS[key]  # ty: ignore


@dataclass
class BuildContext:
    """Context passed to a method builder.

    Carries everything a builder may need that is not in ``cfg.method.params``.
    ``train_loader`` is optional so that :func:`build_model` can also be used
    at checkpoint-load time, where no training loader is available and any
    data-derived buffers will be overwritten by ``load_state_dict`` anyway.

    Attributes:
        base_model_name: Name of the base model requested in the config.
        num_classes: Number of classes for the dataset being used.
        pretrained: Whether to load pretrained weights for the base model.
        train_loader: Training loader, or ``None`` when building for load.
    """

    base_model_name: str
    model_type: str
    num_classes: int
    pretrained: bool
    train_loader: DataLoader | None = None


Builder = Callable[[Callable[..., nn.Module], dict[str, Any], BuildContext], nn.Module]


def _filter_params(fn: Callable[..., Any], params: dict[str, Any]) -> dict[str, Any]:
    """Return only the kwargs accepted by ``fn``, warning about dropped keys."""
    sig = inspect.signature(fn)
    accepts_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if accepts_var_keyword:
        return params
    accepted = set(sig.parameters)
    dropped = {k for k in params if k not in accepted}
    if dropped:
        warnings.warn(
            f"{fn.__name__} does not accept {dropped}; dropping from method params. "  # ty:ignore[unresolved-attribute]
            "Check your recipe/method config for unintended overrides.",
            stacklevel=3,
        )
    return {k: v for k, v in params.items() if k in accepted}


def _default_builder(
    method_fn: Callable[..., nn.Module],
    params: dict[str, Any],
    ctx: BuildContext,
) -> nn.Module:
    """Build a model using only the YAML hyperparameters and a base network."""
    base = models.get_base_model(ctx.base_model_name, ctx.num_classes, ctx.pretrained)
    return method_fn(base, predictor_type=ctx.model_type, **_filter_params(method_fn, params))


def _posterior_network_builder(
    method_fn: Callable[..., nn.Module],
    params: dict[str, Any],
    ctx: BuildContext,
) -> nn.Module:
    """Build a Posterior Network.

    Requires three things that do not belong in the YAML:

    - An encoder that outputs features instead of class logits. We ask
      ``get_base_model`` for the ``<name>_encoder`` variant.
    - The feature dimension, exposed as ``encoder.feature_dim`` by the
      ``*_encoder`` branches of ``get_base_model``.
    - Per-class training-set counts, computed by :func:`_class_counts`.

    When ``ctx.train_loader`` is ``None`` (e.g. at checkpoint load time) a
    uniform placeholder is used for ``class_counts``; the real values are
    restored by ``load_state_dict`` since ``class_counts`` is a buffer on
    the underlying module.
    """
    encoder = models.get_base_model(
        f"{ctx.base_model_name}_encoder",
        ctx.num_classes,
        ctx.pretrained,
    )
    if ctx.train_loader is not None:
        class_counts = _class_counts(ctx.train_loader, ctx.num_classes)
    else:
        class_counts = [1] * ctx.num_classes
    return method_fn(
        encoder,
        num_classes=ctx.num_classes,
        class_counts=class_counts,
        predictor_type=ctx.model_type,
        **_filter_params(method_fn, params),
    )


BUILDERS: dict[str, Builder] = {
    "posterior_network": _posterior_network_builder,
}


def build_model(name: str, params: dict[str, Any], ctx: BuildContext) -> nn.Module:
    """Build the method model for ``name`` using its registered builder.

    Looks up the method callable in :data:`METHODS` and the builder in
    :data:`BUILDERS`, falling back to :func:`_default_builder` for methods
    that do not need special construction.
    """
    method_fn = get_method(name)
    builder = BUILDERS.get(name.lower(), _default_builder)
    return builder(method_fn, params, ctx)


def _class_counts(loader: DataLoader, num_classes: int) -> list[int]:
    """Return per-class sample counts for the dataset behind ``loader``.

    Unwraps ``torch.utils.data.Subset`` layers (so the validation-enabled
    path in :func:`probly_benchmark.data.get_data_train` is handled), and
    tolerates the shapes that ``dataset.targets`` can take across torchvision
    datasets (tensor, numpy array, Python list). Falls back to iterating
    labels from the loader when no ``.targets`` attribute is exposed.
    """
    ds: Any = loader.dataset
    indices: list[int] | None = None
    while isinstance(ds, Subset):
        indices = list(ds.indices) if indices is None else [ds.indices[i] for i in indices]
        ds = ds.dataset

    targets = getattr(ds, "targets", None)
    if targets is None:
        counts = [0] * num_classes
        for _, y in loader:
            for c in y.tolist():
                counts[c] += 1
        return counts

    if isinstance(targets, torch.Tensor) or hasattr(targets, "tolist"):
        targets = targets.tolist()

    if indices is not None:
        targets = [targets[i] for i in indices]

    counts = [0] * num_classes
    for c in targets:
        counts[c] += 1
    return counts
