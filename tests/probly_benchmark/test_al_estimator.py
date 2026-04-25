"""Tests for BenchmarkALEstimator helpers and behavior."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch
from torch import nn

from probly.method.dropout import dropout
from probly.method.ensemble import ensemble
from probly.predictor import LogitClassifier  # ty: ignore[unresolved-import]
from probly.representation.distribution.torch_categorical import (  # ty: ignore[unresolved-import]
    TorchCategoricalDistribution,
)
from probly.representation.sample.torch import TorchSample  # ty: ignore[unresolved-import]
from probly_benchmark.al_estimator import BenchmarkALEstimator
from probly_benchmark.models import get_base_model


def _tiny_classifier(num_classes: int = 3, in_features: int = 4) -> nn.Module:
    """Return a small unregistered tabular MLP suitable as a base for probly methods.

    The model is *not* pre-registered as a ``LogitClassifier``; tests pass
    ``predictor_type="logit_classifier"`` to ``dropout``/``ensemble``/``ddu``
    instead, which is the intended public API.
    """
    return get_base_model("tabular_mlp", num_classes=num_classes, in_features=in_features)


def _make_estimator_with_predictor(predictor: nn.Module) -> BenchmarkALEstimator:
    """Build an estimator instance with a manually-set predictor (skip fit)."""
    from omegaconf import OmegaConf  # noqa: PLC0415

    cfg = OmegaConf.create(
        {
            "epochs": 1,
            "batch_size": 8,
            "optimizer": {"name": "adam", "params": {"lr": 0.001}},
            "scheduler": {"name": "cosine", "params": {}},
            "early_stopping": {"patience": 0},
        }
    )
    est = BenchmarkALEstimator(
        method_name="dropout",
        method_params={},
        train_kwargs={},
        cfg=cfg,
        base_model_name="tabular_mlp",
        model_type="logit_classifier",
        num_classes=3,
        device=torch.device("cpu"),
        in_features=4,
        num_samples=4,
    )
    est.model = predictor
    return est


def test_representation_dropout_returns_sample() -> None:
    """A dropout (RandomPredictor) should yield a TorchSample of CategoricalDistribution."""
    base = _tiny_classifier()
    pred = dropout(base, p=0.5, predictor_type="logit_classifier")
    est = _make_estimator_with_predictor(pred)
    rep = est._representation(torch.zeros(2, 4))  # noqa: SLF001
    assert isinstance(rep, TorchSample)
    assert isinstance(rep.tensor, TorchCategoricalDistribution)
    assert rep.sample_dim == 0
    assert rep.tensor.probabilities.shape == (4, 2, 3)  # (num_samples, batch, classes)


def test_representation_ensemble_returns_sample() -> None:
    """An ensemble (IterablePredictor) should yield a TorchSample of CategoricalDistribution."""
    base = _tiny_classifier()
    pred = ensemble(base, num_members=3, predictor_type="logit_classifier")
    est = _make_estimator_with_predictor(pred)  # ty: ignore[invalid-argument-type]
    rep = est._representation(torch.zeros(2, 4))  # noqa: SLF001
    assert isinstance(rep, TorchSample)
    assert isinstance(rep.tensor, TorchCategoricalDistribution)
    assert rep.sample_dim == 0
    assert rep.tensor.probabilities.shape == (3, 2, 3)  # (members, batch, classes)


def test_representation_plain_classifier_returns_categorical() -> None:
    """A non-stochastic, non-ensemble predictor should yield a CategoricalDistribution.

    The bare ``tabular_mlp`` is registered manually as ``LogitClassifier`` so
    ``predict()`` knows how to wrap its output.
    """
    pred = _tiny_classifier()
    LogitClassifier.register_instance(pred)
    est = _make_estimator_with_predictor(pred)
    rep = est._representation(torch.zeros(2, 4))  # noqa: SLF001
    assert isinstance(rep, TorchCategoricalDistribution)
    assert rep.probabilities.shape == (2, 3)
