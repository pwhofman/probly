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


from probly.representation.credal_set.torch import (  # noqa: E402  # ty: ignore[unresolved-import]
    TorchConvexCredalSet,
    TorchProbabilityIntervalsCredalSet,
)
from probly_benchmark.al_estimator import _probabilities_from_representation  # noqa: E402


def test_probabilities_from_categorical() -> None:
    rep = TorchCategoricalDistribution(torch.tensor([[0.1, 0.6, 0.3], [0.5, 0.2, 0.3]]))
    probs = _probabilities_from_representation(rep)
    assert torch.allclose(probs, rep.probabilities)


def test_probabilities_from_sample() -> None:
    sample_tensor = torch.tensor(
        [
            [[0.2, 0.5, 0.3], [0.4, 0.4, 0.2]],
            [[0.4, 0.3, 0.3], [0.6, 0.2, 0.2]],
        ]
    )  # (num_samples=2, batch=2, classes=3)
    rep = TorchSample(
        tensor=TorchCategoricalDistribution(sample_tensor),
        sample_dim=0,
    )
    probs = _probabilities_from_representation(rep)
    expected = sample_tensor.mean(dim=0)
    assert torch.allclose(probs, expected)


def test_probabilities_from_intervals_credal_set() -> None:
    rep = TorchProbabilityIntervalsCredalSet(
        lower_bounds=torch.tensor([[0.1, 0.2, 0.3]]),
        upper_bounds=torch.tensor([[0.3, 0.4, 0.5]]),
    )
    probs = _probabilities_from_representation(rep)
    expected = torch.tensor([[0.2, 0.3, 0.4]])  # midpoints
    assert torch.allclose(probs, expected)


def test_probabilities_from_convex_credal_set() -> None:
    vertices = torch.tensor(
        [
            [[0.1, 0.6, 0.3], [0.3, 0.4, 0.3], [0.5, 0.2, 0.3]],
        ]
    )  # (batch=1, n_vertices=3, classes=3)
    rep = TorchConvexCredalSet(tensor=TorchCategoricalDistribution(vertices))
    probs = _probabilities_from_representation(rep)
    expected = vertices.mean(dim=-2)
    assert torch.allclose(probs, expected)


def test_probabilities_from_unknown_raises() -> None:
    class Unknown:
        pass

    with pytest.raises(NotImplementedError, match="No probability extraction"):
        _probabilities_from_representation(Unknown())


from probly_benchmark.al_estimator import _MEASURES, _default_measure_for  # noqa: E402


def test_default_measure_for_categorical() -> None:
    rep = TorchCategoricalDistribution(torch.ones(2, 3) / 3)
    assert _default_measure_for(rep) == "entropy"


def test_default_measure_for_sample() -> None:
    rep = TorchSample(
        tensor=TorchCategoricalDistribution(torch.ones(2, 1, 3) / 3),
        sample_dim=0,
    )
    assert _default_measure_for(rep) == "entropy_of_expected_predictive_distribution"


def test_default_measure_for_intervals() -> None:
    rep = TorchProbabilityIntervalsCredalSet(
        lower_bounds=torch.tensor([[0.1, 0.2, 0.3]]),
        upper_bounds=torch.tensor([[0.3, 0.4, 0.5]]),
    )
    assert _default_measure_for(rep) == "upper_entropy"


def test_default_measure_for_convex() -> None:
    rep = TorchConvexCredalSet(
        tensor=TorchCategoricalDistribution(torch.ones(1, 2, 3) / 3),
    )
    assert _default_measure_for(rep) == "upper_entropy"


def test_default_measure_for_unknown_raises() -> None:
    class Unknown:
        pass

    with pytest.raises(NotImplementedError, match="No default measure"):
        _default_measure_for(Unknown())


def test_measures_table_has_expected_entries() -> None:
    expected = {
        "entropy",
        "entropy_of_expected_predictive_distribution",
        "mutual_information",
        "upper_entropy",
        "lower_entropy",
    }
    assert expected.issubset(_MEASURES.keys())
    for fn in _MEASURES.values():
        assert callable(fn)


def test_predict_proba_returns_correct_shape_for_dropout() -> None:
    base = _tiny_classifier(num_classes=3, in_features=4)
    pred = dropout(base, p=0.5, predictor_type="logit_classifier")
    est = _make_estimator_with_predictor(pred)
    probs = est.predict_proba(torch.zeros(5, 4))
    assert probs.shape == (5, 3)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(5), atol=1e-5)


def test_predict_proba_returns_correct_shape_for_ensemble() -> None:
    base = _tiny_classifier(num_classes=3, in_features=4)
    pred = ensemble(base, num_members=2, predictor_type="logit_classifier")
    est = _make_estimator_with_predictor(pred)  # ty: ignore[invalid-argument-type]
    probs = est.predict_proba(torch.zeros(5, 4))
    assert probs.shape == (5, 3)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(5), atol=1e-5)


def test_uncertainty_scores_dropout_default_is_eoepd() -> None:
    """Default measure for a TorchSample is entropy_of_expected_predictive_distribution."""
    base = _tiny_classifier(num_classes=3, in_features=4)
    pred = dropout(base, p=0.5, predictor_type="logit_classifier")
    est = _make_estimator_with_predictor(pred)
    scores = est.uncertainty_scores(torch.zeros(5, 4))
    assert scores.shape == (5,)
    assert (scores >= 0).all()


def test_uncertainty_scores_explicit_measure_overrides_default() -> None:
    base = _tiny_classifier(num_classes=3, in_features=4)
    pred = dropout(base, p=0.5, predictor_type="logit_classifier")
    est = _make_estimator_with_predictor(pred)
    est.measure = "mutual_information"
    scores = est.uncertainty_scores(torch.zeros(5, 4))
    assert scores.shape == (5,)
    assert (scores >= -1e-6).all()  # MI is non-negative


def test_uncertainty_scores_unknown_measure_raises() -> None:
    base = _tiny_classifier(num_classes=3, in_features=4)
    pred = dropout(base, p=0.5, predictor_type="logit_classifier")
    est = _make_estimator_with_predictor(pred)
    est.measure = "this_does_not_exist"
    with pytest.raises(KeyError):
        est.uncertainty_scores(torch.zeros(5, 4))


from probly.method.ddu import ddu  # noqa: E402  # ty: ignore[unresolved-import]


def test_embed_plain_classifier_uses_last_linear() -> None:
    """Plain ``nn.Module`` falls through to the last-Linear forward pre-hook.

    ``_embed_one`` doesn't require the model to be a registered probly
    predictor — it just installs a forward pre-hook on the last
    ``nn.Linear``. The TabularMLP's last ``Linear`` maps
    ``hidden_dim=1024`` → ``num_classes``, so the captured embedding
    has shape ``(batch, 1024)``.
    """
    base = _tiny_classifier(num_classes=3, in_features=4)
    est = _make_estimator_with_predictor(base)
    emb = est.embed(torch.zeros(5, 4))
    assert emb.shape == (5, 1024)


def test_embed_ensemble_averages_members() -> None:
    base = _tiny_classifier(num_classes=3, in_features=4)
    pred = ensemble(base, num_members=2, predictor_type="logit_classifier")
    est = _make_estimator_with_predictor(pred)  # ty: ignore[invalid-argument-type]
    emb = est.embed(torch.zeros(5, 4))
    assert emb.shape == (5, 1024)


def test_embed_ddu_uses_encoder_attribute() -> None:
    """DDU exposes an .encoder attribute; embed should run that, not the last Linear."""
    base = _tiny_classifier(num_classes=3, in_features=4)
    pred = ddu(base, predictor_type="logit_classifier")
    # DDUPredictor protocol guarantees an .encoder attribute.
    assert hasattr(pred, "encoder")
    est = _make_estimator_with_predictor(pred)  # ty: ignore[invalid-argument-type]
    emb = est.embed(torch.zeros(5, 4))
    # Just assert no crash and a 2D tensor of the right batch size.
    assert emb.ndim == 2
    assert emb.shape[0] == 5
