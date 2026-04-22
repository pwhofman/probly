"""Sklearn serialization behavior for probly models and wrappers."""

from __future__ import annotations

import pytest

from flextype import registry_pickle
from probly.method.conformal import conformal_lac
from probly.method.ensemble import ensemble
from probly.predictor import RandomPredictor

pytest.importorskip("sklearn")


@pytest.mark.parametrize(
    "model_fixture",
    [
        "sklearn_logistic_regression",
        "sklearn_mlp_classifier_2d_2d",
        "sklearn_sgd_regressor",
    ],
)
def test_registry_pickle_roundtrip_preserves_sklearn_model_type(
    model_fixture: str,
    request: pytest.FixtureRequest,
) -> None:
    """Sklearn estimators should retain their concrete class after round-trip serialization."""
    model = request.getfixturevalue(model_fixture)

    restored = registry_pickle.loads(registry_pickle.dumps(model))

    assert type(restored) is type(model)


def test_registry_pickle_roundtrip_preserves_sklearn_ensemble_container_type(
    sklearn_logistic_regression: object,
) -> None:
    """Serialized sklearn ensembles should preserve list container and member estimator types."""
    wrapped = ensemble(sklearn_logistic_regression, num_members=3, reset_params=False)

    restored = registry_pickle.loads(registry_pickle.dumps(wrapped))

    assert type(restored) is type(wrapped)
    assert len(restored) == 3
    assert all(type(member) is type(sklearn_logistic_regression) for member in restored)


def test_registry_pickle_roundtrip_preserves_explicit_random_predictor_registration_on_sklearn_model(
    sklearn_logistic_regression: object,
) -> None:
    """Explicit RandomPredictor instance registration on sklearn models should survive round-trips."""
    model = sklearn_logistic_regression
    assert not isinstance(model, RandomPredictor)

    RandomPredictor.register_instance(model)
    assert isinstance(model, RandomPredictor)

    restored = registry_pickle.loads(registry_pickle.dumps(model))

    assert type(restored) is type(model)
    assert isinstance(restored, RandomPredictor)


def test_registry_pickle_roundtrip_preserves_explicit_random_predictor_registration_on_sklearn_ensemble_members(
    sklearn_logistic_regression: object,
) -> None:
    """Explicit RandomPredictor registration on sklearn ensemble members should survive round-trips."""
    wrapped = ensemble(
        sklearn_logistic_regression,
        num_members=3,
        reset_params=False,
        predictor_type=RandomPredictor,
    )

    assert all(isinstance(member, RandomPredictor) for member in wrapped)

    restored = registry_pickle.loads(registry_pickle.dumps(wrapped))

    assert type(restored) is type(wrapped)
    assert len(restored) == len(wrapped)
    assert all(isinstance(member, RandomPredictor) for member in restored)


def test_registry_pickle_roundtrip_for_sklearn_conformal_wrapper(
    sklearn_logistic_regression: object,
) -> None:
    """Regression test for Flexdispatch TypeVar pickling in sklearn conformal wrappers."""
    wrapped = conformal_lac(sklearn_logistic_regression)
    restored = registry_pickle.loads(registry_pickle.dumps(wrapped))
    assert type(restored) is type(wrapped)
