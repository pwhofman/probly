"""Test for implementation of sklearn models."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from probly.method.ensemble import EnsemblePredictor, ensemble, ensemble_generator
from probly.predictor import Predictor, predict

pytest.importorskip("sklearn")
from sklearn.base import BaseEstimator
from sklearn.compose import TransformedTargetRegressor, make_column_transformer
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Estimators that hold seeded estimators inside them, with the keys of every random_state they expose,
# written out by hand.
NESTED_RANDOM_STATES: dict[str, tuple[Callable[[], BaseEstimator], set[str]]] = {
    "pipeline": (
        lambda: make_pipeline(StandardScaler(), MLPClassifier(random_state=0)),
        {"mlpclassifier__random_state"},
    ),
    "pipeline_with_two_random_steps": (
        lambda: make_pipeline(
            PCA(n_components=2, svd_solver="randomized", random_state=0),
            RandomForestClassifier(n_estimators=3, random_state=0),
        ),
        {"pca__random_state", "randomforestclassifier__random_state"},
    ),
    "meta_estimator": (
        lambda: BaggingClassifier(estimator=DecisionTreeClassifier(random_state=0), n_estimators=3, random_state=0),
        {"random_state", "estimator__random_state"},
    ),
    "meta_estimator_in_pipeline": (
        lambda: make_pipeline(
            StandardScaler(),
            BaggingClassifier(estimator=DecisionTreeClassifier(random_state=0), n_estimators=3, random_state=0),
        ),
        {"baggingclassifier__random_state", "baggingclassifier__estimator__random_state"},
    ),
    "column_transformer": (
        lambda: make_pipeline(
            make_column_transformer(
                (PCA(n_components=1, svd_solver="randomized", random_state=0), [0, 1]), remainder="passthrough"
            ),
            DecisionTreeClassifier(random_state=0),
        ),
        {"columntransformer__pca__random_state", "decisiontreeclassifier__random_state"},
    ),
    "target_transformer": (
        lambda: TransformedTargetRegressor(regressor=MLPRegressor(random_state=0)),
        {"regressor__random_state"},
    ),
}


class TestModelGeneration:
    """Tests the correct generation of sklearn ensembles."""

    @pytest.mark.parametrize(
        "model_fixture",
        [
            "sklearn_logistic_regression",
            "sklearn_mlp_regressor_2d_1d",
            "sklearn_mlp_classifier_2d_2d",
            "sklearn_sgd_classifier",
            "sklearn_sgd_regressor",
            "sklearn_svc",
            "sklearn_svr",
        ],
    )
    def test_sklearn_model_gen(self, model_fixture: str, request: pytest.FixtureRequest) -> None:
        """Tests that ensemble generation is correct for given fixtures."""
        num_members = 4
        model = request.getfixturevalue(model_fixture)

        ensemble_list = ensemble(model, num_members=num_members, reset_params=False)

        assert type(ensemble_list) is list
        assert isinstance(ensemble_list, EnsemblePredictor)
        assert len(ensemble_list) == num_members

        for member in ensemble_list:
            assert isinstance(member, type(model))

    def test_unregistered_type_raises(self, dummy_predictor: Predictor) -> None:
        """No ensemble generator is registered for type, NotImplementedError must occur."""
        base = dummy_predictor
        with pytest.raises(
            NotImplementedError,
            match=f"No ensemble generator is registered for type {type(base)}",
        ):
            ensemble_generator(dummy_predictor, num_members=4)

    @pytest.mark.parametrize(
        "model_fixture",
        [
            "sklearn_logistic_regression",
            "sklearn_mlp_regressor_2d_1d",
            "sklearn_mlp_classifier_2d_2d",
            "sklearn_sgd_classifier",
            "sklearn_sgd_regressor",
            "sklearn_svc",
            "sklearn_svr",
        ],
    )
    def test_list_ensemble_independence(self, model_fixture: str, request: pytest.FixtureRequest) -> None:
        """Assures that models in list are independent."""
        model = request.getfixturevalue(model_fixture)
        ensemble_list = ensemble(model, num_members=3, reset_params=False)

        assert ensemble_list[0] is not ensemble_list[1]
        assert ensemble_list[1] is not ensemble_list[2]


class TestResetParams:
    """Tests that the random_state parameter is correctly reset or not."""

    @pytest.mark.parametrize(
        "model_fixture",
        [
            "sklearn_logistic_regression",
            "sklearn_mlp_regressor_2d_1d",
            "sklearn_mlp_classifier_2d_2d",
            "sklearn_sgd_classifier",
            "sklearn_sgd_regressor",
            "sklearn_svc",
            "sklearn_svr",
        ],
    )
    def test_reset_params(self, model_fixture: str, request: pytest.FixtureRequest) -> None:
        """Tests that the random_state parameter is reset when requested."""
        num_members = 3
        model = request.getfixturevalue(model_fixture)
        sklearn_ensemble = ensemble(model, num_members=num_members, reset_params=True)

        assert len(sklearn_ensemble) == num_members
        for member in sklearn_ensemble:
            assert isinstance(member, BaseEstimator)
            assert member.__getattribute__("random_state") is None

    @pytest.mark.parametrize(
        "model_fixture",
        [
            "sklearn_logistic_regression",
            "sklearn_mlp_regressor_2d_1d",
            "sklearn_mlp_classifier_2d_2d",
            "sklearn_sgd_classifier",
            "sklearn_sgd_regressor",
            "sklearn_svc",
            "sklearn_svr",
        ],
    )
    def test_no_reset_params(self, model_fixture: str, request: pytest.FixtureRequest) -> None:
        """Tests that the random_state parameter is not reset when not requested."""
        num_members = 3
        model = request.getfixturevalue(model_fixture)
        sklearn_ensemble = ensemble(model, num_members=num_members, reset_params=False)

        assert len(sklearn_ensemble) == num_members
        for member in sklearn_ensemble:
            assert isinstance(member, BaseEstimator)
            assert member.__getattribute__("random_state") == model.__getattribute__("random_state")


class TestFitandPredict:
    """Tests that ensembles of sklearn models can be fitted and used for prediction."""

    @pytest.mark.parametrize(
        ("model_fixture", "X", "y"),
        [
            ("sklearn_logistic_regression", [[0, 0], [1, 1], [1, 0], [0, 1]], [0, 1, 1, 0]),
            ("sklearn_mlp_regressor_2d_1d", [[0, 0], [1, 1], [2, 2], [3, 3]], [0.0, 1.0, 2.0, 3.0]),
            ("sklearn_mlp_classifier_2d_2d", [[0, 0], [1, 1], [0, 1], [1, 0]], [0, 1, 0, 1]),
            ("sklearn_sgd_classifier", [[-2, -1], [-1, -1], [1, 1], [2, 1]], [0, 0, 1, 1]),
            ("sklearn_sgd_regressor", [[0, 0], [1, 0], [2, 0], [3, 0]], [0.0, 1.0, 2.0, 3.0]),
            ("sklearn_svc", [[0, 0], [1, 1], [1, 0], [0, 1]], [0, 1, 1, 0]),
            ("sklearn_svr", [[0, 0], [1, 1], [2, 2], [3, 3]], [0.0, 1.0, 2.0, 3.0]),
        ],
    )
    def test_fit_and_predict(self, model_fixture: str, X: list, y: list, request: pytest.FixtureRequest) -> None:
        """Tests that the ensemble can be fitted and used for prediction."""
        num_members = 5
        model = request.getfixturevalue(model_fixture)
        sklearn_ensemble = ensemble(model, num_members=num_members, reset_params=False)

        for member in sklearn_ensemble:
            member.fit(X, y)
        predictions = [member.predict(X) for member in sklearn_ensemble]
        for pred in predictions:
            assert len(pred) == len(y)

    @pytest.mark.parametrize(
        ("model_fixture", "X", "y"),
        [
            ("sklearn_mlp_regressor_2d_1d", [[0, 0], [1, 1], [2, 2], [3, 3]], [0.0, 1.0, 2.0, 3.0]),
        ],
    )
    def test_fit_and_predict_resetparams(
        self,
        model_fixture: str,
        X: list,
        y: list,
        request: pytest.FixtureRequest,
    ) -> None:
        """Tests that the ensemble can be fitted and used for prediction when reset_params is True."""
        num_members = 5
        model = request.getfixturevalue(model_fixture)
        sklearn_ensemble = ensemble(model, num_members=num_members, reset_params=True)

        for member in sklearn_ensemble:
            member.fit(X, y)
        predictions = [member.predict(X) for member in sklearn_ensemble]
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                assert not all(predictions[i] == predictions[j])

    @pytest.mark.parametrize(
        ("model_fixture", "X", "y"),
        [
            ("sklearn_logistic_regression", [[0, 0], [1, 1], [1, 0], [0, 1]], [0, 1, 1, 0]),
            ("sklearn_mlp_regressor_2d_1d", [[0, 0], [1, 1], [2, 2], [3, 3]], [0.0, 1.0, 2.0, 3.0]),
            ("sklearn_mlp_classifier_2d_2d", [[0, 0], [1, 1], [0, 1], [1, 0]], [0, 1, 0, 1]),
            ("sklearn_sgd_classifier", [[-2, -1], [-1, -1], [1, 1], [2, 1]], [0, 0, 1, 1]),
            ("sklearn_sgd_regressor", [[0, 0], [1, 0], [2, 0], [3, 0]], [0.0, 1.0, 2.0, 3.0]),
            ("sklearn_svc", [[0, 0], [1, 1], [1, 0], [0, 1]], [0, 1, 1, 0]),
            ("sklearn_svr", [[0, 0], [1, 1], [2, 2], [3, 3]], [0.0, 1.0, 2.0, 3.0]),
        ],
    )
    def test_fit_and_predict_not_reset_params(
        self,
        model_fixture: str,
        X: list,
        y: list,
        request: pytest.FixtureRequest,
    ) -> None:
        """Tests that the ensemble can be fitted and used for prediction when reset_params is False."""
        num_members = 5
        model = request.getfixturevalue(model_fixture)
        sklearn_ensemble = ensemble(model, num_members=num_members, reset_params=False)

        for member in sklearn_ensemble:
            member.fit(X, y)
        predictions = [member.predict(X) for member in sklearn_ensemble]
        for pred in predictions:
            assert all(pred == predictions[0])


def test_ensemble_does_not_mutate_base_estimator() -> None:
    """The caller's estimator keeps its random_state; only the members are reset."""
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

    base = LogisticRegression(random_state=42)

    reset_members = ensemble(base, num_members=3, reset_params=True)
    assert base.random_state == 42
    assert all(member.random_state is None for member in reset_members)

    kept_members = ensemble(base, num_members=2, reset_params=False)
    assert base.random_state == 42
    assert all(member.random_state == 42 for member in kept_members)


def test_ensemble_of_unfitted_sklearn_ensemble() -> None:
    """Sklearn ensembles are only iterable once fitted; wrapping them must not iterate them."""
    from sklearn.ensemble import RandomForestClassifier  # noqa: PLC0415

    base = RandomForestClassifier(n_estimators=5, random_state=1)
    members = ensemble(base, num_members=3)

    assert base.random_state == 1
    assert len(members) == 3
    assert all(isinstance(member, RandomForestClassifier) for member in members)
    assert all(member.n_estimators == 5 for member in members)
    # reset_params=True by default: each member draws its own randomness when fitted.
    assert all(member.random_state is None for member in members)


class TestNestedRandomStates:
    """reset_params handles the random_state of every estimator nested in a pipeline or meta-estimator."""

    @pytest.mark.parametrize("name", list(NESTED_RANDOM_STATES))
    def test_reset_clears_every_random_state(self, name: str) -> None:
        make_base, keys = NESTED_RANDOM_STATES[name]
        base = make_base()

        members = ensemble(base, num_members=3, reset_params=True)

        for member in members:
            params = member.get_params(deep=True)
            assert {key for key in keys if params[key] is not None} == set()
            # Nothing but the random states changed: with the seeds put back, the member is the base.
            assert repr(member.set_params(**dict.fromkeys(keys, 0))) == repr(base)

    @pytest.mark.parametrize("name", list(NESTED_RANDOM_STATES))
    def test_reset_leaves_the_base_estimator_seeded(self, name: str) -> None:
        make_base, keys = NESTED_RANDOM_STATES[name]
        base = make_base()

        ensemble(base, num_members=3, reset_params=True)

        params = base.get_params(deep=True)
        assert all(params[key] == 0 for key in keys)

    @pytest.mark.parametrize("name", list(NESTED_RANDOM_STATES))
    def test_no_reset_keeps_every_random_state(self, name: str) -> None:
        make_base, keys = NESTED_RANDOM_STATES[name]

        members = ensemble(make_base(), num_members=2, reset_params=False)

        for member in members:
            params = member.get_params(deep=True)
            assert all(params[key] == 0 for key in keys)

    @pytest.mark.parametrize(
        "make_base",
        [
            lambda: make_pipeline(
                StandardScaler(), MLPClassifier(hidden_layer_sizes=(8,), max_iter=2000, random_state=0)
            ),
            lambda: make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=5, random_state=0)),
            lambda: BaggingClassifier(estimator=DecisionTreeClassifier(random_state=0), n_estimators=3, random_state=0),
        ],
        ids=["mlp_pipeline", "forest_pipeline", "bagging"],
    )
    def test_reset_members_predict_differently(self, make_base: Callable[[], BaseEstimator]) -> None:
        """Every member of a seeded pipeline used to be fitted to the same model.

        The members are compared on fresh points: on their training points, fully grown trees are
        mostly certain, so two differently seeded members can agree there by chance.
        """
        x, y = make_classification(n_samples=100, n_features=5, random_state=0)
        x_new, _ = make_classification(n_samples=500, n_features=5, random_state=1)
        members = ensemble(make_base(), num_members=3, reset_params=True)

        for member in members:
            member.fit(x, y)
        probabilities = [np.asarray(distribution.probabilities) for distribution in predict(members, x_new)]

        for i in range(len(probabilities)):
            for j in range(i + 1, len(probabilities)):
                assert not np.allclose(probabilities[i], probabilities[j]), (i, j)

    def test_members_without_reset_predict_the_same(self) -> None:
        x, y = make_classification(n_samples=100, n_features=5, random_state=0)
        base = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=5, random_state=0))
        members = ensemble(base, num_members=3, reset_params=False)

        for member in members:
            member.fit(x, y)
        x_new, _ = make_classification(n_samples=500, n_features=5, random_state=1)
        probabilities = [np.asarray(distribution.probabilities) for distribution in predict(members, x_new)]

        assert all(np.array_equal(probabilities[0], p) for p in probabilities[1:])

    def test_estimator_without_random_state(self) -> None:
        base = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3))

        members = ensemble(base, num_members=2, reset_params=True)

        assert all(repr(member) == repr(base) for member in members)
        assert members[0] is not members[1]
