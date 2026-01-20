"""Test for implementation of sklearn models."""

from __future__ import annotations

import pytest

from probly.predictor import Predictor
from probly.transformation import ensemble
from probly.transformation.ensemble.common import ensemble_generator

pytest.importorskip("sklearn")
from sklearn.base import BaseEstimator


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
            ensemble_generator(dummy_predictor)

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
