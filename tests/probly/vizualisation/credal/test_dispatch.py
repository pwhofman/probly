"""Test for dispatch_plot."""

from __future__ import annotations

from unittest.mock import patch

import matplotlib as mpl
import numpy as np
import pytest

mpl.use("Agg")
from probly.visualization.credal.input_handling import (
    IntervalVisualizer,
    MultiVisualizer,
    TernaryVisualizer,
    dispatch_plot,
)

data2d = np.array(
    [
        [0.5, 0.5],
        [0.2, 0.8],
    ],
)

data3d = np.array(
    [
        [0.1, 0.1, 0.8],
        [0.2, 0.2, 0.6],
        [0.2, 0.1, 0.7],
    ],
)
data4d = np.array(
    [
        [0.1, 0.1, 0.1, 0.7],
        [0.2, 0.1, 0.1, 0.6],
        [0.5, 0.1, 0.1, 0.3],
        [0.3, 0.1, 0.2, 0.4],
    ],
)


def test_checks_delegation_to_correct_visualizer() -> None:
    """Tests the delegation to the corresponding plotting function using mocking."""

    def _create_mock_class_to_check_delegation(
        class_path: str,
        method_name: str,
        return_value: str,
        data: np.ndarray,
    ) -> None:
        with patch(class_path) as mock_class:
            mock_instance = mock_class.return_value
            mocked_method = getattr(mock_instance, method_name)
            mocked_method.return_value = return_value
            ax = dispatch_plot(data)
            mock_class.assert_called_once()
            mocked_method.assert_called_once()
            assert ax == return_value

    _create_mock_class_to_check_delegation(
        "probly.visualization.credal.input_handling.IntervalVisualizer",
        "interval_plot",
        "interval_ax",
        data2d,
    )

    _create_mock_class_to_check_delegation(
        "probly.visualization.credal.input_handling.TernaryVisualizer",
        "ternary_plot",
        "ternary_ax",
        data3d,
    )

    _create_mock_class_to_check_delegation(
        "probly.visualization.credal.input_handling.MultiVisualizer",
        "spider_plot",
        "multi_ax",
        data4d,
    )


def test_dispatch_default_labels() -> None:
    """Tests that the default labels are set correctly."""

    def _create_mock_class_to_check_labels_default(
        class_path: str,
        method_name: str,
        data: np.ndarray,
        labels: list[str],
    ) -> None:
        with patch(class_path) as mock_class:
            mock_instance = mock_class.return_value
            mocked_method = getattr(mock_instance, method_name)
            dispatch_plot(data)
            _, label_arg = mocked_method.call_args
            assert label_arg["labels"] == labels

    _create_mock_class_to_check_labels_default(
        "probly.visualization.credal.input_handling.IntervalVisualizer",
        "interval_plot",
        data2d,
        ["C1", "C2"],
    )

    _create_mock_class_to_check_labels_default(
        "probly.visualization.credal.input_handling.TernaryVisualizer",
        "ternary_plot",
        data3d,
        ["C1", "C2", "C3"],
    )
    _create_mock_class_to_check_labels_default(
        "probly.visualization.credal.input_handling.MultiVisualizer",
        "spider_plot",
        data4d,
        ["C1", "C2", "C3", "C4"],
    )


def test_dispatch_custom_labels() -> None:
    """Tests the possibility to add custom labels to the plot."""

    def _create_mock_class_to_check_labels_custom(
        class_path: str,
        method_name: str,
        data: np.ndarray,
        labels: list[str],
    ) -> None:
        with patch(class_path) as mock_class:
            mock_instance = mock_class.return_value
            mocked_method = getattr(mock_instance, method_name)
            dispatch_plot(data, labels=labels)
            _, label_arg = mocked_method.call_args
            assert label_arg["labels"] is labels

    _create_mock_class_to_check_labels_custom(
        "probly.visualization.credal.input_handling.IntervalVisualizer",
        "interval_plot",
        data2d,
        ["Class A", "Class B"],
    )

    _create_mock_class_to_check_labels_custom(
        "probly.visualization.credal.input_handling.TernaryVisualizer",
        "ternary_plot",
        data3d,
        ["Class A", "Class B", "Class C"],
    )
    _create_mock_class_to_check_labels_custom(
        "probly.visualization.credal.input_handling.MultiVisualizer",
        "spider_plot",
        data4d,
        ["Class A", "Class B", "Class C", "Class D"],
    )


def test_dispatch_wrong_label_length() -> None:
    """Tests the ValueError when the added labels are not the right number."""
    with pytest.raises(ValueError, match="Number of labels"):
        dispatch_plot(data2d, labels=["C1"])
    with pytest.raises(ValueError, match="Number of labels"):
        dispatch_plot(data3d, labels=["C1", "C2"])
    with pytest.raises(ValueError, match="Number of labels"):
        dispatch_plot(data4d, labels=["C1", "C2"])


def test_dispatch_default_title() -> None:
    """Tests that the default title is set correctly."""

    def _create_mock_class_to_check_title_default(
        class_path: str,
        method_name: str,
        data: np.ndarray,
        title: str,
    ) -> None:
        with patch(class_path) as mock_class:
            mock_instance = mock_class.return_value
            mocked_method = getattr(mock_instance, method_name)
            dispatch_plot(data)
            _, title_arg = mocked_method.call_args
            assert title_arg["title"] == title

    _create_mock_class_to_check_title_default(
        "probly.visualization.credal.input_handling.IntervalVisualizer",
        "interval_plot",
        data2d,
        "Credal Plot (2 Classes)",
    )

    _create_mock_class_to_check_title_default(
        "probly.visualization.credal.input_handling.TernaryVisualizer",
        "ternary_plot",
        data3d,
        "Credal Plot (3 Classes)",
    )
    _create_mock_class_to_check_title_default(
        "probly.visualization.credal.input_handling.MultiVisualizer",
        "spider_plot",
        data4d,
        "Credal Plot (4 Classes)",
    )


def test_dispatch_custom_title() -> None:
    """Tests the possibility to add a custom title to the plot."""

    def _create_mock_class_to_check_title_custom(
        class_path: str,
        method_name: str,
        data: np.ndarray,
        title: str,
    ) -> None:
        with patch(class_path) as mock_class:
            mock_instance = mock_class.return_value
            mocked_method = getattr(mock_instance, method_name)
            dispatch_plot(data, title=title)
            _, title_arg = mocked_method.call_args
            assert title_arg["title"] is title

    _create_mock_class_to_check_title_custom(
        "probly.visualization.credal.input_handling.IntervalVisualizer",
        "interval_plot",
        data2d,
        "Interval Plot",
    )

    _create_mock_class_to_check_title_custom(
        "probly.visualization.credal.input_handling.TernaryVisualizer",
        "ternary_plot",
        data3d,
        "Ternary Plot",
    )
    _create_mock_class_to_check_title_custom(
        "probly.visualization.credal.input_handling.MultiVisualizer",
        "spider_plot",
        data4d,
        "Spider Plot",
    )


def test_dispatch_choice_none() -> None:
    """Tests the behaviour of choice, when choice is None."""

    def _create_mock_class_to_check_choice_none(class_path: str, method_name: str, data: np.ndarray) -> None:
        with patch(class_path) as mock_class:
            mock_instance = mock_class.return_value
            mocked_method = getattr(mock_instance, method_name)
            dispatch_plot(data)
            _, choice_args = mocked_method.call_args
            assert choice_args["mle_flag"] is True
            assert choice_args["credal_flag"] is True

    _create_mock_class_to_check_choice_none(
        "probly.visualization.credal.input_handling.IntervalVisualizer",
        "interval_plot",
        data2d,
    )

    _create_mock_class_to_check_choice_none(
        "probly.visualization.credal.input_handling.TernaryVisualizer",
        "ternary_plot",
        data3d,
    )

    _create_mock_class_to_check_choice_none(
        "probly.visualization.credal.input_handling.MultiVisualizer",
        "spider_plot",
        data4d,
    )


def test_dispatch_choice_mle() -> None:
    """Tests the behaviour of choice, when choice is ""MLE"."""

    def _create_mock_class_to_check_choice_mle(class_path: str, method_name: str, data: np.ndarray) -> None:
        with patch(class_path) as mock_class:
            mock_instance = mock_class.return_value
            mocked_method = getattr(mock_instance, method_name)
            dispatch_plot(data, choice="MLE")
            _, choice_args = mocked_method.call_args
            assert choice_args["mle_flag"] is True
            assert choice_args["credal_flag"] is False

    _create_mock_class_to_check_choice_mle(
        "probly.visualization.credal.input_handling.IntervalVisualizer",
        "interval_plot",
        data2d,
    )

    _create_mock_class_to_check_choice_mle(
        "probly.visualization.credal.input_handling.TernaryVisualizer",
        "ternary_plot",
        data3d,
    )

    _create_mock_class_to_check_choice_mle(
        "probly.visualization.credal.input_handling.MultiVisualizer",
        "spider_plot",
        data4d,
    )


def test_dispatch_choice_credal() -> None:
    """Tests the behaviour of choice, when choice is "Credal"."""

    def _create_mock_class_to_check_choice_credal(class_path: str, method_name: str, data: np.ndarray) -> None:
        with patch(class_path) as mock_class:
            mock_instance = mock_class.return_value
            mocked_method = getattr(mock_instance, method_name)
            dispatch_plot(data, choice="Credal")
            _, choice_args = mocked_method.call_args
            assert choice_args["mle_flag"] is False
            assert choice_args["credal_flag"] is True

    _create_mock_class_to_check_choice_credal(
        "probly.visualization.credal.input_handling.IntervalVisualizer",
        "interval_plot",
        data2d,
    )

    _create_mock_class_to_check_choice_credal(
        "probly.visualization.credal.input_handling.TernaryVisualizer",
        "ternary_plot",
        data3d,
    )

    _create_mock_class_to_check_choice_credal(
        "probly.visualization.credal.input_handling.MultiVisualizer",
        "spider_plot",
        data4d,
    )


def test_dispatch_choice_probability() -> None:
    """Tests the behaviour of choice, when choice is "Probability"."""

    def _create_mock_class_to_check_choice_probability(class_path: str, method_name: str, data: np.ndarray) -> None:
        with patch(class_path) as mock_class:
            mock_instance = mock_class.return_value
            mocked_method = getattr(mock_instance, method_name)
            dispatch_plot(data, choice="Probability")
            _, choice_args = mocked_method.call_args
            assert choice_args["mle_flag"] is False
            assert choice_args["credal_flag"] is False

    _create_mock_class_to_check_choice_probability(
        "probly.visualization.credal.input_handling.IntervalVisualizer",
        "interval_plot",
        data2d,
    )

    _create_mock_class_to_check_choice_probability(
        "probly.visualization.credal.input_handling.TernaryVisualizer",
        "ternary_plot",
        data3d,
    )

    _create_mock_class_to_check_choice_probability(
        "probly.visualization.credal.input_handling.MultiVisualizer",
        "spider_plot",
        data4d,
    )


def test_dispatch_choice_wrong_input() -> None:
    """Tests the behaviour of choice, when the input of choice does not exist."""
    with pytest.raises(ValueError, match="Choice must be"):
        dispatch_plot(data2d, choice="invalid")
    with pytest.raises(ValueError, match="Choice must be"):
        dispatch_plot(data3d, choice="invalid")
    with pytest.raises(ValueError, match="Choice must be"):
        dispatch_plot(data4d, choice="invalid")


def test_dispatch_minmax_none() -> None:
    """Tests the behaviour of minmax, when minmax is None."""
    with patch("probly.visualization.credal.input_handling.TernaryVisualizer") as mock_class:
        mock_instance = mock_class.return_value
        dispatch_plot(data3d, minmax=None)
        _, minmax_args = mock_instance.ternary_plot.call_args
        assert minmax_args["minmax_flag"] is False


def test_dispatch_minmax_true_credal_false() -> None:
    """Tests the behaviour of minmax, when minmax is True, but credal_flag is False."""
    with patch("probly.visualization.credal.input_handling.TernaryVisualizer") as mock_class:
        mock_instance = mock_class.return_value
        dispatch_plot(data3d, choice="MLE", minmax=True)
        _, minmax_args = mock_instance.ternary_plot.call_args
        assert minmax_args["minmax_flag"] is False


def test_dispatch_minmax_true_credal_true() -> None:
    """Tests the behaviour of minmax, when minmax is True and credal_flag is True."""
    with patch("probly.visualization.credal.input_handling.TernaryVisualizer") as mock_class:
        mock_instance = mock_class.return_value
        dispatch_plot(data3d, choice="Credal", minmax=True)
        _, minmax_args = mock_instance.ternary_plot.call_args
        assert minmax_args["minmax_flag"] is True


def test_simulated_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Catches a simulated error to test that the function would throw an error."""
    error_msg = "crash"

    def _fake_interval(*args: object, **kwargs: object) -> None:  # noqa:ARG001
        raise ValueError(error_msg)

    def _fake_ternary(*args: object, **kwargs: object) -> None:  # noqa:ARG001
        raise ValueError(error_msg)

    def _fake_spider(*args: object, **kwargs: object) -> None:  # noqa:ARG001
        raise ValueError(error_msg)

    def _set_monkeypatch(method_class: object, method_name: str, method_fake: object) -> None:
        monkeypatch.setattr(method_class, method_name, method_fake)

    _set_monkeypatch(IntervalVisualizer, "interval_plot", _fake_interval)
    _set_monkeypatch(TernaryVisualizer, "ternary_plot", _fake_ternary)
    _set_monkeypatch(MultiVisualizer, "spider_plot", _fake_spider)

    for data in [data2d, data3d, data4d]:
        with pytest.raises(ValueError, match=error_msg):
            dispatch_plot(data)
