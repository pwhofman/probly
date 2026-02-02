"""Test for create_credal_plot."""

from __future__ import annotations

from unittest.mock import patch

import matplotlib as mpl
import matplotlib.axes as mplaxes
import numpy as np
import pytest

mpl.use("Agg")

import probly.visualization.credal.credal_visualization as credalviz
from probly.visualization.credal.credal_visualization import create_credal_plot

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

    def _create_mock_class(
        class_path: str,
        method_name: str,
        return_value: str,
        data: np.ndarray,
    ) -> None:
        with patch(class_path) as mock_class:
            mock_instance = mock_class.return_value
            mocked_method = getattr(mock_instance, method_name)
            mocked_method.return_value = return_value
            ax = create_credal_plot(data, show=False)
            mock_class.assert_called_once()
            mocked_method.assert_called_once()
            assert ax == return_value

    _create_mock_class(
        "probly.visualization.credal.input_handling.IntervalVisualizer",
        "interval_plot",
        "interval_ax",
        data2d,
    )

    _create_mock_class(
        "probly.visualization.credal.input_handling.TernaryVisualizer",
        "ternary_plot",
        "ternary_ax",
        data3d,
    )

    _create_mock_class(
        "probly.visualization.credal.input_handling.MultiVisualizer",
        "spider_plot",
        "multi_ax",
        data4d,
    )


def test_show_false_returns_object() -> None:
    """Tests the show flag behaviour when show is False."""

    def _create_mock_show_false(data: np.ndarray) -> None:
        with patch("matplotlib.pyplot.show") as mock_show:
            ax = create_credal_plot(data, show=False)
            mock_show.assert_not_called()
            assert ax is not None
            assert isinstance(ax, mplaxes.Axes)

    _create_mock_show_false(data2d)
    _create_mock_show_false(data3d)
    _create_mock_show_false(data4d)


def test_show_true_shows_plot() -> None:
    """Tests the show flag behaviour when show is False."""

    def _create_mock_show_true(data: np.ndarray) -> None:
        with patch("matplotlib.pyplot.show") as mock_show:
            ax = create_credal_plot(data, show=True)
            mock_show.assert_called_once()
            assert ax is not None
            assert isinstance(ax, mplaxes.Axes)

    _create_mock_show_true(data2d)
    _create_mock_show_true(data3d)
    _create_mock_show_true(data4d)


def test_simulated_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Catches a simulated error to test that the function would throw an error."""
    error_msg = "crash"

    def _fake_dispatch(*args: object, **kwargs: object) -> None:  # noqa:ARG001
        raise ValueError(error_msg)

    monkeypatch.setattr(credalviz, "dispatch_plot", _fake_dispatch)

    for data in (data2d, data3d, data4d):
        with pytest.raises(ValueError, match=error_msg):
            create_credal_plot(data, show=False)
