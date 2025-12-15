from __future__ import annotations

import matplotlib as mpl
import pytest

# Use a non-interactive backend so tests do not open GUI windows.
mpl.use("Agg")

# Import from the same package (tests/vizualisation/)
from .multi_credal import spider_plot


def test_spider_plot_runs_with_minimal_valid_input() -> None:
    """Spider plot runs with the smallest meaningful valid input."""
    labels = ["A", "B"]
    datasets = [
        ("Run 1", [0.5, 0.5]),
    ]

    spider_plot(labels, datasets)


def test_spider_plot_raises_on_empty_datasets() -> None:
    """Spider plot raises a ValueError if no datasets are provided."""
    labels = ["A", "B", "C"]

    with pytest.raises(ValueError, match=r".+"):
        spider_plot(labels, [])


def test_spider_plot_raises_on_invalid_frame() -> None:
    """Spider plot raises ValueError for an unknown frame type."""
    labels = ["A", "B", "C"]
    datasets = [
        ("Run 1", [0.3, 0.3, 0.4]),
    ]

    with pytest.raises(ValueError, match=r"Unknown frame"):
        spider_plot(labels, datasets, frame="triangle")


def test_spider_plot_runs_with_many_classes() -> None:
    """Spider plot runs with a larger number of classes."""
    labels = [f"C{i}" for i in range(10)]
    datasets = [
        ("Run 1", [0.1] * 10),
        ("Run 2", [0.05] * 10),
    ]

    spider_plot(labels, datasets)
