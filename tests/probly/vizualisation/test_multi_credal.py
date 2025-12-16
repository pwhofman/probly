from __future__ import annotations

import matplotlib as mpl
import pytest

# Use a non-interactive backend so tests do not open GUI windows.
mpl.use("Agg")

from .multi_credal import spider_plot


def test_spider_plot_single_dataset() -> None:
    """Spider plot runs without error for a single dataset."""
    labels = ["A", "B", "C", "D"]
    datasets = [
        ("Run 1", [0.1, 0.4, 0.3, 0.2]),
    ]

    spider_plot(labels, datasets, title="Single dataset test")


def test_spider_plot_multiple_datasets() -> None:
    """Spider plot runs with multiple datasets and matching label length."""
    labels = ["A", "B", "C", "D", "E"]
    datasets = [
        ("Run 1", [0.1, 0.5, 0.2, 0.3, 0.7]),
        ("Run 2", [0.3, 0.6, 0.1, 0.4, 0.2]),
        ("Run 3", [0.8, 0.1, 0.05, 0.2, 0.3]),
    ]

    spider_plot(labels, datasets, title="Multiple datasets test")


def test_spider_plot_length_mismatch_raises() -> None:
    """Spider plot raises ValueError if values length does not match labels."""
    labels = ["A", "B", "C"]
    # One extra value on purpose.
    datasets = [
        ("Bad run", [0.1, 0.2, 0.3, 0.4]),
    ]

    # Check that the error message mentions the length mismatch.
    with pytest.raises(ValueError, match=r"has length .* expected .* values"):
        spider_plot(labels, datasets)


def test_spider_plot_circle_frame() -> None:
    """Spider plot runs with circular frame."""
    labels = ["x1", "x2", "x3"]
    datasets = [
        ("Run 1", [0.2, 0.5, 0.3]),
        ("Run 2", [0.1, 0.7, 0.2]),
    ]

    spider_plot(labels, datasets, frame="circle", title="Circle frame test")


def test_spider_plot_six_classes() -> None:
    """Spider plot runs correctly with six classes."""
    labels = ["A", "B", "C", "D", "E", "F"]
    datasets = [
        ("Run 1", [0.1, 0.3, 0.2, 0.4, 0.6, 0.5]),
        ("Run 2", [0.6, 0.2, 0.1, 0.3, 0.4, 0.5]),
        ("Run 3", [0.2, 0.5, 0.4, 0.1, 0.3, 0.6]),
    ]

    spider_plot(labels, datasets, title="Six-class test")
