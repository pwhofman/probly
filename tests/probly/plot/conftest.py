"""Shared fixtures for plot tests."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest

mpl.use("Agg")


@pytest.fixture
def _close_figures():
    yield
    plt.close("all")
