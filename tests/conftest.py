"""Test fixtures for probly."""

from __future__ import annotations

pytest_plugins = [
    "tests.probly.fixtures.common",
    "tests.probly.fixtures.torch_models",
    "tests.probly.fixtures.flax_models",
]
