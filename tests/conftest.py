"""Test fixtures for probly."""

from __future__ import annotations

pytest_plugins = [
    "tests.probly.fixtures.common",
    "tests.probly.fixtures.torch_models",
    "tests.probly.fixtures.torch_data",
    "tests.probly.fixtures.flax_models",
    "tests.probly.fixtures.sklearn_models",
    "tests.probly.fixtures.samples",
    "tests.probly.fixtures.torch_samples",
    "tests.probly.fixtures.jax_samples",
]
