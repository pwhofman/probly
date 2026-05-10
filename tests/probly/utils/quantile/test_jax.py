"""Tests for the jax backend of ``calculate_quantile``."""

from __future__ import annotations

import pytest


def _jax_modules():
    pytest.importorskip("jax")
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    return jax, jnp


class TestQuantileJax:
    """`calculate_quantile` for jax arrays."""

    def test_jax_quantile_basic(self) -> None:
        _, jnp = _jax_modules()
        from probly.utils.quantile import calculate_quantile  # noqa: PLC0415

        scores = jnp.asarray([0.1, 0.2, 0.3, 0.4, 0.5])
        q = calculate_quantile(scores, alpha=0.1)
        assert isinstance(q, float)

    def test_jax_alpha_out_of_range_raises(self) -> None:
        _, jnp = _jax_modules()
        from probly.utils.quantile import calculate_quantile  # noqa: PLC0415

        with pytest.raises(ValueError, match="alpha must be in"):
            calculate_quantile(jnp.asarray([0.1, 0.2]), alpha=1.5)

    def test_jax_empty_scores_raises(self) -> None:
        _, jnp = _jax_modules()
        from probly.utils.quantile import calculate_quantile  # noqa: PLC0415

        with pytest.raises(ValueError, match="empty"):
            calculate_quantile(jnp.asarray([]), alpha=0.1)

    def test_jax_weighted_quantile_unweighted(self) -> None:
        _, jnp = _jax_modules()
        from probly.utils.quantile._common import calculate_weighted_quantile  # noqa: PLC0415

        values = jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
        assert calculate_weighted_quantile(values, 0.5) == pytest.approx(3.0)

    def test_jax_weighted_quantile_with_weights(self) -> None:
        _, jnp = _jax_modules()
        from probly.utils.quantile._common import calculate_weighted_quantile  # noqa: PLC0415

        values = jnp.asarray([1.0, 2.0, 3.0])
        weights = jnp.asarray([1.0, 0.0, 0.0])
        result = calculate_weighted_quantile(values, 0.5, sample_weight=weights)
        assert result == pytest.approx(1.0)
