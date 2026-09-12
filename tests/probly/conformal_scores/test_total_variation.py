"""Tests for the total-variation nonconformity score across backends.

Covers torch and jax dispatch parity with the numpy implementation for
both integer-label and probability-mass ``y_true``, the zero
self-distance property, the requirement that the callable form rejects
calls without ``y_true``, and the unsupported-type fallback path.
"""

from __future__ import annotations

import numpy as np
import pytest


def _torch():
    """Return torch module or skip the calling test."""
    return pytest.importorskip("torch")


def _jax_modules():
    """Return jax + jax.numpy or skip the calling test."""
    pytest.importorskip("jax")
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    return jax, jnp


Y_PRED = np.array([[0.2, 0.5, 0.3], [0.1, 0.1, 0.8]])
Y_TRUE_LABELS = np.array([0, 2])
Y_TRUE_PMF = np.array([[0.1, 0.6, 0.3], [0.3, 0.2, 0.5]])


class TestTotalVariationBackends:
    """Total-variation dispatch correctness across backends."""

    def test_torch_matches_numpy(self) -> None:
        torch = _torch()
        from probly.conformal_scores import tv_score_func  # noqa: PLC0415

        expected = tv_score_func(Y_PRED, Y_TRUE_LABELS)
        result = tv_score_func(torch.tensor(Y_PRED), torch.tensor(Y_TRUE_LABELS))
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-6)

    def test_torch_matches_numpy_for_distribution_targets(self) -> None:
        torch = _torch()
        from probly.conformal_scores import tv_score_func  # noqa: PLC0415

        expected = tv_score_func(Y_PRED, Y_TRUE_PMF)
        result = tv_score_func(torch.tensor(Y_PRED), torch.tensor(Y_TRUE_PMF))
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-6)

    def test_torch_self_distance_is_zero(self) -> None:
        torch = _torch()
        from probly.conformal_scores import tv_score_func  # noqa: PLC0415

        p = torch.tensor([[0.3, 0.4, 0.3]])
        result = tv_score_func(p, p)
        assert result.item() == pytest.approx(0.0, abs=1e-6)

    def test_jax_matches_numpy(self) -> None:
        _, jnp = _jax_modules()
        from probly.conformal_scores import tv_score_func  # noqa: PLC0415

        expected = tv_score_func(Y_PRED, Y_TRUE_LABELS)
        result = tv_score_func(jnp.asarray(Y_PRED), jnp.asarray(Y_TRUE_LABELS))
        np.testing.assert_allclose(np.asarray(result), expected, atol=1e-6)

    def test_jax_matches_numpy_for_distribution_targets(self) -> None:
        _, jnp = _jax_modules()
        from probly.conformal_scores import tv_score_func  # noqa: PLC0415

        expected = tv_score_func(Y_PRED, Y_TRUE_PMF)
        result = tv_score_func(jnp.asarray(Y_PRED), jnp.asarray(Y_TRUE_PMF))
        np.testing.assert_allclose(np.asarray(result), expected, atol=1e-6)

    def test_jax_self_distance_is_zero(self) -> None:
        _, jnp = _jax_modules()
        from probly.conformal_scores import tv_score_func  # noqa: PLC0415

        p = jnp.asarray([[0.3, 0.4, 0.3]])
        result = tv_score_func(p, p)
        assert float(np.asarray(result).item()) == pytest.approx(0.0, abs=1e-6)


class TestTotalVariationCallable:
    """The ``TVScore`` callable requires ``y_true``."""

    def test_numpy_callable_requires_y_true(self) -> None:
        from probly.conformal_scores import tv_score  # noqa: PLC0415

        with pytest.raises(ValueError, match="y_true is required"):
            tv_score(Y_PRED)

    def test_torch_callable_requires_y_true(self) -> None:
        torch = _torch()
        from probly.conformal_scores import tv_score  # noqa: PLC0415

        with pytest.raises(ValueError, match="y_true is required"):
            tv_score(torch.tensor(Y_PRED))

    def test_jax_callable_requires_y_true(self) -> None:
        _, jnp = _jax_modules()
        from probly.conformal_scores import tv_score  # noqa: PLC0415

        with pytest.raises(ValueError, match="y_true is required"):
            tv_score(jnp.asarray(Y_PRED))


class TestTotalVariationFallback:
    """The total-variation dispatch raises for unknown types."""

    def test_total_variation_unsupported_type_raises(self) -> None:
        from probly.conformal_scores import tv_score_func  # noqa: PLC0415

        with pytest.raises(NotImplementedError, match="not implemented"):
            tv_score_func(object(), object())
