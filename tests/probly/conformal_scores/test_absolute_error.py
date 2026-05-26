"""Tests for the absolute-error nonconformity score across backends.

Covers torch and jax dispatch parity with the numpy implementation,
shape validation, evaluation-axis averaging, the torch sample adapter
(including using a sample for ``y_true``), and the unsupported-type
fallback path.
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


class TestAbsoluteErrorBackends:
    """Absolute-error dispatch correctness across backends."""

    def test_torch_matches_numpy(self) -> None:
        torch = _torch()
        from probly.conformal_scores import absolute_error_score  # noqa: PLC0415

        y_pred = np.array([1.0, 2.0, 3.0])
        y_true = np.array([1.5, 2.5, 2.5])
        expected = absolute_error_score(y_pred, y_true)
        result = absolute_error_score(torch.tensor(y_pred), torch.tensor(y_true))
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-6)

    def test_torch_averages_evaluation_axis(self) -> None:
        torch = _torch()
        from probly.conformal_scores import absolute_error_score  # noqa: PLC0415

        # n_estimations=2, n_samples=3
        y_pred = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        y_true = torch.tensor([2.0, 2.0, 2.0])
        scores = absolute_error_score(y_pred, y_true)
        # mean of preds is [2, 2, 2]; abs(mean-y_true) = 0
        np.testing.assert_allclose(scores.numpy(), [0.0, 0.0, 0.0], atol=1e-6)

    def test_torch_shape_error(self) -> None:
        torch = _torch()
        from probly.conformal_scores import absolute_error_score  # noqa: PLC0415

        with pytest.raises(ValueError, match="match y_true shape"):
            absolute_error_score(torch.zeros((1, 1, 5)), torch.zeros((5,)))

    def test_torch_sample_input(self) -> None:
        torch = _torch()
        from probly.conformal_scores import absolute_error_score  # noqa: PLC0415
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415

        y_pred = torch.tensor([1.0, 2.0])
        y_true = torch.tensor([1.5, 2.5])
        result = absolute_error_score(TorchSample(y_pred, sample_dim=0), y_true)
        np.testing.assert_allclose(result.numpy(), [0.5, 0.5], atol=1e-6)

    def test_torch_sample_for_y_true(self) -> None:
        torch = _torch()
        from probly.conformal_scores import absolute_error_score  # noqa: PLC0415
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415

        y_pred = TorchSample(torch.tensor([1.0, 2.0]), sample_dim=0)
        y_true = TorchSample(torch.tensor([1.5, 2.5]), sample_dim=0)
        result = absolute_error_score(y_pred, y_true)
        np.testing.assert_allclose(result.numpy(), [0.5, 0.5], atol=1e-6)

    def test_jax_matches_numpy(self) -> None:
        _, jnp = _jax_modules()
        from probly.conformal_scores import absolute_error_score  # noqa: PLC0415

        y_pred_np = np.array([1.0, 2.0, 3.0])
        y_true_np = np.array([1.5, 2.5, 2.5])
        expected = absolute_error_score(y_pred_np, y_true_np)
        result = absolute_error_score(jnp.asarray(y_pred_np), jnp.asarray(y_true_np))
        np.testing.assert_allclose(np.asarray(result), expected, atol=1e-6)

    def test_jax_shape_error(self) -> None:
        _, jnp = _jax_modules()
        from probly.conformal_scores import absolute_error_score  # noqa: PLC0415

        with pytest.raises(ValueError, match="match y_true shape"):
            absolute_error_score(jnp.zeros((1, 1, 5)), jnp.zeros((5,)))

    def test_jax_averages_evaluation_axis(self) -> None:
        _, jnp = _jax_modules()
        from probly.conformal_scores import absolute_error_score  # noqa: PLC0415

        y_pred = jnp.asarray([[1.0, 2.0], [3.0, 2.0]])
        y_true = jnp.asarray([2.0, 2.0])
        result = absolute_error_score(y_pred, y_true)
        np.testing.assert_allclose(np.asarray(result), [0.0, 0.0], atol=1e-6)


class TestAbsoluteErrorFallback:
    """The absolute-error dispatch raises for unknown types."""

    def test_absolute_error_unsupported_raises(self) -> None:
        from probly.conformal_scores import absolute_error_score  # noqa: PLC0415

        with pytest.raises(NotImplementedError, match="not implemented"):
            absolute_error_score(object(), object())
