"""Tests for the CQR nonconformity score across backends and adapters.

Covers torch and jax dispatch parity with the numpy implementation, the
shape and rank validation paths, the leading evaluation-axis averaging,
the torch sample adapter, numpy-side validation, and the unsupported-type
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


class TestCQRBackends:
    """CQR dispatch correctness across backends."""

    def test_torch_matches_numpy(self) -> None:
        torch = _torch()
        from probly.conformal_scores import cqr_score  # noqa: PLC0415

        y_pred_np = np.array([[0.2, 0.8], [0.1, 0.5], [0.3, 0.6]])
        y_true_np = np.array([0.5, 0.7, 0.4])
        expected = cqr_score(y_pred_np, y_true_np)
        result = cqr_score(torch.tensor(y_pred_np), torch.tensor(y_true_np))
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-6)

    def test_torch_averages_evaluation_axis(self) -> None:
        torch = _torch()
        from probly.conformal_scores import cqr_score  # noqa: PLC0415

        # shape (n_evals, n_samples, 2)  # noqa: ERA001
        y_pred = torch.tensor(
            [
                [[0.0, 1.0], [0.1, 0.9]],
                [[0.0, 1.0], [0.1, 0.9]],
            ]
        )
        y_true = torch.tensor([0.5, 0.5])
        scores = cqr_score(y_pred, y_true)
        # mean intervals are unchanged from each estimate, so y=0.5 is inside both
        assert scores.shape == y_true.shape
        assert torch.all(scores <= 0)

    def test_torch_shape_error(self) -> None:
        torch = _torch()
        from probly.conformal_scores import cqr_score  # noqa: PLC0415

        with pytest.raises(ValueError, match="must have shape"):
            cqr_score(torch.tensor([0.1, 0.5]), torch.tensor([0.3]))

    def test_torch_rank_mismatch(self) -> None:
        torch = _torch()
        from probly.conformal_scores import cqr_score  # noqa: PLC0415

        # y_pred has 2 extra leading axes, y_true has none; should raise.
        y_pred = torch.zeros((1, 1, 1, 1, 2))
        y_true = torch.zeros((1,))
        with pytest.raises(ValueError, match="leading evaluation axis"):
            cqr_score(y_pred, y_true)

    def test_torch_sample_input(self) -> None:
        torch = _torch()
        from probly.conformal_scores import cqr_score  # noqa: PLC0415
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415

        tensor = torch.tensor([[0.0, 1.0]])
        y_true = torch.tensor([0.5])
        result = cqr_score(TorchSample(tensor, sample_dim=0), y_true)
        assert torch.allclose(result, cqr_score(tensor, y_true))

    def test_jax_matches_numpy(self) -> None:
        _, jnp = _jax_modules()
        from probly.conformal_scores import cqr_score  # noqa: PLC0415

        y_pred_np = np.array([[0.2, 0.8], [0.1, 0.5], [0.3, 0.6]])
        y_true_np = np.array([0.5, 0.7, 0.4])
        expected = cqr_score(y_pred_np, y_true_np)
        result = cqr_score(jnp.asarray(y_pred_np), jnp.asarray(y_true_np))
        np.testing.assert_allclose(np.asarray(result), expected, atol=1e-6)

    def test_jax_shape_error(self) -> None:
        _, jnp = _jax_modules()
        from probly.conformal_scores import cqr_score  # noqa: PLC0415

        with pytest.raises(ValueError, match="must have shape"):
            cqr_score(jnp.asarray([0.1, 0.5]), jnp.asarray([0.3]))

    def test_jax_rank_mismatch(self) -> None:
        _, jnp = _jax_modules()
        from probly.conformal_scores import cqr_score  # noqa: PLC0415

        y_pred = jnp.zeros((1, 1, 1, 1, 2))
        y_true = jnp.zeros((1,))
        with pytest.raises(ValueError, match="leading evaluation axis"):
            cqr_score(y_pred, y_true)

    def test_jax_averages_evaluation_axis(self) -> None:
        _, jnp = _jax_modules()
        from probly.conformal_scores import cqr_score  # noqa: PLC0415

        y_pred = jnp.asarray([[[0.0, 1.0]], [[0.0, 1.0]]])
        y_true = jnp.asarray([0.5])
        result = cqr_score(y_pred, y_true)
        assert result.shape == y_true.shape


class TestCQRNumpyValidation:
    def test_wrong_shape_raises(self) -> None:
        from probly.conformal_scores import cqr_score  # noqa: PLC0415

        # y_pred should have a trailing class axis of size 2.
        with pytest.raises(ValueError, match="must have shape"):
            cqr_score(np.array([0.5, 0.7]), np.array([0.5, 0.5]))

    def test_rank_mismatch_raises(self) -> None:
        from probly.conformal_scores import cqr_score  # noqa: PLC0415

        # y_pred has too many leading axes for y_true.
        with pytest.raises(ValueError, match="leading evaluation axis"):
            cqr_score(np.zeros((2, 1, 3, 2)), np.zeros((1,)))

    def test_averaging_evaluation_axis(self) -> None:
        from probly.conformal_scores import cqr_score  # noqa: PLC0415

        # n_evaluations = 2 -> averaged across leading axis.
        y_pred = np.array([[[0.0, 1.0]], [[0.0, 1.0]]])
        y_true = np.array([0.5])
        scores = cqr_score(y_pred, y_true)
        assert scores.shape == y_true.shape


class TestCQRFallback:
    """The CQR dispatch raises for unknown types."""

    def test_unsupported_type_raises(self) -> None:
        from probly.conformal_scores import cqr_score  # noqa: PLC0415

        with pytest.raises(NotImplementedError, match="not implemented"):
            cqr_score(object(), object())
