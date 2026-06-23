"""Tests for the UACQR nonconformity score across backends and adapters.

Covers torch and jax dispatch parity with the numpy implementation,
shape validation, the torch sample adapter, and the unsupported-type
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


class TestUACQRBackends:
    """UACQR dispatch correctness across backends."""

    def test_torch_matches_numpy(self) -> None:
        torch = _torch()
        from probly.conformal_scores import uacqr_score  # noqa: PLC0415

        y_pred_np = np.array(
            [
                [[0.0, 1.0], [0.1, 0.9]],
                [[0.2, 1.2], [0.3, 1.0]],
                [[0.1, 0.9], [0.2, 1.0]],
            ]
        )
        y_true_np = np.array([0.5, 0.6])
        expected = uacqr_score(y_pred_np, y_true_np)
        result = uacqr_score(torch.tensor(y_pred_np), torch.tensor(y_true_np))
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-5)

    def test_torch_shape_error(self) -> None:
        torch = _torch()
        from probly.conformal_scores import uacqr_score  # noqa: PLC0415

        with pytest.raises(ValueError, match="intervals must have shape"):
            uacqr_score(torch.zeros((3, 2)), torch.zeros((2,)))

    def test_torch_sample_input(self) -> None:
        torch = _torch()
        from probly.conformal_scores import uacqr_score  # noqa: PLC0415
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415

        tensor = torch.tensor([[[0.0, 1.0]], [[0.1, 0.9]], [[0.2, 1.0]]])
        y_true = torch.tensor([0.5])
        result_sample = uacqr_score(TorchSample(tensor, sample_dim=0), y_true)
        result_tensor = uacqr_score(tensor, y_true)
        assert torch.allclose(result_sample, result_tensor)

    def test_jax_matches_numpy(self) -> None:
        _, jnp = _jax_modules()
        from probly.conformal_scores import uacqr_score  # noqa: PLC0415

        y_pred_np = np.array(
            [
                [[0.0, 1.0], [0.1, 0.9]],
                [[0.2, 1.2], [0.3, 1.0]],
                [[0.1, 0.9], [0.2, 1.0]],
            ]
        )
        y_true_np = np.array([0.5, 0.6])
        expected = uacqr_score(y_pred_np, y_true_np)
        result = uacqr_score(jnp.asarray(y_pred_np), jnp.asarray(y_true_np))
        np.testing.assert_allclose(np.asarray(result), expected, atol=1e-5)

    def test_jax_shape_error(self) -> None:
        _, jnp = _jax_modules()
        from probly.conformal_scores import uacqr_score  # noqa: PLC0415

        with pytest.raises(ValueError, match="intervals must have shape"):
            uacqr_score(jnp.zeros((3, 2)), jnp.zeros((2,)))


class TestUACQRFallback:
    """The UACQR dispatch raises for unknown types."""

    def test_uacqr_unsupported_type_raises(self) -> None:
        from probly.conformal_scores import uacqr_score  # noqa: PLC0415

        with pytest.raises(NotImplementedError, match="not implemented"):
            uacqr_score(object(), object())
