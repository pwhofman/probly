"""Tests for the LAC nonconformity score across backends and adapters.

Covers torch and jax dispatch parity with the numpy implementation, label
shape validation, zero-dim rejection, the categorical-distribution and
sample adapters, and the unsupported-type fallback path.
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


class TestLACBackends:
    """LAC dispatch correctness across backends."""

    def test_torch_matches_numpy(self) -> None:
        torch = _torch()
        from probly.conformal_scores import lac_score  # noqa: PLC0415

        probs_np = np.array([[0.2, 0.5, 0.3]])
        labels_np = np.array([1])
        expected = lac_score(probs_np, labels_np)
        result = lac_score(torch.tensor(probs_np), torch.tensor(labels_np))
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-6)

    def test_torch_no_labels_returns_full(self) -> None:
        torch = _torch()
        from probly.conformal_scores import lac_score  # noqa: PLC0415

        probs = torch.tensor([[0.2, 0.5, 0.3]])
        scores = lac_score(probs)
        assert scores.shape == probs.shape
        # LAC: score = 1 - probs  # noqa: ERA001
        expected = torch.tensor([[0.8, 0.5, 0.7]])
        assert torch.allclose(scores, expected)

    def test_torch_label_shape_validation(self) -> None:
        torch = _torch()
        from probly.conformal_scores import lac_score  # noqa: PLC0415

        with pytest.raises(ValueError, match="y_cal must match"):
            lac_score(torch.tensor([[0.5, 0.5]]), torch.tensor([[0]]))

    def test_torch_rejects_zero_dim(self) -> None:
        torch = _torch()
        from probly.conformal_scores import lac_score  # noqa: PLC0415

        with pytest.raises(ValueError, match="at least one dimension"):
            lac_score(torch.tensor(0.5))

    def test_jax_matches_numpy(self) -> None:
        _, jnp = _jax_modules()
        from probly.conformal_scores import lac_score  # noqa: PLC0415

        probs_np = np.array([[0.2, 0.5, 0.3]])
        labels_np = np.array([1])
        expected = lac_score(probs_np, labels_np)
        result = lac_score(jnp.asarray(probs_np), jnp.asarray(labels_np))
        np.testing.assert_allclose(np.asarray(result), expected, atol=1e-6)

    def test_jax_label_shape_validation(self) -> None:
        _, jnp = _jax_modules()
        from probly.conformal_scores import lac_score  # noqa: PLC0415

        with pytest.raises(ValueError, match="y_cal must match"):
            lac_score(jnp.asarray([[0.5, 0.5]]), jnp.asarray([[0]]))

    def test_jax_rejects_zero_dim(self) -> None:
        _, jnp = _jax_modules()
        from probly.conformal_scores import lac_score  # noqa: PLC0415

        with pytest.raises(ValueError, match="at least one dimension"):
            lac_score(jnp.asarray(0.5))

    def test_torch_sample_input(self) -> None:
        torch = _torch()
        from probly.conformal_scores import lac_score  # noqa: PLC0415
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415

        tensor = torch.tensor([[0.1, 0.2, 0.7]])
        sample_scores = lac_score(TorchSample(tensor, sample_dim=0))
        tensor_scores = lac_score(tensor)
        assert torch.allclose(sample_scores, tensor_scores)

    def test_torch_categorical_distribution_input(self) -> None:
        torch = _torch()
        from probly.conformal_scores import lac_score  # noqa: PLC0415
        from probly.representation.distribution.torch_categorical import (  # noqa: PLC0415
            TorchProbabilityCategoricalDistribution,
        )

        probs = torch.tensor([[0.1, 0.2, 0.7]])
        dist = TorchProbabilityCategoricalDistribution(probs)
        result = lac_score(dist)
        assert torch.allclose(result, 1.0 - probs)


class TestLACFallback:
    """The LAC dispatch raises for unknown types."""

    def test_unsupported_type_raises(self) -> None:
        from probly.conformal_scores import lac_score  # noqa: PLC0415

        with pytest.raises(NotImplementedError, match="not implemented"):
            lac_score(object())
