"""Tests for the RAPS nonconformity score across backends and adapters.

Covers torch and jax dispatch parity with the numpy implementation, label
shape validation, zero-dim rejection, the categorical-distribution and
sample adapters (both torch and array variants), and the unsupported-type
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


class TestRAPSBackends:
    """RAPS dispatch correctness across backends."""

    def test_torch_matches_numpy_when_not_randomized(self) -> None:
        torch = _torch()
        from probly.conformal_scores.raps._common import (  # noqa: PLC0415
            _raps_score_dispatch as dispatch,
            compute_raps_score_numpy,
        )

        probs_np = np.array([[0.05, 0.85, 0.10], [0.40, 0.35, 0.25]])
        labels_np = np.array([1, 0])
        expected = compute_raps_score_numpy(
            probs_np,
            labels_np,
            randomized=False,
            lambda_reg=0.2,
            k_reg=1,
        )
        scores = dispatch(
            torch.tensor(probs_np),
            torch.tensor(labels_np),
            randomized=False,
            lambda_reg=0.2,
            k_reg=1,
        )
        np.testing.assert_allclose(scores.numpy(), expected, atol=1e-6)

    def test_torch_label_shape_validation(self) -> None:
        torch = _torch()
        from probly.conformal_scores.raps._common import _raps_score_dispatch as dispatch  # noqa: PLC0415

        probs = torch.tensor([[0.1, 0.2, 0.7]])
        with pytest.raises(ValueError, match="y_cal must match"):
            dispatch(probs, torch.tensor([[0]]), randomized=False)

    def test_torch_rejects_zero_dim(self) -> None:
        torch = _torch()
        from probly.conformal_scores.raps._common import _raps_score_dispatch as dispatch  # noqa: PLC0415

        with pytest.raises(ValueError, match="at least one dimension"):
            dispatch(torch.tensor(0.5))

    def test_jax_matches_numpy_when_not_randomized(self) -> None:
        _, jnp = _jax_modules()
        from probly.conformal_scores.raps._common import (  # noqa: PLC0415
            _raps_score_dispatch as dispatch,
            compute_raps_score_numpy,
        )

        probs_np = np.array([[0.05, 0.85, 0.10], [0.40, 0.35, 0.25]])
        labels_np = np.array([1, 0])
        expected = compute_raps_score_numpy(probs_np, labels_np, randomized=False, lambda_reg=0.2, k_reg=1)
        scores = dispatch(
            jnp.asarray(probs_np),
            jnp.asarray(labels_np),
            randomized=False,
            lambda_reg=0.2,
            k_reg=1,
        )
        np.testing.assert_allclose(np.asarray(scores), expected, atol=1e-6)

    def test_jax_label_shape_validation(self) -> None:
        _, jnp = _jax_modules()
        from probly.conformal_scores.raps._common import _raps_score_dispatch as dispatch  # noqa: PLC0415

        with pytest.raises(ValueError, match="y_cal must match"):
            dispatch(jnp.asarray([[0.1, 0.9]]), jnp.asarray([[0]]), randomized=False)

    def test_jax_rejects_zero_dim(self) -> None:
        _, jnp = _jax_modules()
        from probly.conformal_scores.raps._common import _raps_score_dispatch as dispatch  # noqa: PLC0415

        with pytest.raises(ValueError, match="at least one dimension"):
            dispatch(jnp.asarray(0.5))

    def test_torch_sample_input(self) -> None:
        torch = _torch()
        from probly.conformal_scores.raps._common import _raps_score_dispatch as dispatch  # noqa: PLC0415
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415

        tensor = torch.tensor([[0.2, 0.5, 0.3]])
        scores_via_sample = dispatch(TorchSample(tensor, sample_dim=0), randomized=False, lambda_reg=0.2, k_reg=1)
        scores_via_tensor = dispatch(tensor, randomized=False, lambda_reg=0.2, k_reg=1)
        assert torch.allclose(scores_via_sample, scores_via_tensor)

    def test_torch_categorical_distribution_input(self) -> None:
        torch = _torch()
        from probly.conformal_scores.raps._common import _raps_score_dispatch as dispatch  # noqa: PLC0415
        from probly.representation.distribution.torch_categorical import (  # noqa: PLC0415
            TorchProbabilityCategoricalDistribution,
        )

        probs = torch.tensor([[0.2, 0.5, 0.3]])
        dist = TorchProbabilityCategoricalDistribution(probs)
        scores = dispatch(dist, randomized=False, lambda_reg=0.0, k_reg=0)
        # with lambda_reg=0, RAPS reduces to APS for the same probs
        from probly.conformal_scores.aps._common import _aps_score_dispatch as aps_dispatch  # noqa: PLC0415

        aps_scores = aps_dispatch(probs, randomized=False)
        assert torch.allclose(scores, aps_scores, atol=1e-6)


class TestRAPSFallbacks:
    """The dispatch raises for unknown types and accepts wrapped sample types."""

    def test_unsupported_type_raises(self) -> None:
        from probly.conformal_scores.raps._common import _raps_score_dispatch  # noqa: PLC0415

        with pytest.raises(NotImplementedError, match="not implemented"):
            _raps_score_dispatch(object())

    def test_array_categorical_distribution_input(self) -> None:
        from probly.conformal_scores.raps._common import _raps_score_dispatch  # noqa: PLC0415
        from probly.representation.distribution.array_categorical import (  # noqa: PLC0415
            ArrayProbabilityCategoricalDistribution,
        )

        d = ArrayProbabilityCategoricalDistribution(array=np.array([[0.2, 0.5, 0.3]]))
        labels = np.array([1])
        out = _raps_score_dispatch(d, labels, randomized=False, lambda_reg=0.0, k_reg=0)
        assert np.isfinite(out).all()

    def test_array_sample_input(self) -> None:
        from probly.conformal_scores.raps._common import _raps_score_dispatch  # noqa: PLC0415
        from probly.representation.sample.array import ArraySample  # noqa: PLC0415

        sample = ArraySample(array=np.array([[0.2, 0.5, 0.3]]), sample_axis=0)
        out = _raps_score_dispatch(sample, randomized=False, lambda_reg=0.0, k_reg=0)
        assert np.isfinite(out).all()
