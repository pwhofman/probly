"""Tests for the SAPS nonconformity score across backends and adapters.

Covers torch and jax dispatch parity with the numpy implementation,
numpy-side validation paths (zero-dim and label-shape errors), the
categorical-distribution and sample adapters, and the unsupported-type
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


class TestSAPSBackends:
    """SAPS dispatch correctness across backends."""

    def test_torch_matches_numpy_when_not_randomized(self) -> None:
        torch = _torch()
        from probly.conformal_scores.saps._common import (  # noqa: PLC0415
            _saps_score_dispatch as dispatch,
            compute_saps_score_func_numpy,
        )

        probs_np = np.array([[0.05, 0.85, 0.10], [0.40, 0.35, 0.25]])
        labels_np = np.array([0, 2])
        expected = compute_saps_score_func_numpy(probs_np, labels_np, randomized=False, lambda_val=0.25)
        scores = dispatch(
            torch.tensor(probs_np),
            torch.tensor(labels_np),
            randomized=False,
            lambda_val=0.25,
        )
        np.testing.assert_allclose(scores.numpy(), expected, atol=1e-6)

    def test_torch_label_shape_validation(self) -> None:
        torch = _torch()
        from probly.conformal_scores.saps._common import _saps_score_dispatch as dispatch  # noqa: PLC0415

        probs = torch.tensor([[0.1, 0.2, 0.7]])
        with pytest.raises(ValueError, match="y_cal must match"):
            dispatch(probs, torch.tensor([[0, 1]]), randomized=False, lambda_val=0.1)

    def test_torch_rejects_zero_dim(self) -> None:
        torch = _torch()
        from probly.conformal_scores.saps._common import _saps_score_dispatch as dispatch  # noqa: PLC0415

        with pytest.raises(ValueError, match="at least one dimension"):
            dispatch(torch.tensor(0.5))

    def test_jax_matches_numpy_when_not_randomized(self) -> None:
        _, jnp = _jax_modules()
        from probly.conformal_scores.saps._common import (  # noqa: PLC0415
            _saps_score_dispatch as dispatch,
            compute_saps_score_func_numpy,
        )

        probs_np = np.array([[0.05, 0.85, 0.10], [0.40, 0.35, 0.25]])
        labels_np = np.array([0, 2])
        expected = compute_saps_score_func_numpy(probs_np, labels_np, randomized=False, lambda_val=0.25)
        scores = dispatch(jnp.asarray(probs_np), jnp.asarray(labels_np), randomized=False, lambda_val=0.25)
        np.testing.assert_allclose(np.asarray(scores), expected, atol=1e-6)

    def test_jax_label_shape_validation(self) -> None:
        _, jnp = _jax_modules()
        from probly.conformal_scores.saps._common import _saps_score_dispatch as dispatch  # noqa: PLC0415

        with pytest.raises(ValueError, match="y_cal must match"):
            dispatch(jnp.asarray([[0.2, 0.8]]), jnp.asarray([[0]]), randomized=False)

    def test_jax_rejects_zero_dim(self) -> None:
        _, jnp = _jax_modules()
        from probly.conformal_scores.saps._common import _saps_score_dispatch as dispatch  # noqa: PLC0415

        with pytest.raises(ValueError, match="at least one dimension"):
            dispatch(jnp.asarray(0.5))

    def test_torch_categorical_distribution_input(self) -> None:
        torch = _torch()
        from probly.conformal_scores.saps._common import _saps_score_dispatch as dispatch  # noqa: PLC0415
        from probly.representation.distribution.torch_categorical import (  # noqa: PLC0415
            TorchProbabilityCategoricalDistribution,
        )

        probs = torch.tensor([[0.05, 0.85, 0.10]])
        dist = TorchProbabilityCategoricalDistribution(probs)
        labels = torch.tensor([1])
        scores = dispatch(dist, labels, randomized=False, lambda_val=0.1)
        # the top-1 class case yields u * max_probs = 0
        assert scores.item() == pytest.approx(0.0, abs=1e-6)

    def test_torch_sample_input(self) -> None:
        torch = _torch()
        from probly.conformal_scores.saps._common import _saps_score_dispatch as dispatch  # noqa: PLC0415
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415

        tensor = torch.tensor([[0.2, 0.5, 0.3]])
        scores = dispatch(TorchSample(tensor, sample_dim=0), randomized=False, lambda_val=0.1)
        scores_tensor = dispatch(tensor, randomized=False, lambda_val=0.1)
        assert torch.allclose(scores, scores_tensor)


class TestSAPSNumpyValidation:
    def test_zero_dim_raises(self) -> None:
        from probly.conformal_scores import saps_score  # noqa: PLC0415

        with pytest.raises(ValueError, match="at least one dimension"):
            saps_score(np.array(0.5))

    def test_label_shape_mismatch_raises(self) -> None:
        from probly.conformal_scores import saps_score  # noqa: PLC0415

        with pytest.raises(ValueError, match="y_cal must match"):
            saps_score(np.array([[0.2, 0.5, 0.3]]), np.array([[0]]))


class TestSAPSFallback:
    """The SAPS dispatch raises for unknown types."""

    def test_unsupported_type_raises(self) -> None:
        from probly.conformal_scores.saps._common import _saps_score_dispatch  # noqa: PLC0415

        with pytest.raises(NotImplementedError, match="not implemented"):
            _saps_score_dispatch(object())
