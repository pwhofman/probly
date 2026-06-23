"""Tests for the APS nonconformity score across backends and adapters.

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


class TestAPSBackends:
    """APS dispatch correctness across numpy, torch, jax."""

    def test_torch_matches_numpy_when_not_randomized(self) -> None:
        torch = _torch()
        from probly.conformal_scores.aps._common import (  # noqa: PLC0415
            _aps_score_dispatch as dispatch,
            compute_aps_score_numpy,
        )

        probs_np = np.array([[0.1, 0.6, 0.3], [0.4, 0.4, 0.2]])
        labels_np = np.array([1, 0])
        expected = compute_aps_score_numpy(probs_np, labels_np, randomized=False)

        scores = dispatch(torch.tensor(probs_np), torch.tensor(labels_np), randomized=False)
        assert isinstance(scores, torch.Tensor)
        np.testing.assert_allclose(scores.numpy(), expected, atol=1e-6)

    def test_torch_full_score_matrix_when_no_labels(self) -> None:
        torch = _torch()
        from probly.conformal_scores.aps._common import _aps_score_dispatch as dispatch  # noqa: PLC0415

        probs = torch.tensor([[0.1, 0.6, 0.3]])
        scores = dispatch(probs, randomized=False)
        # APS without labels returns one score per class.
        assert scores.shape == probs.shape
        # The score for the most-probable class should equal that class's
        # probability when randomized is False.
        assert scores[0, 1].item() == pytest.approx(0.6, abs=1e-6)

    def test_torch_label_shape_validation(self) -> None:
        torch = _torch()
        from probly.conformal_scores.aps._common import _aps_score_dispatch as dispatch  # noqa: PLC0415

        probs = torch.tensor([[0.1, 0.6, 0.3], [0.4, 0.4, 0.2]])
        bad_labels = torch.tensor([[0, 1, 2]])
        with pytest.raises(ValueError, match="y_cal must match probs"):
            dispatch(probs, bad_labels, randomized=False)

    def test_torch_rejects_zero_dim(self) -> None:
        torch = _torch()
        from probly.conformal_scores.aps._common import _aps_score_dispatch as dispatch  # noqa: PLC0415

        with pytest.raises(ValueError, match="at least one dimension"):
            dispatch(torch.tensor(0.5))

    def test_jax_matches_numpy_when_not_randomized(self) -> None:
        _, jnp = _jax_modules()
        from probly.conformal_scores.aps._common import (  # noqa: PLC0415
            _aps_score_dispatch as dispatch,
            compute_aps_score_numpy,
        )

        probs_np = np.array([[0.2, 0.5, 0.3], [0.7, 0.2, 0.1]])
        labels_np = np.array([1, 0])
        expected = compute_aps_score_numpy(probs_np, labels_np, randomized=False)

        scores = dispatch(jnp.asarray(probs_np), jnp.asarray(labels_np), randomized=False)
        np.testing.assert_allclose(np.asarray(scores), expected, atol=1e-6)

    def test_jax_label_shape_validation(self) -> None:
        _, jnp = _jax_modules()
        from probly.conformal_scores.aps._common import _aps_score_dispatch as dispatch  # noqa: PLC0415

        probs = jnp.asarray([[0.2, 0.5, 0.3]])
        with pytest.raises(ValueError, match="y_cal must match"):
            dispatch(probs, jnp.asarray([[0, 1]]), randomized=False)

    def test_jax_rejects_zero_dim(self) -> None:
        _, jnp = _jax_modules()
        from probly.conformal_scores.aps._common import _aps_score_dispatch as dispatch  # noqa: PLC0415

        with pytest.raises(ValueError, match="at least one dimension"):
            dispatch(jnp.asarray(0.5))

    def test_torch_categorical_distribution_input(self) -> None:
        torch = _torch()
        from probly.conformal_scores.aps._common import (  # noqa: PLC0415
            _aps_score_dispatch as dispatch,
            compute_aps_score_numpy,
        )
        from probly.representation.distribution.torch_categorical import (  # noqa: PLC0415
            TorchProbabilityCategoricalDistribution,
        )

        probs = torch.tensor([[0.2, 0.5, 0.3]])
        dist = TorchProbabilityCategoricalDistribution(probs)
        labels = torch.tensor([2])
        expected = compute_aps_score_numpy(probs.numpy(), labels.numpy(), randomized=False)
        result = dispatch(dist, labels, randomized=False)
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-6)

    def test_torch_sample_input(self) -> None:
        torch = _torch()
        from probly.conformal_scores.aps._common import _aps_score_dispatch as dispatch  # noqa: PLC0415
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415

        tensor = torch.tensor([[0.2, 0.5, 0.3], [0.1, 0.1, 0.8]])
        sample = TorchSample(tensor, sample_dim=0)
        scores_via_sample = dispatch(sample, randomized=False)
        scores_via_tensor = dispatch(tensor, randomized=False)
        assert torch.allclose(scores_via_sample, scores_via_tensor)


class TestAPSFallback:
    """The APS dispatch raises for unknown types."""

    def test_unsupported_type_raises(self) -> None:
        from probly.conformal_scores.aps._common import _aps_score_dispatch  # noqa: PLC0415

        with pytest.raises(NotImplementedError, match="not implemented"):
            _aps_score_dispatch(object())
