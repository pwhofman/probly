"""Tests for ``probly.representer.sampler.sklearn``."""

from __future__ import annotations

import numpy as np
import pytest


class TestSamplerSklearn:
    """Sampling-preparation enforces a fitted sklearn estimator."""

    def test_unfitted_estimator_raises(self) -> None:
        pytest.importorskip("sklearn")
        from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

        from probly.representer.sampler._common import get_sampling_predictor  # noqa: PLC0415

        # Force registration of the sklearn handler.
        import probly.representer.sampler.sklearn  # noqa: F401, PLC0415

        unfitted = LogisticRegression()
        with pytest.raises(ValueError, match="must be fitted"):
            get_sampling_predictor(unfitted)

    def test_fitted_estimator_passes(self) -> None:
        pytest.importorskip("sklearn")
        from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

        from probly.representer.sampler._common import get_sampling_predictor  # noqa: PLC0415
        import probly.representer.sampler.sklearn  # noqa: F401, PLC0415

        clf = LogisticRegression()
        clf.fit(np.array([[0], [1], [2]]), np.array([0, 1, 0]))
        out, cleanup = get_sampling_predictor(clf)
        assert out is clf
        cleanup()
