"""Tests for the distribution common factory fallbacks."""

from __future__ import annotations

import numpy as np
import pytest


class TestDistributionFactoryFallbacks:
    """The distribution factories raise for unregistered input types."""

    def test_create_categorical_distribution_raises(self) -> None:
        from probly.representation.distribution._common import create_categorical_distribution  # noqa: PLC0415

        with pytest.raises(NotImplementedError, match="categorical"):
            create_categorical_distribution(object())

    def test_create_categorical_distribution_from_logits_raises(self) -> None:
        from probly.representation.distribution._common import (  # noqa: PLC0415
            create_categorical_distribution_from_logits,
        )

        with pytest.raises(NotImplementedError, match="categorical"):
            create_categorical_distribution_from_logits(object())

    def test_create_bernoulli_distribution_raises(self) -> None:
        from probly.representation.distribution._common import create_bernoulli_distribution  # noqa: PLC0415

        with pytest.raises(NotImplementedError, match="Bernoulli"):
            create_bernoulli_distribution(object())

    def test_create_bernoulli_distribution_from_logits_raises(self) -> None:
        from probly.representation.distribution._common import (  # noqa: PLC0415
            create_bernoulli_distribution_from_logits,
        )

        with pytest.raises(NotImplementedError, match="Bernoulli"):
            create_bernoulli_distribution_from_logits(object())

    def test_create_dirichlet_distribution_from_alphas_raises(self) -> None:
        from probly.representation.distribution._common import (  # noqa: PLC0415
            create_dirichlet_distribution_from_alphas,
        )

        with pytest.raises(NotImplementedError, match="Dirichlet"):
            create_dirichlet_distribution_from_alphas(object())

    def test_create_dirichlet_mixture_distribution_from_alphas_and_weights_raises(self) -> None:
        from probly.representation.distribution._common import (  # noqa: PLC0415
            create_dirichlet_mixture_distribution_from_alphas_and_weights,
        )

        with pytest.raises(NotImplementedError, match="Dirichlet mixture"):
            create_dirichlet_mixture_distribution_from_alphas_and_weights(object(), object())

    def test_create_gaussian_distribution_raises(self) -> None:
        from probly.representation.distribution._common import create_gaussian_distribution  # noqa: PLC0415

        with pytest.raises(NotImplementedError, match="Gaussian"):
            create_gaussian_distribution(object())

    def test_create_categorical_passthrough(self) -> None:
        """Passing in an existing CategoricalDistribution returns it unchanged."""
        from probly.representation.distribution._common import create_categorical_distribution  # noqa: PLC0415
        from probly.representation.distribution.array_categorical import (  # noqa: PLC0415
            ArrayProbabilityCategoricalDistribution,
        )

        d = ArrayProbabilityCategoricalDistribution(array=np.array([[0.5, 0.5]]))
        assert create_categorical_distribution(d) is d

    def test_create_categorical_from_logits_passthrough(self) -> None:
        from probly.representation.distribution._common import (  # noqa: PLC0415
            create_categorical_distribution_from_logits,
        )
        from probly.representation.distribution.array_categorical import (  # noqa: PLC0415
            ArrayLogitCategoricalDistribution,
        )

        d = ArrayLogitCategoricalDistribution(array=np.array([[0.0, 0.0]]))
        assert create_categorical_distribution_from_logits(d) is d
