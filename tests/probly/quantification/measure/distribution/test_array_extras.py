"""Cover the array-input branches of distribution-uncertainty measures.

Each registered measure accepts either an ``ArrayDirichletDistribution`` /
``ArrayGaussianDistribution`` wrapper or a raw numpy array of the underlying
parameters. The wrapper branches are well-tested elsewhere; this module
exercises the raw-array (``isinstance == False``) branches and the
sample-based variants.
"""

from __future__ import annotations

import numpy as np

from probly.quantification.measure.distribution.array import (
    array_dirichlet_conditional_entropy,
    array_dirichlet_entropy,
    array_dirichlet_max_probability_complement_of_expected,
    array_dirichlet_mutual_information,
    array_dirichlet_vacuity,
    array_gaussian_entropy,
)


class TestRawArrayDirichletPaths:
    """Each Dirichlet measure accepts a raw numpy array of alphas."""

    def test_entropy_with_raw_alphas(self) -> None:
        alphas = np.array([[2.0, 3.0, 5.0], [1.0, 1.0, 1.0]])
        # Wrapper and raw-array should agree.
        from probly.representation.distribution.array_dirichlet import ArrayDirichletDistribution  # noqa: PLC0415

        wrapper_result = array_dirichlet_entropy(ArrayDirichletDistribution(alphas=alphas))
        raw_result = array_dirichlet_entropy(alphas)
        np.testing.assert_allclose(raw_result, wrapper_result, atol=1e-10)

    def test_conditional_entropy_with_raw_alphas(self) -> None:
        alphas = np.array([[2.0, 3.0, 5.0]])
        from probly.representation.distribution.array_dirichlet import ArrayDirichletDistribution  # noqa: PLC0415

        wrapper_result = array_dirichlet_conditional_entropy(ArrayDirichletDistribution(alphas=alphas))
        raw_result = array_dirichlet_conditional_entropy(alphas)
        np.testing.assert_allclose(raw_result, wrapper_result, atol=1e-10)

    def test_mutual_information_with_raw_alphas(self) -> None:
        alphas = np.array([[2.0, 3.0, 5.0]])
        from probly.representation.distribution.array_dirichlet import ArrayDirichletDistribution  # noqa: PLC0415

        wrapper_result = array_dirichlet_mutual_information(ArrayDirichletDistribution(alphas=alphas))
        raw_result = array_dirichlet_mutual_information(alphas)
        np.testing.assert_allclose(raw_result, wrapper_result, atol=1e-10)

    def test_vacuity_with_raw_alphas(self) -> None:
        alphas = np.array([[1.0, 1.0, 1.0]])
        # vacuity = K / alpha_0 = 3 / 3 = 1
        result = array_dirichlet_vacuity(alphas)
        np.testing.assert_allclose(result, [1.0])

    def test_max_probability_complement_of_expected_with_raw_alphas(self) -> None:
        alphas = np.array([[1.0, 1.0, 8.0]])
        # mean = [1/10, 1/10, 8/10], max = 0.8, complement = 0.2
        result = array_dirichlet_max_probability_complement_of_expected(alphas)
        np.testing.assert_allclose(result, [0.2], atol=1e-10)


class TestRawArrayGaussianPaths:
    """Gaussian entropy accepts a raw numpy array of variances."""

    def test_entropy_with_raw_var(self) -> None:
        var = np.array([1.0, 4.0, 9.0])
        from probly.representation.distribution.array_gaussian import ArrayGaussianDistribution  # noqa: PLC0415

        wrapper_result = array_gaussian_entropy(ArrayGaussianDistribution(mean=np.zeros_like(var), var=var))
        raw_result = array_gaussian_entropy(var)
        np.testing.assert_allclose(raw_result, wrapper_result, atol=1e-10)


class TestGaussianSamplePaths:
    """Sample-based Gaussian uncertainty measures."""

    def _gaussian_sample(self, n_samples: int = 5, batch: int = 3):
        from probly.representation.distribution.array_gaussian import (  # noqa: PLC0415
            ArrayGaussianDistribution,
            ArrayGaussianDistributionSample,
        )

        rng = np.random.default_rng(0)
        means = rng.normal(size=(n_samples, batch))
        variances = np.abs(rng.normal(size=(n_samples, batch))) + 0.1
        gaussians = ArrayGaussianDistribution(mean=means, var=variances)
        return ArrayGaussianDistributionSample(array=gaussians, sample_axis=0)

    def test_entropy_of_expected_predictive_distribution(self) -> None:
        from probly.quantification.measure.distribution.array import (  # noqa: PLC0415
            array_gaussian_sample_entropy_of_expected_predictive_distribution,
        )

        sample = self._gaussian_sample()
        result = array_gaussian_sample_entropy_of_expected_predictive_distribution(sample)
        # One scalar entropy per batch element.
        assert result.shape == (3,)
        assert np.isfinite(result).all()

    def test_conditional_entropy(self) -> None:
        from probly.quantification.measure.distribution.array import (  # noqa: PLC0415
            array_gaussian_sample_conditional_entropy,
        )

        sample = self._gaussian_sample()
        result = array_gaussian_sample_conditional_entropy(sample)
        assert result.shape == (3,)
        assert np.isfinite(result).all()

    def test_mutual_information(self) -> None:
        from probly.quantification.measure.distribution.array import (  # noqa: PLC0415
            array_gaussian_sample_mutual_information,
        )

        sample = self._gaussian_sample()
        result = array_gaussian_sample_mutual_information(sample)
        assert result.shape == (3,)
        # MI = total - aleatoric, both finite -> result finite.
        assert np.isfinite(result).all()
