import numpy as np
from ..utils import differential_entropy_gaussian, kl_divergence_gaussian

def total_variance(probs):
    """
    Computes the total uncertainty using variance-based measures.
    Assumes that the input is from a distribution over parameters of
    a normal distribution. The first element of the parameter vector is the mean
    and the second element is the variance.
    The total uncertainty is the variance of the mixture of normal distributions.
    Args:
        probs: numpy.ndarray, shape (n_instances, n_samples, (mu, sigma^2))
    Returns:
        tu: numpy.ndarray, shape (n_instances,)
    """
    tv = np.mean(probs[:, :, 1], axis=1) + np.var(probs[:, :, 0], axis=1)
    return tv

def expected_conditional_variance(probs):
    """
    Computes the aleatoric uncertainty using variance-based measures.
    Assumes that the input is from a distribution over parameters of
    a normal distribution. The first element of the parameter vector is the mean
    and the second element is the variance.
    The aleatoric uncertainty is the mean of the variance of the samples.
    Args:
        probs: numpy.ndarray, shape (n_instances, n_samples, (mu, sigma^2))
    Returns:
        au: numpy.ndarray, shape (n_instances,)
    """
    ecv = np.mean(probs[:, :, 1], axis=1)
    return ecv

def variance_conditional_expectation(probs):
    """
    Computes the epistemic uncertainty using variance-based measures.
    Assumes that the input is from a distribution over parameters of
    a normal distribution. The first element of the parameter vector is the mean
    and the second element is the variance.
    The epistemic uncertainty is the variance of the mean of the samples.
    Args:
        probs: numpy.ndarray, shape (n_instances, n_samples, (mu, sigma^2))
    Returns:
        eu: numpy.ndarray, shape (n_instances,)
    """
    vce = np.var(probs[:, :, 0], axis=1)
    return vce

def total_differential_entropy(probs):
    """
    Computes the epistemic uncertainty using entropy-based measures.
    Assumes that the input is from a distribution over parameters of
    a normal distribution. The first element of the parameter vector is the mean
    and the second element is the variance.
    The total uncertainty is the differential entropy of the mixture of normal distributions.
    Args:
        probs: numpy.ndarray, shape (n_instances, n_samples, (mu, sigma^2))
    Returns:
        tu: numpy.ndarray, shape (n_instances,)
    """
    sigma2_mean = np.mean(probs[:, :, 1], axis=1) + np.var(probs[:, :, 0], axis=1)
    tde = differential_entropy_gaussian(sigma2_mean)
    return tde

def conditional_differential_entropy(probs):
    """
    Computes the aleatoric uncertainty using entropy-based measures.
    Assumes that the input is from a distribution over parameters of
    a normal distribution. The first element of the parameter vector is the mean
    and the second element is the variance.
    The aleatoric uncertainty is the mean of the differential entropy of the samples.
    Args:
        probs: numpy.ndarray, shape (n_instances, n_samples, (mu, sigma^2))
    Returns:
        au: numpy.ndarray, shape (n_instances,)
    """
    cde = np.mean(differential_entropy_gaussian(probs[:, :, 1]), axis=1)
    return cde

def mutual_information(probs):
    """
    Computes the epistemic uncertainty using entropy-based measures.
    Assumes that the input is from a distribution over parameters of
    a normal distribution. The first element of the parameter vector is the mean
    and the second element is the variance.
    The epistemic uncertainty is the expected KL-divergence of the samples
    to the mean distribution.
    Args:
        probs: numpy.ndarray, shape (n_instances, n_samples, (mu, sigma^2))
    Returns:
        eu: numpy.ndarray, shape (n_instances,)
    """
    mu_mean = np.mean(probs[:, :, 0], axis=1)
    sigma2_mean = np.mean(probs[:, :, 1], axis=1) + np.var(probs[:, :, 0], axis=1)
    mu_mean = np.repeat(np.expand_dims(mu_mean, 1), repeats=probs.shape[1], axis=1)
    sigma2_mean = np.repeat(np.expand_dims(sigma2_mean, 1), repeats=probs.shape[1], axis=1)
    mi = np.mean(kl_divergence_gaussian(probs[:, :, 0], probs[:, :, 1], mu_mean, sigma2_mean), axis=1)
    return mi
