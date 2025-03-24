import itertools

import numpy as np


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return list(itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1)))

def capacity(Q, A):
    """Computes the capacity of a set Q given a set A"""
    sum = np.sum(Q[:, :, A], axis=2)
    min = np.min(sum, axis=1)
    return min

def moebius(Q, A):
    """Computes the Moebius function of a set Q given a set A.
    Q: array of shape (num_samples, num_members, num_classes)
    A: set of indices
    """
    ps_B = powerset(A)  # powerset of A
    ps_B.pop(0)  # remove empty set
    m_A = np.zeros(Q.shape[0])
    for B in ps_B:
        dl = len(set(A) - set(B))
        m_A += ((-1) ** dl) * capacity(Q, B)
    return m_A

def differential_entropy_gaussian(sigma2, base=2):
    """
    Compute the differential entropy of a Gaussian distribution given the variance.
    https://en.wikipedia.org/wiki/Differential_entropy#
    Args:
        sigma2: float, variance of the Gaussian distribution
        base: float, base of the logarithm
    Returns:
        diff_ent: float, differential entropy of the Gaussian distribution
    """
    diff_ent = 0.5 * np.log(2 * np.pi * np.e * sigma2) / np.log(base)
    return diff_ent

def kl_divergence_gaussian(mu1, sigma21, mu2, sigma22, base=2):
    """
    Compute the KL-divergence between two Gaussian distributions.
    https://en.wikipedia.org/wiki/Kullback–Leibler_divergence#Examples
    Args:
        mu1: float, mean of the first Gaussian distribution
        sigma21: float, variance of the first Gaussian distribution
        mu2: float, mean of the second Gaussian distribution
        sigma22: float, variance of the second Gaussian distribution
        base: float, base of the logarithm
    Returns:
        kl_div: float, KL-divergence between the two Gaussian distributions
    """
    kl_div = (0.5 * np.log(sigma22 / sigma21) / np.log(base)
              + (sigma21 + (mu1 - mu2) ** 2) / (2 * sigma22)
              - 0.5)
    return kl_div

def torch_reset_all_parameters(model):
    """
    Reset all parameters of a torch model.
    Args:
        model: torch.nn.Module, model to reset parameters
    """
    for child in model.children():
        if hasattr(child, 'reset_parameters'):
            child.reset_parameters()
