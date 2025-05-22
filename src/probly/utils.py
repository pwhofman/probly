"""General utility functions for all other modules."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Iterable


def powerset(iterable: Iterable[int]) -> list[tuple[()]]:
    """Generate the power set of a given iterable.

    Args:
        iterable: Iterable
    Returns:
        List[tuple], power set of the given iterable

    """
    s = list(iterable)
    return list(itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1)))


def capacity(q: np.ndarray, a: Iterable[int]) -> np.ndarray:
    """Compute the capacity of set q given set a.

    Args:
        q: numpy.ndarray, shape (n_instances, n_samples, n_classes)
        a: Iterable, shape (n_classes,), indices indicating subset of classes
    Returns:
        min_capacity: numpy.ndarray, shape (n_instances,), capacity of q given a

    """
    selected_sum = np.sum(q[:, :, a], axis=2)
    min_capacity = np.min(selected_sum, axis=1)
    return min_capacity


def moebius(q: np.ndarray, a: Iterable[int]) -> np.ndarray:
    """Compute the Moebius function of a set q given a set a.

    Args:
        q: numpy.ndarray of shape (num_samples, num_members, num_classes)
        a: numpy.ndarray, shape (n_classes,), indices indicating subset of classes
    Returns:
        m_a: numpy.ndarray, shape (n_instances,), moebius value of q given a

    """
    ps_a = powerset(a)  # powerset of A
    ps_a.pop(0)  # remove empty set
    m_a = np.zeros(q.shape[0])
    for b in ps_a:
        dl = len(set(a) - set(b))
        m_a += ((-1) ** dl) * capacity(q, b)
    return m_a


def differential_entropy_gaussian(sigma2: float | np.ndarray, base: float = 2) -> float | np.ndarray:
    """Compute the differential entropy of a Gaussian distribution given the variance.

    https://en.wikipedia.org/wiki/Differential_entropy
    Args:
        sigma2: float or numpy.ndarray shape (n_instances,), variance of the Gaussian distribution
        base: float, base of the logarithm
    Returns:
        diff_ent: float or numpy.ndarray shape (n_instances,), differential entropy of the Gaussian distribution
    """
    return 0.5 * np.log(2 * np.pi * np.e * sigma2) / np.log(base)


def kl_divergence_gaussian(
    mu1: float | np.ndarray,
    sigma21: float | np.ndarray,
    mu2: float | np.ndarray,
    sigma22: float | np.ndarray,
    base: float = 2,
) -> float | np.ndarray:
    """Compute the KL-divergence between two Gaussian distributions.

    https://en.wikipedia.org/wiki/Kullback-Leibler_divergence#Examples
    Args:
        mu1: float or numpy.ndarray shape (n_instances,), mean of the first Gaussian distribution
        sigma21: float or numpy.ndarray shape (n_instances,), variance of the first Gaussian distribution
        mu2: float or numpy.ndarray shape (n_instances,), mean of the second Gaussian distribution
        sigma22: float or numpy.ndarray shape (n_instances,), variance of the second Gaussian distribution
        base: float, base of the logarithm
    Returns:
        kl_div: float or numpy.ndarray shape (n_instances,), KL-divergence between the two Gaussian distributions
    """
    kl_div = 0.5 * np.log(sigma22 / sigma21) / np.log(base) + (sigma21 + (mu1 - mu2) ** 2) / (2 * sigma22) - 0.5
    return kl_div


def torch_reset_all_parameters(module: torch.nn.Module) -> None:
    """Reset all parameters of a torch module.

    Args:
        module: torch.nn.Module, module to reset parameters

    """
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()
    for child in module.children():
        if hasattr(child, "reset_parameters"):
            child.reset_parameters()


@torch.no_grad()
def torch_collect_outputs(
    model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect outputs and targets from a model for a given data loader.

    Args:
        model: torch.nn.Module, model to collect outputs from
        loader: torch.utils.data.DataLoader, data loader to collect outputs from
        device: torch.device, device to move data to
    Returns:
        outputs: torch.Tensor, shape (n_instances, n_classes), model outputs
        targets: torch.Tensor, shape (n_instances,), target labels
    """
    outputs = torch.empty(0, device=device)
    targets = torch.empty(0, device=device)
    for inpt, target in tqdm(loader):
        outputs = torch.cat((outputs, model(inpt.to(device))), dim=0)
        targets = torch.cat((targets, target.to(device)), dim=0)
    return outputs, targets


def temperature_softmax(logits: torch.Tensor, temperature: float | torch.Tensor) -> torch.Tensor:
    """Compute the softmax of logits with temperature scaling applied.

    Computes the softmax based on the logits divided by the temperature. Assumes that the last dimension
    of logits is the class dimension.

    Args:
        logits: torch.Tensor, shape (n_instances, n_classes), logits to apply softmax on
        temperature: float, temperature scaling factor
    Returns:
        ts: torch.Tensor, shape (n_instances, n_classes), softmax of logits with temperature scaling applied
    """
    ts = F.softmax(logits / temperature, dim=-1)
    return ts


def intersection_probability(probs: np.ndarray) -> np.ndarray:
    """Compute the intersection probability of a credal set based on intervals of lower and upper probabilities.

    Computes the intersection probability from https://arxiv.org/pdf/2201.01729.

    Args:
        probs: numpy.ndarray, shape (n_instances, n_samples, n_classes), credal sets
    Returns:
        int_probs: numpy.ndarray, shape (n_instances, n_classes), intersection probability of the credal sets
    """
    lower = np.min(probs, axis=1)
    upper = np.max(probs, axis=1)
    diff = upper - lower
    diff_sum = np.sum(diff, axis=1)
    lower_sum = np.sum(lower, axis=1)
    # Compute alpha for instances for which probability intervals are not empty, otherwise set alpha to 0.
    alpha = np.zeros(probs.shape[0])
    nonzero_idxs = diff_sum != 0
    alpha[nonzero_idxs] = (1 - lower_sum[nonzero_idxs]) / diff_sum[nonzero_idxs]
    int_probs = lower + alpha[:, None] * diff
    return int_probs
