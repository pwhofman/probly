"""This module defines the dataclasses for different representations of uncertainty."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import torch


@dataclass
class Representation(ABC):
    """Base dataclass for representations of uncertainty.

    Attributes:
        data: The data representing the uncertainty.
        problem: The type of problem (e.g. classification, regression).
        n_instances: The number of data instances.
        n_samples: The number of samples of first-order distributions.
        n_targets: The number of targets to predict.
    """

    data: np.ndarray
    problem: Problem | str  # e.g. 'classification', 'regression'

    # Derived attributes
    n_instances: int = field(init=False)
    n_samples: int = field(init=False)
    n_targets: int = field(init=False)

    def __post_init__(self) -> None:
        """Post init method to convert data to numpy array (if possible) and set derived attributes."""
        # Convert torch tensor to a numpy array if necessary, at this moment we support only torch and numpy
        if isinstance(self.data, torch.Tensor):
            self.data = self.data.detach().cpu().numpy()
        elif isinstance(self.data, np.ndarray):
            pass
        else:
            raise TypeError("data must be a numpy array or torch tensor")

        # Set the derived attributes / fields
        self.n_instances = self.data.shape[0]
        self.n_samples = self.data.shape[1]
        self.n_targets = self.data.shape[2]


@dataclass
class CredalRepresentation(Representation):
    """Dataclass representing a credal set.

    Attributes:
        data: The data representing the credal set.
        problem: The type of problem (e.g. classification, regression).
        n_instances: The number of data instances.
        n_samples: The number of samples of first-order distributions.
        n_targets: The number of targets to predict.
        structure: The structure of the credal set (e.g. convex hull, minmax interval, finite approx.).
    """

    structure: CredalStructure | str  # convex hull, minmax interval, finite approx.


@dataclass
class DistributionRepresentation(Representation):
    """Dataclass representing a second-order distribution.

    Attributes:
        data: The data representing the second-order distribution.
        problem: The type of problem (e.g. classification, regression).
        n_instances: The number of data instances.
        n_samples: The number of samples of first-order distributions.
        n_targets: The number of targets to predict.
        first_order_distribution: The first-order distribution.
        second_order_distribution: The second-order distribution.
    """

    first_order_distribution: DistributionFirstOrder | str
    second_order_distribution: DistributionSecondOrder | str


class Problem(str, Enum):
    """Enum representing the type of problem."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class CredalStructure(str, Enum):
    """Enum representing the structure of the credal set."""

    CONVEX_HULL = "convex_hull"
    INTERVAL = "interval"
    FINITE = "finite"


class DistributionFirstOrder(str, Enum):
    """Enum representing the first-order distribution."""

    CATEGORICAL = "categorical"
    NORMAL = "normal"


class DistributionSecondOrder(str, Enum):
    """Enum representing the second-order distribution."""

    DIRICHLET = "dirichlet"
    SAMPLES = "samples"
