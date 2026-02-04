"""Base Implementation For Platt-, Vector- and Temperature Scaling."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from flax import nnx
import jax
import jax.numpy as jnp

JaxDataLoader = Iterable[tuple[jax.Array, jax.Array]]


class ScalerFlax(nnx.Module, ABC):
    """Base class for Flax Scaling Implementations."""

    def __init__(self, base: nnx.Module, num_classes: int) -> None:
        """Initialize the scaler with a base module and number of classes.

        Args:
            base: The base model which outputs get calibrated.
            num_classes: The number of classes the model uses.
        """
        super().__init__()
        self.base = base
        self.num_classes = num_classes

    def __call__(self, x: jax.Array) -> jax.Array:
        """Call the scaler and returne scaled logits.

        Args:
            x: The input the scaled logits are produced from

        Returns:
            scaled_logits: The scaled logits based on the input.
        """
        logits = self.base(x)
        return self._scale_logits(logits)

    @abstractmethod
    def _scale_logits(self, logits: jax.Array) -> jax.Array:
        """Scale logits with optimized parameters."""
        raise NotImplementedError

    @abstractmethod
    def _init_opt_params(self) -> dict:
        """Return a dictionary of all parameters to optimize."""
        raise NotImplementedError

    @abstractmethod
    def _assign_opt_params(self, params: dict) -> None:
        """Apply optimized parameters from dictionary."""
        raise NotImplementedError

    @abstractmethod
    def _loss_with_params(self, params: dict, logits: jax.Array, labels: jax.Array) -> jax.Array:
        """Loss function purely functional."""
        raise NotImplementedError

    def _collect_logits_and_labels(self, model: nnx.Module, dataset: JaxDataLoader) -> tuple[jax.Array, jax.Array]:
        """Get all the models output for a given dataset and collect corresponding labels."""
        logits_list = []
        labels_list = []

        for inputs, targets in dataset:
            x = jax.device_put(inputs)
            y = jax.device_put(targets)

            logits = model(x)
            logits_list.append(logits)
            labels_list.append(y)

        logits = jnp.concatenate(logits_list, axis=0)
        labels = jnp.concatenate(labels_list, axis=0)

        return logits, labels

    def fit(self, calibration_set: JaxDataLoader, learning_rate: float = 0.01, max_iter: int = 50) -> None:
        """Optimizes the parameters based on the given calibration set.

        Args:
            calibration_set: The data set used for optimizing the parameters
            learning_rate: The learning rate used by the optimizer
            max_iter: The maximum steps the optimizer takes
        """
        logits, labels = self._collect_logits_and_labels(self.base, calibration_set)
        params = self._init_opt_params()

        def loss_fn(parameters: dict, logits: jax.Array, labels: jax.Array) -> jax.Array:
            return self._loss_with_params(parameters, logits, labels)

        value_and_grad = jax.value_and_grad(loss_fn)

        for _ in range(max_iter):
            _loss, grads = value_and_grad(params, logits, labels)
            params = jax.tree_util.tree_map(lambda w, g: w - learning_rate * g, params, grads)

        self._assign_opt_params(params)

    def predict(self, x: jax.Array) -> jax.Array:
        """Makes prediction based on the input and parameters.

        Args:
            x: The input Array to make predictions on

        Returns:
            probs: The calibrated probabilities
        """
        logits = self(x)

        if self.num_classes == 1:
            return jax.nn.sigmoid(logits)

        return jax.nn.softmax(logits, axis=-1)
