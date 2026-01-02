"""Common utilities for CP and LazyDispatch Prediction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import numpy as np
    import numpy.typing as npt

from flax import nnx
import jax
import jax.numpy as jnp
import torch
import torch.nn.functional as F

from lazy_dispatch import lazydispatch


@lazydispatch
def predict_probs[T](model: Predictor, x: T) -> T:
    """Universal probability prediction function.

    Args:
        model: The model to use for prediction.
        x: Input data for which to predict probabilities.

    Returns:
        ArrayLike: Predicted probabilities.
    """
    # fallback for scikit-learn-like models
    if hasattr(model, "predict_probs"):
        return model.predict_probs(x)  # type: ignore[no-any-return]

    # fallback for other models (that only have predict)
    if hasattr(model, "predict"):
        return model.predict(x)  # type: ignore[no-any-return]

    msg = f"Model type {type(model)} is not supported directly. Please register it via @predict_probs.register"
    raise TypeError(msg)


@predict_probs.register(torch.nn.Module)
def predict_probs_torch(model: torch.nn.Module, x: Sequence[Any]) -> torch.Tensor:
    """Handler for PyTorch models: stays on GPU (Tensor)."""
    # decide device

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    # convert input to tensor on correct device
    x_tensor = x.to(device) if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float32, device=device)

    # set model to evaluation mode and disable gradient calculation
    model.eval()
    with torch.no_grad():
        logits = model(x_tensor)

        # fix tuple outputs
        if isinstance(logits, tuple):
            logits = logits[0]

        # use softmax for multiclass, sigmoid for binary
        # binary classification or flat output -> sigmoid
        probs = F.softmax(logits, dim=1) if logits.ndim > 1 and logits.shape[1] > 1 else torch.sigmoid(logits)

    return probs


@predict_probs.register(nnx.Module)
def predict_probs_flax(model: nnx.Module, x: Any) -> jnp.ndarray:  # noqa: ANN401
    """Predict probabilities for Flax NNX models."""
    if not hasattr(x, "shape"):
        x = jnp.asarray(x)

    if callable(model):
        logits = model(x)
    elif hasattr(model, "apply"):
        logits = model.apply(x)
    else:
        msg = "Model must be callable or expose apply()."
        raise TypeError(msg)

    # handle tuple outputs (some models return tuples)
    if isinstance(logits, tuple):
        logits = logits[0]

    # convert to probabilities
    probs = (
        jax.nn.softmax(logits, axis=-1) if logits.ndim > 1 and logits.shape[1] > 1 else jax.nn.sigmoid(logits)
    )  # for binary classification

    return probs


class Predictor(Protocol):
    """Protocol for models used with ConformalPredictor."""

    def __call__(self, x: Sequence[Any]) -> Sequence[Any]:
        """Callable method signature for conformal models."""


class ConformalPredictor(ABC):
    """Base class for Conformal Prediction."""

    def __init__(
        self,
        model: Predictor,
        nonconformity_func: Callable[..., npt.NDArray[np.floating]] | None = None,
    ) -> None:
        """Initialize the Conformal Predictor."""
        self.model = model
        self.conformity_func = nonconformity_func

        # saves the ML-model and nonconformity function
        self.nonconformity_scores: npt.NDArray[np.floating] | None = None
        self.threshold: float | None = None
        self.is_calibrated: bool = False

    @abstractmethod
    def predict(self, x_test: Sequence[Any], alpha: float) -> npt.NDArray[np.bool_]:
        """Generate prediction sets as boolean matrix (n_samples, n_classes) at given significance level.

        Args:
            x_test (Sequence[Any]): Test input data.
            alpha (float): Significance level for prediction sets.
        """
        raise NotImplementedError

    @abstractmethod
    def calibrate(self, x_cal: Sequence[Any], y_cal: Sequence[Any], alpha: float) -> float:
        """Virtual method to calibrate the calibration set.

        Args:
            x_cal (Sequence[Any]): Calibration input data.
            y_cal (Sequence[Any]): Calibration labels.
            alpha (float): The significance level.
        """
        raise NotImplementedError

    def __str__(self) -> str:
        """String representation of the class."""
        model_name = self.model.__class__.__name__
        status = "calibrated" if self.is_calibrated else "not calibrated"
        return f"{self.__class__.__name__}(model={model_name}, status={status})"
