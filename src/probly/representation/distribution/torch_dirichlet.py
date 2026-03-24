"""Torch-based distribution representation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Self, override

import numpy as np
import torch
from torch.special import digamma, gammaln

from probly.representation.distribution.common import DirichletDistribution, create_distribution
from probly.representation.sampling.torch_sample import TorchTensorSample


@dataclass(frozen=True, slots=True, weakref_slot=True)
class TorchDirichlet(DirichletDistribution):
    """A Dirichlet distribution stored as a torch tensor.

    Shape: (..., num_classes)
    The last axis represents the category dimension.
    """

    alphas: torch.Tensor

    def __post_init__(self) -> None:
        """Validate the concentration parameters."""
        if not isinstance(self.alphas, torch.Tensor):
            msg = "alphas must be a torch.Tensor"
            raise TypeError(msg)

        if self.alphas.ndim < 1:
            msg = "alphas must have at least one dimension."
            raise ValueError(msg)

        if torch.any(self.alphas <= 0):
            msg = "alphas must be strictly positive."
            raise ValueError(msg)

        if self.alphas.shape[-1] < 2:
            msg = "Dirichlet distribution requires at least 2 classes."
            raise ValueError(msg)

    @classmethod
    def from_tensor(cls, alphas: torch.Tensor | list, dtype: torch.dtype | None = None) -> Self:
        """Create a TorchDirichlet from a tensor or nested list."""
        return cls(alphas=torch.as_tensor(alphas, dtype=dtype))

    @classmethod
    def from_numpy(cls, alphas: np.ndarray | list, dtype: torch.dtype | None = None) -> Self:
        """Create a TorchDirichlet from a numpy array or nested list."""
        return cls(alphas=torch.as_tensor(alphas, dtype=dtype))

    def __len__(self) -> int:
        """Return the length along the first dimension."""
        if self.ndim == 0:
            msg = "len() of unsized distribution"
            raise TypeError(msg)
        return len(self.alphas)

    @property
    def dtype(self) -> torch.dtype:
        """The data type of the underlying tensor."""
        return self.alphas.dtype

    @property
    def device(self) -> torch.device:
        """The device of the underlying tensor."""
        return self.alphas.device

    @property
    def ndim(self) -> int:
        """Number of batch dimensions (excluding category axis)."""
        return self.alphas.ndim - 1

    @property
    def shape(self) -> torch.Size:
        """Batch shape (excluding category axis)."""
        return self.alphas.shape[:-1]

    @property
    def size(self) -> int:
        """The total number of distributions."""
        return math.prod(self.shape) if self.shape else 1

    @property
    def T(self) -> Self:  # noqa: N802
        """The transposed version of the distribution (reverses all axes of alphas)."""
        return type(self)(alphas=self.alphas.T)

    @property
    def mT(self) -> Self:  # noqa: N802
        """Transpose the last two dimensions of alphas."""
        return type(self)(alphas=self.alphas.mT)

    @property
    @override
    def entropy(self) -> torch.Tensor:
        """Compute the entropy of the Dirichlet distribution."""
        alpha_0 = self.alphas.sum(dim=-1)
        K = self.alphas.shape[-1]  # noqa: N806

        log_beta = gammaln(self.alphas).sum(dim=-1) - gammaln(alpha_0)
        digamma_sum = (alpha_0 - K) * digamma(alpha_0)
        digamma_individual = ((self.alphas - 1) * digamma(self.alphas)).sum(dim=-1)

        return log_beta + digamma_sum - digamma_individual

    def sample(self, num_samples: int = 1) -> TorchTensorSample:
        """Sample from the Dirichlet distribution (torch backend).

        Args:
            num_samples: Number of samples to draw.

        Returns:
            TorchTensorSample with sample_dim=0, tensor shape (num_samples, *batch_shape, num_classes).
        """
        # torch.distributions.Dirichlet does not accept a generator; use manual gamma sampling
        gammas = torch._standard_gamma(  # noqa: SLF001
            self.alphas.expand(num_samples, *self.alphas.shape),
        )
        samples = gammas / gammas.sum(dim=-1, keepdim=True)
        return TorchTensorSample(tensor=samples, sample_dim=0)

    def __setitem__(
        self,
        index: int | slice | tuple | torch.Tensor,
        value: Self | torch.Tensor,
    ) -> None:
        """Set a subset of the alphas by index (mutates the underlying tensor in-place)."""
        if isinstance(value, TorchDirichlet):
            self.alphas[index] = value.alphas
        else:
            self.alphas[index] = value

    def clone(self) -> Self:
        """Create a deep copy of the distribution."""
        return type(self)(alphas=self.alphas.clone())

    def to(self, *args: Any, **kwargs: Any) -> Self:  # noqa: ANN401
        """Move and/or cast the underlying tensor. See `torch.Tensor.to` for details."""
        return type(self)(alphas=self.alphas.to(*args, **kwargs))

    def numpy(
        self,
        dtype: Any = None,  # noqa: ANN401
        copy: bool | None = None,
    ) -> Any:  # noqa: ANN401
        """Return the underlying alphas as a numpy array."""
        tensor = self.alphas.detach().cpu()
        return np.asarray(tensor, dtype=dtype, copy=copy)

    def detach(self) -> Self:
        """Return a new TorchDirichlet detached from the computation graph."""
        return type(self)(alphas=self.alphas.detach())

    def cpu(self) -> Self:
        """Move the distribution to CPU."""
        return type(self)(alphas=self.alphas.cpu())

    def cuda(self, device: int | torch.device | None = None) -> Self:
        """Move the distribution to a CUDA device."""
        return type(self)(alphas=self.alphas.cuda(device))

    def __getitem__(self, index: Any) -> Self | torch.Tensor:  # noqa: ANN401
        """Index into the batch dimensions."""
        result = self.alphas[index]
        if isinstance(result, torch.Tensor) and result.ndim >= 1:
            return type(self)(alphas=result)
        return result

    def __eq__(self, value: TorchDirichlet | Any) -> torch.Tensor:  # type: ignore[override]  # noqa: ANN401
        """Vectorized equality comparison of alphas."""
        if isinstance(value, TorchDirichlet):
            return self.alphas == value.alphas
        return self.alphas == value

    def __hash__(self) -> int:
        """Compute the hash of the distribution."""
        return super().__hash__()

    def __repr__(self) -> str:
        """Human-readable representation."""
        return f"{type(self).__name__}(alphas={self.alphas})"


create_distribution.register(torch.Tensor, TorchDirichlet)
