"""Collection of fully-qualified type names for lazy type checking."""

from __future__ import annotations

TORCH_MODULE = "torch.nn.modules.module.Module"
TORCH_MODULE_LIST = "torch.nn.modules.container.ModuleList"
TORCH_TENSOR = "torch.Tensor"
TORCH_TENSOR_LIKE = "probly.representation.torch_like.TorchTensorLikeImplementation"

FLAX_MODULE = "flax.nnx.module.Module"
FLAX_LIST = "flax.nnx.list.List"
JAX_ARRAY = "jax.Array"
JAX_ARRAY_LIKE = "probly.representation.jax_like.JaxArrayLikeImplementation"

SKLEARN_MODULE = "sklearn.base.BaseEstimator"
