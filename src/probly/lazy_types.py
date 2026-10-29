"""Collection of fully-qualified type names for lazy type checking."""

from __future__ import annotations

TORCH_MODULE = "torch.nn.modules.module.Module"
TORCH_MODULE_LIST = "torch.nn.modules.container.ModuleList"
TORCH_TENSOR = "torch.Tensor"
TORCH_TENSOR_LIKE = "ndlike.torch_like.TorchLikeImplementation"

TORCH_VALUE = (TORCH_TENSOR, TORCH_TENSOR_LIKE)
"""Lazy dispatch types for PyTorch tensors and tensor-like representations."""

FLAX_MODULE = "flax.nnx.module.Module"
FLAX_LIST = "flax.nnx.helpers.List"
FLAX_VARIABLE = "flax.nnx.variablelib.Variable"
JAX_ARRAY = "jax.Array"
JAX_TRACER = "jax._src.core.Tracer"
JAX_ARRAY_LIKE = "ndlike.jax_like.JaxLikeImplementation"

JAX_VALUE = (JAX_ARRAY, JAX_ARRAY_LIKE, JAX_TRACER)
"""Lazy dispatch types for JAX arrays, array-like representations, and traced values."""

SKLEARN_MODULE = "sklearn.base.BaseEstimator"
SKLEARN_CALIBRATED_CLASSIFIER_CV = "sklearn.calibration.CalibratedClassifierCV"

RIVER_ARF_CLASSIFIER = "river.forest.adaptive_random_forest.ARFClassifier"
RIVER_ARF_REGRESSOR = "river.forest.adaptive_random_forest.ARFRegressor"

LAPLACE_BASE = "laplace.baselaplace.BaseLaplace"
