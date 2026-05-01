"""Collection of fully-qualified type names for lazy type checking."""

from __future__ import annotations

TORCH_MODULE = "torch.nn.modules.module.Module"
TORCH_MODULE_LIST = "torch.nn.modules.container.ModuleList"
TORCH_TENSOR = "torch.Tensor"
TORCH_TENSOR_LIKE = "probly.representation.torch_like.TorchLikeImplementation"
TORCH_SAMPLE = "probly.representation.sample.torch.TorchSample"
TORCH_CATEGORICAL_DISTRIBUTION = "probly.representation.distribution.torch_categorical.TorchCategoricalDistribution"

FLAX_MODULE = "flax.nnx.module.Module"
FLAX_LIST = "flax.nnx.list.List"
JAX_ARRAY = "jax.Array"
JAX_ARRAY_LIKE = "probly.representation.jax_like.JaxLikeImplementation"

SKLEARN_MODULE = "sklearn.base.BaseEstimator"
SKLEARN_CALIBRATED_CLASSIFIER_CV = "sklearn.calibration.CalibratedClassifierCV"

RIVER_ARF_CLASSIFIER = "river.forest.adaptive_random_forest.ARFClassifier"
RIVER_ARF_REGRESSOR = "river.forest.adaptive_random_forest.ARFRegressor"

LAPLACE_BASE = "laplace.baselaplace.BaseLaplace"
