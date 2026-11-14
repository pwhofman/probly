# Changelog
This changelog is updated with every release of `probly`.

## Development

- Raised the minimum Python version to 3.13 for ndlike 0.5. Representation types now distinguish wrapper owners, arithmetic results, and shape results; DEUP, DUQ, and Mahalanobis array generics are bounded by `WeakArrayLike`. Torch Gaussian addition/subtraction and sign operations use ndlike dispatch with independent-Gaussian variance semantics, and Torch Gaussian/Dirichlet comparisons return batch masks. Torch sample augmented assignment now updates the existing storage through ndlike's `out=` dispatch. Gaussian arithmetic currently requires `alpha=1` and does not support `out=`.

- Updated compatibility with `ndlike>=0.5.0`, ty 0.0.101, and Ruff 0.17.5. DEUP batch operations now keep error scores aligned, token generation validates batch shapes explicitly, and representer initialization preserves subclass assignment hooks.
- Moved array-like protocols, backend function wrappers, and protected-axis infrastructure to `ndlike>=0.2.4`. Sample elementwise arithmetic now preserves sample axes and weights across NumPy, Torch, and JAX, and multi-output NumPy ufuncs preserve metadata per output. Reductions over the sample axis remove the sample wrapper regardless of `keepdims`.
- Added possiblity to create ensemble of torch models without resetting the weights of each model.
- Refactored Efficient Credal Prediction calibration into the library via `flexdispatch`. Added a optimized PyTorch bisection solver (`compute_efficient_credal_prediction_bounds`) that reduces calibration time from days to minutes, while preserving the legacy SciPy implementation as a fallback for NumPy arrays.

## 0.1.0 (2024-03-14)
Initial pre-release of `probly` without functionalities.
