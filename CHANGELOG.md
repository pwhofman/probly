# Changelog
This changelog is updated with every release of `probly`.

## Development

- Moved array-like protocols, backend function wrappers, and protected-axis infrastructure to `ndlike>=0.2.4`. Sample elementwise arithmetic now preserves sample axes and weights across NumPy, Torch, and JAX, and multi-output NumPy ufuncs preserve metadata per output. Reductions over the sample axis remove the sample wrapper regardless of `keepdims`.
- Added possiblity to create ensemble of torch models without resetting the weights of each model.
- Refactored Efficient Credal Prediction calibration into the library via `flexdispatch`. Added a optimized PyTorch bisection solver (`compute_efficient_credal_prediction_bounds`) that reduces calibration time from days to minutes, while preserving the legacy SciPy implementation as a fallback for NumPy arrays.

## 0.1.0 (2024-03-14)
Initial pre-release of `probly` without functionalities.
