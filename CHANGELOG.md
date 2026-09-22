# Changelog
This changelog is updated with every release of `probly`.

## Development

- Added possiblity to create ensemble of torch models without resetting the weights of each model.
- Refactored Efficient Credal Prediction calibration into the library via `flexdispatch`. Added a optimized PyTorch bisection solver (`compute_efficient_credal_prediction_bounds`) that reduces calibration time from days to minutes, while preserving the legacy SciPy implementation as a fallback for NumPy arrays.

## 0.1.0 (2024-03-14)
Initial pre-release of `probly` without functionalities.
