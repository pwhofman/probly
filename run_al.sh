#!/usr/bin/env bash
# Active learning experiment sweep.
#
# Arm 1: 2 baselines x 3 strategies x 5 datasets x 10 seeds = 300 runs
# Arm 2: 11 UQ methods x 2 strategies x 5 datasets x 10 seeds = 1100 runs
# Total: 1400 runs
#
# Local smoke test (fast):
#   bash run_al.sh --smoke
#
# Results logged to wandb (cluster) or al_results.json (local smoke).
set -euo pipefail
export TORCHDYNAMO_DISABLE=1

if [[ "${1:-}" == "--smoke" ]]; then
    SEEDS=(0)
    DATASETS=(openml_6)
    COMMON="wandb.enabled=false save_results=true initial_size=100 query_size=100 n_iterations=3 epochs=5 device=cpu"
    echo "=== SMOKE TEST MODE ==="
else
    SEEDS=(0 1 2 3 4 5 6 7 8 9)
    DATASETS=(openml_6 openml_155 openml_156 fashion_mnist cifar10)
    COMMON="wandb.enabled=true save_results=false"
fi

# Arm 1: tailored AL baselines
BASELINE_METHODS=(plain ensemble)
BASELINE_STRATEGIES=(random margin badge)

for seed in "${SEEDS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for method in "${BASELINE_METHODS[@]}"; do
      for strategy in "${BASELINE_STRATEGIES[@]}"; do
        echo "=== BASELINE / $method / $strategy / $dataset / seed=$seed ==="
        uv run python -m probly_benchmark.active_learning \
          method=$method al_strategy=$strategy al_dataset=$dataset \
          seed=$seed $COMMON
      done
    done
  done
done

# Arm 2: uncertainty methods (uncertainty + random ablation)
# efficient_credal_prediction: predict_single produces invalid distributions (probly#xxx)
UQ_METHODS=(dropout dropconnect bayesian ddu ensemble
            evidential_classification posterior_network
            credal_ensembling credal_relative_likelihood)

for seed in "${SEEDS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for method in "${UQ_METHODS[@]}"; do
      for strategy in uncertainty random; do
        echo "=== UQ / $method / $strategy / $dataset / seed=$seed ==="
        uv run python -m probly_benchmark.active_learning \
          method=$method al_strategy=$strategy al_dataset=$dataset \
          seed=$seed $COMMON
      done
    done
  done
done

echo "=== Done ==="
