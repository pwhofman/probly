#!/usr/bin/env bash
# Active learning experiment sweep.
#
# Arm 1: Tailored AL baselines
#   2 methods (plain, ensemble) x 3 strategies (margin, badge, random)
#
# Arm 2: Uncertainty methods
#   10 methods x 4 strategies (uncertainty-EU, uncertainty-TU, margin, random)
#   Exception: DDU has no TU (density-based EU only)
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

# ---------------------------------------------------------------------------
# Arm 1: tailored AL baselines
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Arm 2: uncertainty methods
# ---------------------------------------------------------------------------
# Not included (upstream issues):
#   efficient_credal_prediction — canonical_element produces invalid distributions
#   het_nets — aleatoric-only decomposition, no epistemic signal for AL
#   batchensemble / natural_posterior_network — no representer support

# Methods that support both EU and TU decomposition
EU_TU_METHODS=(dropout dropconnect bayesian dare ensemble
               evidential_classification posterior_network
               credal_ensembling credal_relative_likelihood)

# Methods that only support EU (no TotalUncertainty in decomposition)
EU_ONLY_METHODS=(ddu)

# --- EU + margin + random (all methods) ---
ALL_UQ_METHODS=("${EU_TU_METHODS[@]}" "${EU_ONLY_METHODS[@]}")

for seed in "${SEEDS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for method in "${ALL_UQ_METHODS[@]}"; do
      for strategy in uncertainty margin random; do
        echo "=== UQ / $method / $strategy (EU) / $dataset / seed=$seed ==="
        uv run python -m probly_benchmark.active_learning \
          method=$method al_strategy=$strategy al_dataset=$dataset \
          seed=$seed $COMMON
      done
    done
  done
done

# --- TU ablation (methods that support it) ---
for seed in "${SEEDS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for method in "${EU_TU_METHODS[@]}"; do
      echo "=== UQ / $method / uncertainty (TU) / $dataset / seed=$seed ==="
      uv run python -m probly_benchmark.active_learning \
        method=$method al_strategy=uncertainty al_dataset=$dataset \
        uncertainty_decomposition=TU \
        seed=$seed $COMMON
    done
  done
done

echo "=== Done ==="
