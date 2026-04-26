#!/usr/bin/env bash
# Fair-comparison tabular AL sweep.
#
# Decomposes the experiment into two clean questions:
#
# 1. AL strategy comparison: same backbone (plain MLP and deep ensemble),
#    vary only the strategy. Answers "which strategy works?".
#       2 backbones * 4 strategies = 8 configs
#
# 2. UQ method comparison: same strategy (uncertainty), vary only the UQ
#    method. Answers "which UQ produces useful AL signal?".
#       7 single-model and multi-model UQ methods * 1 strategy = 7 configs
#
# Total: 8 + 7 = 15 configs * 10 seeds = 150 runs (vs. 320 in the full
# Cartesian product).
#
# Results are appended to ./al_results.json. Plot with:
#   uv run python -m probly_benchmark.plot.plot_al --output=al_accuracy.png
#   uv run python -m probly_benchmark.plot.plot_al --metric=ece --output=al_ece.png

set -euo pipefail
export TORCHDYNAMO_DISABLE=1

COMMON="al_dataset=openml_6 \
    wandb.enabled=false \
    save_results=true \
    initial_size=100 \
    query_size=100 \
    n_iterations=10 \
    epochs=20 \
    device=cpu"

SEEDS=(0 1 2 3 4 5 6 7 8 9)
STRATEGIES=(random margin badge uncertainty)

# Block 1: AL-strategy comparison on UQ-free backbones
BASELINE_BACKBONES=(plain ensemble)
for seed in "${SEEDS[@]}"; do
    for method in "${BASELINE_BACKBONES[@]}"; do
        for strategy in "${STRATEGIES[@]}"; do
            echo "=== STRAT-CMP / $method / $strategy / seed=$seed ==="
            uv run python -m probly_benchmark.active_learning \
                method=$method al_strategy=$strategy seed=$seed $COMMON
        done
    done
done

# Block 2: UQ-method comparison with the strategy that actually uses UQ
UQ_METHODS=(dropout ddu evidential_classification posterior_network \
    efficient_credal_prediction credal_ensembling credal_relative_likelihood)
for seed in "${SEEDS[@]}"; do
    for method in "${UQ_METHODS[@]}"; do
        echo "=== UQ-CMP / $method / uncertainty / seed=$seed ==="
        uv run python -m probly_benchmark.active_learning \
            method=$method al_strategy=uncertainty seed=$seed $COMMON
    done
done

echo "=== Done: 150 runs appended to al_results.json ==="
