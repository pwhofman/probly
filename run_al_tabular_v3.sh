#!/usr/bin/env bash
# Streamlined tabular AL sweep.
#
# Two well-defined questions, mapped onto the two estimator paths in
# probly_benchmark.al_estimator:
#
# 1. AL strategy comparison on UQ-free baselines (BaselineALEstimator):
#    same backbone (plain MLP and deep ensemble), vary only the strategy.
#    Answers "which traditional AL strategy works on a baseline model?".
#       2 baselines * 3 strategies = 6 configs
#
# 2. UQ method comparison on uncertainty-vs-random (UQALEstimator):
#    every probly UQ method paired with `uncertainty` and `random`.
#    Answers "does this UQ method's score beat random selection?".
#       8 methods * 2 strategies = 16 configs
#       (ensemble appears here only with `uncertainty`; its `random`
#        baseline is already covered in block 1, so we omit the dup.)
#       8 methods * 2 - 1 = 15 configs
#
# Total: 6 + 15 = 21 configs * 10 seeds = 210 runs.
#
# Results are appended to ./al_results.json (POSIX-flock dedupe on
# (method, dataset, strategy, seed)). Plot with:
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

# Block 1: traditional AL strategies on UQ-free baselines.
BASELINE_METHODS=(plain ensemble)
BASELINE_STRATEGIES=(random margin badge)
for seed in "${SEEDS[@]}"; do
    for method in "${BASELINE_METHODS[@]}"; do
        for strategy in "${BASELINE_STRATEGIES[@]}"; do
            echo "=== BASELINE / $method / $strategy / seed=$seed ==="
            uv run python -m probly_benchmark.active_learning \
                method=$method al_strategy=$strategy seed=$seed $COMMON
        done
    done
done

# Block 2: UQ comparison -- uncertainty vs random.
# `ensemble` lives in both blocks: its `random` row is already in block 1,
# so we only run `uncertainty` here for it.
UQ_METHODS=(dropout ddu evidential_classification posterior_network \
    efficient_credal_prediction credal_ensembling credal_relative_likelihood)
for seed in "${SEEDS[@]}"; do
    for method in "${UQ_METHODS[@]}"; do
        for strategy in random uncertainty; do
            echo "=== UQ / $method / $strategy / seed=$seed ==="
            uv run python -m probly_benchmark.active_learning \
                method=$method al_strategy=$strategy seed=$seed $COMMON
        done
    done
    echo "=== UQ / ensemble / uncertainty / seed=$seed ==="
    uv run python -m probly_benchmark.active_learning \
        method=ensemble al_strategy=uncertainty seed=$seed $COMMON
done

echo "=== Done: 210 runs appended to al_results.json ==="
