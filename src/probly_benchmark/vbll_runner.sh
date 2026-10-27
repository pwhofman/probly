#!/usr/bin/env bash
# CIFAR-10 active learning sweep for the VBLL method only, seeds 0-2 (12 runs):
# margin, random, uncertainty:epistemic, uncertainty:total per seed -- the same
# four strategies each UQ method gets in runner.sh.
#
# Preview without launching anything:   DRY_RUN=1 ./src/probly_benchmark/vbll_runner.sh
# Pin to one GPU (recommended while     AL_DEVICE=cuda:1 ./src/probly_benchmark/vbll_runner.sh
# the main campaign is running):
#
# Runs land in wandb probly/rebuttal-runs. VBLL is not in the run_al_sweep.py
# grid (kept byte-identical to the published tabular results), so combos are
# launched directly and there is NO skip logic -- rerunning repeats all 12 runs.
# Run inside tmux/screen; runs are sequential on one device.
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."

SEEDS=(0 1 2)
WANDB_PROJECT="${WANDB_PROJECT:-rebuttal-runs}"

DIRECT=(
    # Same optimizer recipe as every other method; the gradient clipping VBLL
    # needs for stability comes from configs/method/vbll.yaml (grad_clip_norm).
    method=vbll
    al_dataset=cifar10
    wandb.enabled=true
    "wandb.project=$WANDB_PROJECT"
    wandb.entity=probly
    save_results=false
)
[[ -z "${AL_DEVICE:-}" ]] || DIRECT+=("device=$AL_DEVICE")

run_direct() {
    local cmd=(uv run python -m probly_benchmark.active_learning "${DIRECT[@]}" "$@")
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        printf '  $ %s\n' "${cmd[*]}"
    else
        "${cmd[@]}" || echo "  (failed: $* -- continuing)"
    fi
}

echo "=== vbll: $((4 * ${#SEEDS[@]})) runs ==="
for seed in "${SEEDS[@]}"; do
    for strategy in margin random; do
        run_direct "al_strategy=$strategy" "seed=$seed"
    done
    for notion in epistemic total; do
        run_direct al_strategy=uncertainty "al_strategy.notion=$notion" "seed=$seed"
    done
done
