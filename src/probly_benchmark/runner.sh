#!/usr/bin/env bash
# CIFAR-10 active learning lanes for the camera-ready benchmark paper.
#
# A lane is a group of methods x the camera-ready strategies x a seed range, run
# sequentially on one GPU through scripts/run_al_sweep.py, the single source of truth
# for the grid: every machine launches identical commands. Runs log to wandb
# probly/crv-al-cifar10 and the sweep skips combos already finished there, so a crashed
# lane is restarted with the same command, and wave 2 is the same command with
# --seeds 3-4.
#
# Usage
#   ./src/probly_benchmark/runner.sh --list
#   ./src/probly_benchmark/runner.sh --lane NAME [--seeds SPEC] [--strategy NAME]...
#                                    [--device DEV] [--wandb-project NAME] [--limit N]
#   DRY_RUN=1 ./src/probly_benchmark/runner.sh --lane crebnn --seeds 0   # print, launch nothing
#   SKIP_REFRESH=1 ...                                                    # reuse the local wandb cache
# Defaults: --seeds 0-2, project crv-al-cifar10, device auto-selected.
#
# Lanes (estimated A40 hours per seed, all strategies):
#   singles-a  base evidential_classification vbll duq ddu het_net efficient_credal_prediction sngp  (~32)
#   singles-b  posterior_network laplace dropout dropconnect bayesian                               (~30)
#   ensemble   ensemble                                                                              (~32)
#   heads      batchensemble subensemble dare                                                        (~32)
#   crewra     credal_wrapper                                                                        (~34)
#   creens     credal_ensembling                                                                     (~34)
#   crerl      credal_relative_likelihood                                                            (~42)
#   crebnn     credal_bnn                                                                            (~100, ~25 h per run)
#
# Colleague setup (a Linux machine with a CUDA GPU):
#   git clone <repo> && cd probly && git checkout rebuttal && git pull   # the commit Paul names
#   uv sync
#   uv run wandb login            # an account with write access to the wandb entity "probly"
#   tmux new -s al                # an SSH disconnect would otherwise kill the lane
#   ./src/probly_benchmark/runner.sh --lane crebnn --seeds 0 --device cuda:0
# Split a large lane across GPUs by seed (--seeds 0, 1, 2 on three GPUs) or by strategy.
# Report back when the final "Done. N succeeded, M failed." line prints; restart a failed
# lane with the same command. Trial runs never go through this script: call
# `uv run python -m probly_benchmark.active_learning ... wandb.enabled=false` directly.
#
# Preflight before anything launches: the wandb login, the resolved config (200 epochs,
# 9 query rounds, clipping 1.0, drop_last, the project), the commit when run from a git
# checkout, one CIFAR-10 download, and a wandb cache refresh for the skip logic. Per-lane
# output goes to logs/al_cifar10_<lane>_seeds<spec>_<timestamp>.log.
set -euo pipefail
# Server deployments are plain copies, not git checkouts: resolve the root from this file.
cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."

LANE_ORDER="singles-a singles-b ensemble heads crewra creens crerl crebnn"
lane_methods() {
    case "$1" in
        singles-a) echo "base evidential_classification vbll duq ddu het_net efficient_credal_prediction sngp" ;;
        singles-b) echo "posterior_network laplace dropout dropconnect bayesian" ;;
        ensemble) echo "ensemble" ;;
        heads) echo "batchensemble subensemble dare" ;;
        crewra) echo "credal_wrapper" ;;
        creens) echo "credal_ensembling" ;;
        crerl) echo "credal_relative_likelihood" ;;
        crebnn) echo "credal_bnn" ;;
        *) return 1 ;;
    esac
}

usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; }
die() { echo "runner.sh: $*" >&2; exit 1; }

LANE=""
SEEDS="0-2"
PROJECT="crv-al-cifar10"
DEVICE=""
LIMIT=""
STRATEGIES=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --list)
            for l in $LANE_ORDER; do printf '%-10s %s\n' "$l" "$(lane_methods "$l")"; done
            exit 0
            ;;
        --lane) LANE="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --strategy) STRATEGIES+=("$2"); shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --wandb-project) PROJECT="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1 (see --help)" ;;
    esac
done
[[ -n "$LANE" ]] || die "--lane NAME is required (see --list)"
METHOD_LIST=$(lane_methods "$LANE") || die "unknown lane '$LANE' (see --list)"
[[ -f scripts/run_al_sweep.py ]] || die "scripts/run_al_sweep.py not found next to this deployment -- sync it first."
read -r -a METHODS <<<"$METHOD_LIST"
DRY="${DRY_RUN:-0}"

echo "### lane $LANE: $METHOD_LIST | seeds $SEEDS | project $PROJECT | device ${DEVICE:-auto} ###"

# 1. wandb login by a member of the entity the runs log to.
uv run python -c "
import wandb
viewer = wandb.Api().viewer
teams = list(viewer.teams)
print('wandb user:', viewer.entity, '| teams:', teams)
assert 'probly' in teams or viewer.entity == 'probly', 'this wandb account is not a member of entity probly'
" || die "wandb preflight failed -- run 'uv run wandb login' with an account that belongs to entity probly."

# 2. The resolved config of the lane's first method must carry the camera-ready protocol.
cfg=$(uv run python -m probly_benchmark.active_learning --cfg job "method=${METHODS[0]}" al_dataset=cifar10 "wandb.project=$PROJECT") \
    || die "config composition failed"
echo "$cfg"
for want in "epochs: 200" "n_iterations: 9" "grad_clip_norm: 1.0" "drop_last: true" "project: $PROJECT"; do
    grep -qE "^[[:space:]]*${want}\$" <<<"$cfg" || die "resolved config lacks '$want' -- wrong checkout or stale configs."
done

# 3. Informational only; the server copy is not a git checkout and paths never depend on it.
echo "commit: $(git rev-parse --short HEAD 2>/dev/null || echo '(not a git checkout)')"

if [[ "$DRY" != "1" ]]; then
    # 4. One download before the lane starts; concurrent first-time downloads collide.
    uv run python -c "
from probly_benchmark.data import get_data_al
get_data_al('cifar10', seed=0)
print('CIFAR-10 present.')
"
fi

# 5. Refresh the wandb cache the sweep's skip logic reads. A project that does not exist
#    yet (first launch of a campaign) has nothing finished, so the refresh is skipped; any
#    other refresh failure aborts, because a stale cache would relaunch finished runs.
if [[ "${SKIP_REFRESH:-0}" != "1" ]]; then
    if uv run python -c "import wandb; len(wandb.Api().runs('probly/$PROJECT', per_page=1))" >/dev/null 2>&1; then
        uv run python scripts/inspect_al_runs.py --refresh --source "probly/$PROJECT" || die "wandb cache refresh failed"
    else
        echo "wandb project probly/$PROJECT not found yet: nothing finished, skipping the cache refresh."
    fi
fi

# 6. The sweep: identical commands on every machine.
cmd=(uv run python scripts/run_al_sweep.py --dataset cifar10 --seeds "$SEEDS" --wandb-project "$PROJECT")
for m in "${METHODS[@]}"; do cmd+=(--method "$m"); done
for s in ${STRATEGIES[@]+"${STRATEGIES[@]}"}; do cmd+=(--strategy "$s"); done
[[ -z "$DEVICE" ]] || cmd+=(--device "$DEVICE")
[[ -z "$LIMIT" ]] || cmd+=(--limit "$LIMIT")
if [[ "$DRY" == "1" ]]; then
    "${cmd[@]}" --show-commands
    exit 0
fi
mkdir -p logs
log="logs/al_cifar10_${LANE}_seeds${SEEDS//,/_}_$(date +%Y%m%d_%H%M%S).log"
echo "log: $log"
"${cmd[@]}" --execute 2>&1 | tee "$log"
exit "${PIPESTATUS[0]}"
