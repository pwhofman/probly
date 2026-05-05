#!/usr/bin/env bash
# run_dcic_training.sh
# Trains all DCIC first-order datasets for every credal method.
#
# Usage:
#   bash run_dcic_training.sh                  # run everything
#   bash run_dcic_training.sh micebone         # single dataset
#   bash run_dcic_training.sh micebone credal_relative_likelihood  # single combo
#
# Exit behaviour: a failed run is logged and the script continues so that
# one broken dataset/method doesn't block the rest.

set -euo pipefail

# ── configurable ─────────────────────────────────────────────────────────────
PYTHON_VERSION="${PYTHON_VERSION:-3.13}"
LOG_DIR="${LOG_DIR:-logs/dcic}"
# ─────────────────────────────────────────────────────────────────────────────

DATASETS=(
    benthic
    cifar10h
    micebone
    pig
    plankton
    qualitymri
    synthetic
    treeversity1
    treeversity6
    turkey
)

METHODS=(
    credal_relative_likelihood
    credal_wrapper
    credal_ensembling
    credal_bnn
    credal_net
    efficient_credal
)

# filter to command-line args if provided
if [[ $# -ge 1 ]]; then
    DATASETS=("$1")
fi
if [[ $# -ge 2 ]]; then
    METHODS=("$2")
fi

mkdir -p "${LOG_DIR}"

FAILED=()
TOTAL=0
SUCCEEDED=0

for DATASET in "${DATASETS[@]}"; do
    RECIPE="resnet50_${DATASET}"
    for METHOD in "${METHODS[@]}"; do
        TOTAL=$(( TOTAL + 1 ))
        LOG_FILE="${LOG_DIR}/${METHOD}_${DATASET}.log"

        echo "════════════════════════════════════════════════════════════"
        echo "  method  : ${METHOD}"
        echo "  dataset : ${DATASET}  (recipe=${RECIPE})"
        echo "  log     : ${LOG_FILE}"
        echo "════════════════════════════════════════════════════════════"

        if uv run --python "${PYTHON_VERSION}" train.py \
                method="${METHOD}" \
                recipe="${RECIPE}" \
                2>&1 | tee "${LOG_FILE}"; then
            SUCCEEDED=$(( SUCCEEDED + 1 ))
            echo "  ✓ done"
        else
            echo "  ✗ FAILED — see ${LOG_FILE}"
            FAILED+=("${METHOD}/${DATASET}")
        fi

        echo ""
    done
done

# ── summary ──────────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════"
echo "  Summary: ${SUCCEEDED}/${TOTAL} runs succeeded"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "  Failed runs:"
    for f in "${FAILED[@]}"; do
        echo "    - ${f}"
    done
    exit 1
fi
echo "  All runs completed successfully."
