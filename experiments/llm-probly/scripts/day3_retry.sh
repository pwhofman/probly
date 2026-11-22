#!/usr/bin/env bash
# Week 3, day 3 retry: the three MMLU runs that ran out of GPU memory, now on cpu.
#
# Run from experiments/llm-probly, with the Mac kept awake:
#   caffeinate -i bash scripts/day3_retry.sh
#
# Every step is logged to logs/day3_queue.log (start, end, minutes, ok/FAILED).
# A failed step does not stop the queue. Every step can be re-run: generation
# resumes from the .partial.jsonl and existing method results are skipped, so
# after a crash just start the script again.
# Order: smoke test of the new model family first (fails fast while you watch),
# then what other members wait for (7B no empty cells, 3B semantic), then the
# long caches.

set -u

# transformers tries to open a safetensors conversion PR on the Hub for models that only
# ship .bin weights (deberta-base-mnli) and prints a KeyError from a background thread.
export DISABLE_SAFETENSORS_CONVERSION=1
export PYTHONUNBUFFERED=1  # show progress lines right away, not at the end of a step

cd "$(dirname "$0")/.."
mkdir -p logs
LOG=logs/day3_retry.log
FAILED=0
exec > >(tee -a logs/day3_retry_full.log) 2>&1  # everything also goes to logs/day3_retry_full.log (git ignores *.log)

stop() {
    local msg
    msg="$(date '+%F %T') STOPPED by Ctrl+C, run the script again to continue"
    echo "$msg" >>"$LOG"  # no tee here: Ctrl+C also stops tee
    echo "$msg" >/dev/tty 2>/dev/null || true
    exit 130
}
trap stop INT TERM  # one Ctrl+C stops the whole queue, not only the current step

step() {
    local name=$1
    shift
    local start rc
    start=$(date +%s)
    echo "$(date '+%F %T') START $name: $*" | tee -a "$LOG"
    "$@"
    rc=$?
    # 130 = the step was stopped with Ctrl+C (Python catches it, so bash may not see the signal)
    [ "$rc" -eq 130 ] && stop
    if [ "$rc" -eq 0 ]; then status=ok; else status="FAILED (exit $rc)"; FAILED=$((FAILED + 1)); fi
    echo "$(date '+%F %T') END   $name: $status ($(( ($(date +%s) - start) / 60 )) min)" | tee -a "$LOG"
}

run() { uv run python -m cli "$@"; }

# The unfinished mps files must not be continued on cpu (generate.py resumes by settings,
# not by device), so they are moved aside, not deleted.
set_aside() {
    local base=data/$1/mmlu_200
    [ -f "$base.jsonl" ] && return 0  # finished cache: nothing to set aside
    for kind in partial.jsonl meta.json; do
        if [ -f "$base.$kind" ]; then
            mv "$base.$kind" "$base.mps-oom.$kind" && echo "moved $base.$kind aside"
        fi
    done
    return 0
}

for model in smollm2-1.7b qwen3b qwen7b; do set_aside "$model"; done

step "SmolLM2 MMLU 200 (cpu)" run run --config configs/smollm2-1.7b_mmlu_200.yaml
step "3B MMLU 200 (cpu)" run run --config configs/qwen3b_mmlu_200.yaml
step "7B MMLU 200 (cpu, bfloat16)" run run --config configs/qwen7b_mmlu_200.yaml

echo "$(date '+%F %T') RETRY DONE, $FAILED failed step(s)" | tee -a "$LOG"
