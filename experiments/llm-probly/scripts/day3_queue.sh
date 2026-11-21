#!/usr/bin/env bash
# Week 3, day 3 long jobs (E3 + E5), one after another on one machine (24 GB Mac).
#
# Run from experiments/llm-probly, with the Mac kept awake:
#   caffeinate -i bash scripts/day3_queue.sh
#
# Every step is logged to logs/day3_queue.log (start, end, minutes, ok/FAILED).
# A failed step does not stop the queue. Every step can be re-run: generation
# resumes from the .partial.jsonl and existing method results are skipped, so
# after a crash just start the script again.
# Order: smoke test of the new model family first (fails fast while you watch),
# then what other members wait for (7B no empty cells, 3B semantic), then the
# long caches.

set -u -o pipefail  # pipefail: a failing command fails its "| tee" pipeline too
cd "$(dirname "$0")/.."
mkdir -p logs
LOG=logs/day3_queue.log
FAILED=0

step() {
    local name=$1
    shift
    local start
    start=$(date +%s)
    echo "$(date '+%F %T') START $name: $*" | tee -a "$LOG"
    if "$@" 2>&1 | tee -a "$LOG.full"; then
        status=ok
    else
        status=FAILED
        FAILED=$((FAILED + 1))
    fi
    echo "$(date '+%F %T') END   $name: $status ($(( ($(date +%s) - start) / 60 )) min)" | tee -a "$LOG"
}

run() { uv run python -m cli "$@"; }

# 1. second model family: does SmolLM2 work end to end? (download ~3.4 GB + 1 min)
step "smollm2 smoke test" run run --config configs/smollm2-1.7b_triviaqa_5.yaml

# 2. E5 with C: self-evaluation on 7B, plus SE / discrete SE on 7B (about 50 min, downloads 7B if needed)
step "7B p(True), verbalized, SE, DSE" run run --config configs/qwen7b_triviaqa_200.yaml \
    --methods p_true_zeroshot,verbalized_2s,p_true_brainstorm,p_true_empirical,semantic_entropy,discrete_semantic_entropy

# 3. E5 with A: SE / discrete SE on 3B (cache + NLI only, about 10 min)
step "3B SE, DSE" run run --config configs/qwen3b_triviaqa_200.yaml \
    --methods semantic_entropy,discrete_semantic_entropy

# 4. 1.5B: the random reference was the only empty cell (seconds)
step "1.5B dummy_random" run run --config configs/qwen1.5b_triviaqa_200.yaml --methods dummy_random

# 5. E3: TriviaQA 1000 for 1.5B and 3B (about 66 + 86 min), each followed by its prefix check
step "1.5B TriviaQA 1000" run run --config configs/qwen1.5b_triviaqa_1000.yaml
step "1.5B prefix check" run check-prefix --model qwen1.5b --small 200 --large 1000
step "3B TriviaQA 1000" run run --config configs/qwen3b_triviaqa_1000.yaml
step "3B prefix check" run check-prefix --model qwen3b --small 200 --large 1000

# 6. E3: MMLU 200 for every model (about 15 + 25 + 30 + 55 min)
step "0.5B MMLU 200" run run --config configs/qwen0.5b_mmlu_200.yaml
step "0.5B MMLU prefix check" run check-prefix --model qwen0.5b --dataset mmlu --small 100 --large 200
step "1.5B MMLU 200" run run --config configs/qwen1.5b_mmlu_200.yaml
step "3B MMLU 200" run run --config configs/qwen3b_mmlu_200.yaml
step "7B MMLU 200" run run --config configs/qwen7b_mmlu_200.yaml

# 7. E3: second model family on the same questions (about 15 + 25 min)
step "SmolLM2 TriviaQA 200" run run --config configs/smollm2-1.7b_triviaqa_200.yaml
step "SmolLM2 MMLU 200" run run --config configs/smollm2-1.7b_mmlu_200.yaml

echo "$(date '+%F %T') QUEUE DONE, $FAILED failed step(s)" | tee -a "$LOG"
