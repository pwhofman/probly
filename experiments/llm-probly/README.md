# LLM-probly experiments

Base: pwhofman/probly @ 1189b3a

## Setup
    git clone git@github.com:elifkaracaa/probly.git
    cd probly
    uv sync
    cd experiments/llm-probly

All commands below are run from `experiments/llm-probly`.

## Shared data (B1)
    from common.data import load_data
    rows = load_data("qwen0.5b", 50)   # or 5

Fields per row: `id, question, aliases` (canonical answer first), `greedy_answer,
greedy_ll_sum, greedy_ll_mean, samples` (10), `samples_ll_mean, model,
correct_alias, correct, samples_correct`.

Re-create the data (Qwen2.5-0.5B-Instruct, temperature 0.7, max 32 new tokens,
stop at the first newline, seed 42):

    uv run python -m scripts.generate --model qwen0.5b --n 50
    uv run python -m scripts.label --model qwen0.5b --n 50

## Tests
    uv run python -m pytest tests

## Structure
- common/   shared code (data.py + correctness.py: B1, evaluate.py + io.py: B2, reflexive.py: R1+R2)
- scripts/  generate.py, label.py (B1)
- data/qwen0.5b/      triviaqa_5.jsonl, triviaqa_50.jsonl
- results/qwen0.5b/   <method>.json
- results/sanity/     <method>_<model>.json
- notebooks/          one notebook per person + 00_summary.ipynb
- tests/              unit tests for common/

## Rules
- git pull at the start of the day, git push at the end
- only edit your own notebook and results files
- changes to common/ via pull request
