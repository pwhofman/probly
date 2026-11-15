# LLM-probly experiments

Base: pwhofman/probly @ 1189b3a

## Setup
    git clone git@github.com:elifkaracaa/probly.git
    cd probly
    uv sync
    cd experiments/llm-probly

All commands below are run from `experiments/llm-probly`.

## Shared data (DATA)
    from common.data import load_data
    rows = load_data("qwen1.5b", 200)   # models: qwen0.5b, qwen1.5b, qwen3b; n: 5 or 200

One cache per model, shared by every method (cache v2). Fields per row:
`id, dataset, question, aliases` (canonical answer first), `greedy_answer, greedy_ll_sum,
greedy_ll_mean, greedy_token_logprobs, greedy_token_entropies, samples` (10),
`samples_ll_mean, samples_ll_sum, model, correct_alias, correct, samples_correct`.
Settings and library versions of each run are in `data/<model>/<dataset>_<n>.meta.json`.

Re-create a cache (TriviaQA, temperature 0.7, max 32 new tokens, stop at the first
newline, seed 42; saves after every question and resumes):

    uv run python -m scripts.generate --model qwen1.5b --n 200
    uv run python -m scripts.label    --model qwen1.5b --n 200

Details, accuracy per model and the label error rate: [docs/data_and_labelling.md](docs/data_and_labelling.md).

## Running the benchmark (PIPE)
One command runs the whole chain (generate → label → methods → evaluate → table) for one
model and dataset, skipping steps that are already done:

    uv run python -m scripts.run_benchmark --config configs/qwen1.5b_triviaqa_200.yaml

Compare model sizes and make the plots after running each model:

    uv run python -m scripts.compare_models --dataset triviaqa --n 200
    uv run python -m scripts.plot_results --dataset triviaqa --n 200

A new uncertainty method is one file in `methods/` (see `CONTRACT.md`). Configs, metrics
(AUROC, AURC, PRR, ECE) and plots: [docs/pipeline.md](docs/pipeline.md).

## Tests
    uv run python -m pytest tests

## Structure
- common/     shared code (data.py, generation.py, correctness.py: DATA; evaluate.py, io.py, summary.py: PIPE; reflexive.py: SELF)
- scripts/    generate.py, label.py, manual_check.py (DATA); run_benchmark.py, compare_models.py, plot_results.py (PIPE); method scripts
- methods/    one plugin per uncertainty method (see CONTRACT.md)
- configs/    one YAML file per benchmark run
- data/<model>/      <dataset>_<n>.jsonl, .meta.json, .manual_check_part<p>.json
- results/<model>/<dataset>_<n>/   <method>.json + summary.md (benchmark runs)
- results/<model>/   <method>.json (week 1; score contract, see CONTRACT.md)
- results/comparison_<dataset>_<n>.md, results/plots/   model comparison (PIPE)
- results/sanity/    <method>_<model>.json
- notebooks/  one notebook per person
- docs/       README sections per part
- tests/      unit tests

## Rules
- git pull at the start of the day, git push at the end
- work on a branch and open a PR into **our** main:
  github.com/elifkaracaa/probly/compare/main...BRANCH
  (the link git prints after a push targets pwhofman/probly; don't use it)
- every PR is reviewed by your second before it is merged
