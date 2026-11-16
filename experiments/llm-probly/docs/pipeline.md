# Pipeline & evaluation

*Owner: PIPE. Describes the code in `scripts/run_benchmark.py`, `scripts/compare_models.py`,
`scripts/plot_results.py`, `common/evaluate.py` and `common/summary.py`.*

## One command per run

    uv run python -m scripts.run_benchmark --config configs/qwen1.5b_triviaqa_200.yaml

The runner does five steps. Each step is skipped if its output already exists, so an
interrupted run continues where it stopped.

| Step | What happens | Output |
|---|---|---|
| 1. generate | answers + 10 samples per question (`scripts/generate.py`) | `data/<model>/<dataset>_<n>.jsonl` + `.meta.json` |
| 2. label | right/wrong per answer (`scripts/label.py`) | adds `correct` to the cache |
| 3. methods | every plugin in `methods/` computes an uncertainty per question | `results/<model>/<dataset>_<n>/<method>.json` |
| 4. evaluate | metrics per method (below) | |
| 5. table | one row per method | `results/<model>/<dataset>_<n>/summary.md` |

Options:

    --methods p_true_zeroshot,likelihood_mean   run only these methods
    --skip-model-methods                        skip methods that need the LLM
    --force                                     recompute results that already exist

## Configs

One YAML file per run in `configs/`:

```yaml
model: qwen1.5b       # short name from scripts/generate.py
dataset: triviaqa     # any dataset in common/data.py (triviaqa, mmlu)
n: 200                # number of questions
precision: float32    # float32 | float16 | bfloat16
device: cpu           # cpu | mps | cuda, only used when a method loads the model
methods: all          # "all" or a list of method names
```

**Precision must match the cache.** The runner reads the precision from the cache's
`.meta.json` and stops if the config asks for another one. Precision changes the answers
(bfloat16 → float32 changed 9 of 50 answers on 0.5B), so a table must never mix them
silently.

**Memory.** Methods with `NEEDS_MODEL = True` load the model once per run. In float32,
1.5B needs about 6–7 GB of RAM; on an 8 GB laptop it swaps and becomes unusably slow,
so 1.5B and larger run on a machine with more memory.

## Adding a method

One file in `methods/`, nothing else to change:

```python
NAME = "my_method"         # also the result file name
FAMILY = "self_eval"       # "agreement" | "self_eval" | "baseline" | "hybrid"
NEEDS_MODEL = True         # the runner loads the LLM and passes model, tok

def score(rows, model=None, tok=None) -> dict:
    ...  # return a score record (CONTRACT.md); do not save or load anything
```

If the score is a probability of being correct (uncertainty = 1 − probability), add
`"probability": True` to the record so ECE is computed. Template: `methods/dummy_random.py`.

## Metrics

| Metric | Question it answers | Range |
|---|---|---|
| **AUROC** | Does the method give wrong answers a higher uncertainty than right ones? | 0.5 = random, 1 = perfect |
| **AURC** | Average error rate when the most uncertain answers are rejected | lower is better; depends on the model's accuracy |
| **PRR** | Share of the perfect method's improvement when rejecting the most uncertain answers | 0 = random, 1 = perfect; fair across model sizes |
| **ECE** | Does a stated confidence match the real accuracy? (probability methods only) | 0 = perfectly calibrated |

AUROC comes with a 95 % bootstrap interval. Ties in the uncertainty are broken randomly
and averaged over 20 orders (AURC, PRR, rejection curves). PRR and ECE are checked against
lm-polygraph's implementations in `tests/test_evaluate.py`.

Why PRR next to AURC: a bigger model is right more often, which alone improves AURC.
PRR divides that out, so it shows whether the *ranking* got better.

## Comparing models and plots

After the runner has run for each model:

    uv run python -m scripts.compare_models --dataset triviaqa --n 200
    uv run python -m scripts.plot_results --dataset triviaqa --n 200 --curves-model qwen1.5b

- `results/comparison_<dataset>_<n>.md`: one table per metric, methods × model sizes
- `results/plots/<dataset>_<n>_auroc_vs_size.png`, `..._prr_vs_size.png`: one line per method
- `results/plots/<dataset>_<n>_<model>_rejection.png`: accuracy of the kept answers vs share rejected,
  with the perfect (oracle) and random reference

Each method keeps the same colour and marker in every plot.
