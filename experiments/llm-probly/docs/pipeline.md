# Pipeline & evaluation

*Owner: PIPE. Describes the code in `cli.py`, `common/registry.py`, `scripts/run_benchmark.py`, `scripts/compare_models.py`,
`scripts/plot_results.py`, `common/evaluate.py` and `common/summary.py`.*

## One command per run

    uv run python -m cli run --config configs/qwen1.5b_triviaqa_200.yaml

`cli.py` is the one entry point for everything (`uv run python -m cli --help`):

| Command | What it does |
|---|---|
| `run --config <yaml>` | the five steps below for one config |
| `methods [--access text_only] [--max-cost 11] [--json]` | the method registry: family, access, cost |
| `generate`, `label` | one data step on its own (same options as `scripts/generate.py`, `scripts/label.py`) |
| `compare`, `plot` | methods x models table and plots (`scripts/compare_models.py`, `scripts/plot_results.py`) |
| `check-prefix --model m --small 200 --large 1000` | the first rows of a larger cache are the smaller cache |

`python -m scripts.<name>` still works the same.

The runner does five steps. Each step is skipped if its output already exists, so an
interrupted run continues where it stopped.

| Step | What happens | Output |
|---|---|---|
| 1. generate | answers + 10 samples per question (`scripts/generate.py`) | `data/<model>/<dataset>_<n>.jsonl` + `.meta.json` |
| 2. label | right/wrong per answer (`scripts/label.py`) | adds `correct` to the cache |
| 3. methods | every method in `common/registry.py` computes an uncertainty per question | `results/<model>/<dataset>_<n>/<method>.json` |
| 4. evaluate | metrics per method (below) | |
| 5. table | one row per method, with family, access and cost from the registry | `results/<model>/<dataset>_<n>/summary.md` |

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

**Memory.** Methods with `needs_model=True` load the model once per run. In float32,
1.5B needs about 6–7 GB of RAM; on an 8 GB laptop it swaps and becomes unusably slow,
so 1.5B and larger run on a machine with more memory.

## Adding a method

One file in `methods/` and one line in `common/registry.py`:

```python
# methods/my_method.py
NAME = "my_method"  # also the file name and the result file name


def score(rows, model=None, tok=None) -> dict: ...  # return a score record (CONTRACT.md); do not save or load anything


# common/registry.py, in _SPECS
MethodSpec("my_method", "self_eval", "logprobs", GREEDY + 1, needs_model=True, probability=True, note="...")
```

The fields are explained in CONTRACT.md. `probability=True` means uncertainty = 1 − probability
of being correct, so ECE is computed. Template: `methods/dummy_random.py`.

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

## The report (E4)

    uv run python -m cli report            # writes results/report.md and results/report.html

One command, no model needed: it reads every `results/<model>/<dataset>_<n>/` that has a data cache
(runs under 100 questions are skipped) and writes

1. **Runs**: model, dataset, questions, accuracy, precision and device, and two picks per run:
   - **pick**: the cheapest method that is best or *tied* with the best (paired bootstrap below);
   - **pick via API**: the same among methods that need only text or log-probabilities.
   The random reference is never a pick. If it is itself tied with the best method, the run says
   "none (no method beats random)", as for 0.5B and 1.5B on MMLU.
2. **Methods x models** on TriviaQA 200: AUROC per model next to family, access and cost.
3. **One table per run**: AUROC with its interval, best / tied / worse with the paired difference,
   AURC, PRR, ECE, method seconds per question and the registry cost.

The HTML page is self-contained (one file, no scripts) and marks the best row, the tied rows and
the pick. `--n-boot 200` gives a quick draft; the default 1000 takes about two minutes.

## Is method X better than method Y? (paired bootstrap, E4)

Two separate 95 % AUROC intervals often overlap even when one method is better on
almost every resample, because both methods find the same questions hard. The
paired bootstrap resamples the questions once per round, scores *both* methods on
that resample and keeps the difference. Its interval is much narrower.

Example, qwen0.5b on TriviaQA 1000: likelihood_mean 0.712 [0.672, 0.747] and
mean_token_entropy 0.696 [0.656, 0.735] overlap almost completely, yet the paired
difference is +0.016 [+0.003, +0.028]: likelihood_mean is better, by a little.

    uv run python -m cli diff --run qwen0.5b/triviaqa_1000 likelihood_mean mean_token_entropy
    # On qwen0.5b/triviaqa_1000, likelihood_mean is better than mean_token_entropy: AUROC 0.712 vs 0.696, ...

`--metric prr` or `--metric aurc` compare another metric. In code:
`common.evaluate.paired_bootstrap(u_a, u_b, correct)` returns the difference, its
interval and `.verdict` (`"A"`, `"B"` or `"tie"`).

Every `summary.md` compares each method with the best one of the run:

| column | meaning |
|---|---|
| `vs_best` | `best`; `tied` = the interval of the difference contains 0, the data cannot tell it apart from the best; `worse` |
| `d_auroc` | AUROC minus the best AUROC |
| `d_ci_low`, `d_ci_high` | paired 95 % interval of `d_auroc` |

"Tied" is the same test the selector (D1) uses before it picks the cheapest of the
tied methods.

## Comparing models and plots

After the runner has run for each model:

    uv run python -m scripts.compare_models --dataset triviaqa --n 200
    uv run python -m scripts.plot_results --dataset triviaqa --n 200 --curves-model qwen1.5b

- `results/comparison_<dataset>_<n>.md`: one table per metric, methods × model sizes
- `results/plots/<dataset>_<n>_auroc_vs_size.png`, `..._prr_vs_size.png`: one line per method
- `results/plots/<dataset>_<n>_<model>_rejection.png`: accuracy of the kept answers vs share rejected,
  with the perfect (oracle) and random reference

Each method keeps the same colour and marker in every plot.
