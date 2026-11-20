# Score contract (B2)

Every method saves one file: `results/qwen0.5b/<method>.json`

1. Direction: higher = more uncertain. For reflexive methods, uncertainty = 1 - confidence.
2. Raw values: store unnormalized scores. No rescaling.
3. No missing values: if parsing fails, uncertainty = 1.0 and failed = true.
4. Same answer: all methods score the main (greedy) answer.
5. Same question ids: `ids` in the same order as `id` in `data/qwen0.5b/triviaqa_50.jsonl`.

Example:

    {"method": "p_true_zeroshot_norm", "model": "Qwen/Qwen2.5-0.5B-Instruct",
     "answer": "greedy", "ids": [0, 1, 2], "uncertainty": [0.12, 0.87, 0.40],
     "failed": [false, false, true], "seconds_per_question": 0.8}


## Pipeline (PIPE, week 2)

### Folders
    data/<model>/<dataset>_<n>.jsonl         cache: one row per question (DATA)
    data/<model>/<dataset>_<n>.meta.json     generation settings
    results/<model>/<dataset>_<n>/<method>.json  one score file per method
    configs/<model>_<dataset>.yaml           one config per run
    methods/<method>.py                      one plugin per method

`<model>` is the short name from `scripts/generate.py` (`qwen0.5b`, `qwen1.5b`, `qwen3b`, `qwen7b`).

### Method plugin
Every file in `methods/` has only a name and a score function:

```python
NAME = "likelihood_mean"  # same as the file name and the result file name


def score(rows, model=None, tok=None) -> dict:
    """Return one score record (rules 1-5 above) for these rows."""
```

and one entry in `common/registry.py` (week 3, E2) with what the selector needs to know:

```python
MethodSpec("likelihood_mean", "baseline", "logprobs", GREEDY, note="mean greedy token log-prob")
#           name              family      access      cost (generation passes per question)
```

- `family`: `agreement` | `self_eval` | `baseline` | `hybrid`
- `access`: `text_only` (generated text only, works behind any API) | `logprobs` (token
  log-probabilities, as APIs with a logprobs option give them) | `white_box` (full next-token
  distribution or weights, local model only)
- `cost`: LLM generation passes per question, greedy answer included: 1 = greedy only,
  11 = greedy + the 10 cached samples
- `needs_model=True`: the runner loads the LLM and passes `model, tok`
- `uses_nli=True`: the method also runs the NLI model
- `probability=True`: `1 - uncertainty` is a probability of being correct, so ECE is computed

`tests/test_registry.py` fails if a plugin has no entry or an entry has no plugin.
`uv run python -m cli methods` prints the registry.

Rules:
- `score()` does **not** save and does **not** load data. The runner gives it the rows and saves what it returns.
- `ids` in the record are `[row["id"] for row in rows]`, in the same order.
- No hardcoded model or dataset names inside a plugin.
- Put `seconds_per_question` in the record. Use 0.0 if the method only reads the cache.
- If your score is a probability of being correct (uncertainty = 1 - probability), set `probability=True` in its registry entry, so ECE is computed for it.

### Runner
    uv run python -m cli run --config configs/qwen0.5b_triviaqa_200.yaml
    # steps: generate -> label -> methods -> evaluate -> table
    # a step is skipped if its output file already exists
    # --methods a,b            run only these methods
    # --skip-model-methods     skip methods with needs_model=True in the registry
    # --force                  recompute even if the file exists
