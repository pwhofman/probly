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
Every file in `methods/` has:

```python
NAME = "likelihood_mean"   # same as the result file name
FAMILY = "baseline"        # "agreement" | "self_eval" | "baseline" | "hybrid"
NEEDS_MODEL = False        # True: the runner loads the LLM and passes model, tok

def score(rows, model=None, tok=None) -> dict:
    """Return one score record (rules 1-5 above) for these rows."""
```

Rules:
- `score()` does **not** save and does **not** load data. The runner gives it the rows and saves what it returns.
- `ids` in the record are `[row["id"] for row in rows]`, in the same order.
- No hardcoded model or dataset names inside a plugin.
- Put `seconds_per_question` in the record. Use 0.0 if the method only reads the cache.
- If your score is a probability of being correct (uncertainty = 1 - probability), add `"probability": True` to the record, so ECE is computed for it.

### Runner
    uv run python -m scripts.run_benchmark --config configs/qwen0.5b_triviaqa.yaml
    # steps: generate -> label -> methods -> evaluate -> table
    # a step is skipped if its output file already exists
    # --methods a,b            run only these methods
    # --skip-model-methods     skip plugins with NEEDS_MODEL = True
    # --force                  recompute even if the file exists
