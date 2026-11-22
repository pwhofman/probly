# Week-1 scripts (archived)

These scripts produced the week-1 and early week-2 results. They are superseded by the
pipeline (`scripts/run_benchmark.py`, `methods/`, `common/`) and are kept only so the
week-1 numbers stay reproducible. They are not linted, not tested and not maintained.

| Script | What it did | Replaced by |
|---|---|---|
| `empirical_v1.py`, `verbalized_v1.py` | First versions of empirical p(True) and verbalized 2S | `scripts/r3_methods.py` → `methods/p_true_empirical.py`, `methods/verbalized_2s.py` (C2 moves them into the library) |
| `r3_real_data.py` | Week-1 R3 run on 50 TriviaQA questions | `scripts/run_benchmark.py` with a config |
| `sanity_r3.py`, `sanity_table.py` | Week-1 sanity test and its table | `common/sanity.py`, `common/summary.sanity_table` |
| `r3_cost_table.py` | Week-1 cost per question for the reflexive methods | cost column in the registry (E2) and report (E4) |
| `model_list.py` | Model list for R3 (`results/model_list.txt`) | `configs/` |
| `predownload.py` | Download model weights before a run | `scripts/generate.py` downloads on first use |
| `diagnose_mps.py` | One-off: MPS vs CPU differences vs week 1 | finding documented in `docs/data_and_labelling.md` |
| `check_self_plugins.py` | One-off: SELF plugins reproduce the week-1 score files | equivalence tests (C1) |

The R3 notebook (`notebooks/R3_text_only.ipynb`) still points here. To re-run one of them,
run it from `experiments/llm-probly` with
`PYTHONPATH=.:scripts uv run python archive/week1/scripts/<name>.py` (they import `common/`,
`methods/` and `scripts/r3_methods.py`).
