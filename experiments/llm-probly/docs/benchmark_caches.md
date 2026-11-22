# Benchmark caches and accuracy per model

*Owner: PIPE (E3). One row per cached generation file in `data/<model>/`. Updated week 3, day 3.*

Every cache holds, per question, the greedy answer, 10 samples (temperature 0.7) and their token
log-probabilities, plus the correctness label of the greedy answer. All methods read these files, so the
numbers below are shared by every method.

| model | dataset | questions | accuracy | precision, device | s / question | likelihood_mean AUROC [95 %] |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | TriviaQA | 200 | 0.190 | float32, cpu | 2.4 | 0.791 [0.70, 0.88] |
| Qwen2.5-0.5B | TriviaQA | **1000** | 0.207 | float32, cpu | 2.8 | 0.712 [0.67, 0.75] |
| Qwen2.5-0.5B | MMLU | 200 | 0.195 | float32, cpu | 4.9 | 0.543 [0.44, 0.65] |
| Qwen2.5-1.5B | TriviaQA | 200 | 0.375 | float32, cpu | 4.0 | 0.790 [0.72, 0.85] |
| Qwen2.5-1.5B | TriviaQA | **1000** | 0.370 | float32, cpu | 4.8 | 0.789 [0.76, 0.82] |
| Qwen2.5-1.5B | MMLU | 200 | 0.360 | float32, cpu | 14.3 | 0.505 [0.42, 0.59] |
| Qwen2.5-3B | TriviaQA | 200 | 0.485 | float32, mps | 5.1 | 0.786 [0.72, 0.84] |
| Qwen2.5-3B | TriviaQA | **1000** | 0.445 | float32, mps | 5.5 | 0.798 [0.77, 0.82] |
| Qwen2.5-3B | MMLU | 200 | 0.635 | float32, cpu | 29.1 | 0.768 [0.70, 0.83] |
| Qwen2.5-7B | TriviaQA | 200 | 0.590 | float16, mps | 9.2 | 0.817 [0.75, 0.87] |
| Qwen2.5-7B | MMLU | 200 | *missing, see below* | | | |
| SmolLM2-1.7B | TriviaQA | 200 | 0.360 | float32, mps | 2.3 | 0.809 [0.74, 0.87] |
| SmolLM2-1.7B | MMLU | 200 | 0.450 | float32, cpu | 33.0 | 0.583 [0.51, 0.67] |

MMLU has 4 options, so 0.25 is chance. TriviaQA accuracy is the share of greedy answers that match an alias
(`docs/data_and_labelling.md`).

## What the 1000-question caches show

- **Prefix check.** The first 200 rows of every 1000-question cache are identical to the 200-question cache
  (questions, greedy answers, samples and labels 200/200 for 0.5B, 1.5B and 3B; `cli check-prefix`). The
  0.5B MMLU 200 cache also starts with the old `mmlu_100` (100/100). So the 200-question results are exactly
  the first fifth of the 1000-question results.
- **The 200 set was optimistic only for 0.5B.** 0.5B drops from 0.791 to 0.712 (every method loses
  0.05-0.10 on the 800 new questions); 1.5B (0.790 → 0.789) and 3B (0.786 → 0.798) stay where they were.
- **Intervals halve.** At 1000 questions the 95 % interval is about ±0.03-0.04 instead of ±0.07-0.09,
  so differences of a few points become testable (paired bootstrap, `cli diff`).

## MMLU

- **0.5B is at chance (0.195)** and answers "A" for 133 of 200 questions; 1.5B is close to chance too
  (0.360). As noted in the data doc, MMLU is only informative from 3B on.
- **Likelihood is no signal on MMLU for small models** (AUROC 0.51-0.58 for 0.5B, 1.5B and SmolLM2): the
  answer is one letter, so its log-probability says little. 3B reaches 0.768.
- **Qwen2.5-7B MMLU is missing.** On mps, the long MMLU prompts (up to 2.5k characters) x 10 samples ran
  out of GPU memory (3B, 7B and SmolLM2 all failed). 3B and SmolLM2 were re-run on cpu. 7B on cpu needs
  bfloat16 (float32 does not fit in 24 GB), and bfloat16 on the Apple cpu ran at about 20 minutes per
  question (35 questions in 11.5 h), so the run was stopped. Next step: generate the samples in smaller
  batches on mps, after checking that the longest question fits.

## Second model family

SmolLM2-1.7B-Instruct (Hugging Face, Llama architecture, open weights) on the same 200 TriviaQA and 200 MMLU
questions as Qwen. On TriviaQA it is as accurate as Qwen2.5-1.5B (0.360 vs 0.375) and its likelihood ranks
answers as well (AUROC 0.809 vs 0.790), so the method ranking can be compared across two families at the
same size.

## How the caches were made

`configs/<model>_<dataset>_<n>.yaml`, run with `uv run python -m cli run --config ...` (one at a time) or all
day-3 jobs with `scripts/day3_queue.sh` and `scripts/day3_retry.sh`. Precision and device are stored in
each cache's `.meta.json`; a cache is only extended (resumed) with the same settings. Larger caches of a
model use the same precision and device as its 200-question cache, so the prefix stays identical. MMLU on
cpu for 3B and SmolLM2 differs from their TriviaQA device (mps); MMLU is a separate cache, so this only
affects run time.
