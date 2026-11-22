Run `qwen3b/mmlu_200`, 200 questions, precision float32

| method | family | access | cost | status | auroc | ci_low | ci_high | aurc | prr | ece | n_failed | seconds_per_question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| msp | baseline | logprobs | 1 | ok | 0.768 | 0.700 | 0.834 | 0.209 | 0.587 |  | 0 | 0.000 |
| likelihood_mean | baseline | logprobs | 1 | ok | 0.768 | 0.700 | 0.834 | 0.209 | 0.587 |  | 0 | 0.000 |
| mean_token_entropy | baseline | white_box | 1 | ok | 0.769 | 0.702 | 0.834 | 0.208 | 0.592 |  | 0 | 0.000 |
| lexical_similarity | agreement | text_only | 11 | ok | 0.601 | 0.553 | 0.650 | 0.318 | 0.172 |  | 0 | 0.000 |
| dummy_random | baseline | text_only | 1 | ok | 0.551 | 0.468 | 0.629 | 0.333 | 0.150 |  | 0 | 0.000 |
