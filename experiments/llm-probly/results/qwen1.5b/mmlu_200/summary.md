Run `qwen1.5b/mmlu_200`, 200 questions, precision float32

| method | family | access | cost | status | auroc | ci_low | ci_high | aurc | prr | ece | n_failed | seconds_per_question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| msp | baseline | logprobs | 1 | ok | 0.505 | 0.419 | 0.587 | 0.646 | 0.006 |  | 0 | 0.000 |
| likelihood_mean | baseline | logprobs | 1 | ok | 0.505 | 0.419 | 0.587 | 0.646 | 0.006 |  | 0 | 0.000 |
| mean_token_entropy | baseline | white_box | 1 | ok | 0.508 | 0.421 | 0.594 | 0.638 | 0.007 |  | 0 | 0.000 |
| lexical_similarity | agreement | text_only | 11 | ok | 0.526 | 0.451 | 0.602 | 0.625 | 0.055 |  | 0 | 0.000 |
| dummy_random | baseline | text_only | 1 | ok | 0.553 | 0.469 | 0.638 | 0.610 | 0.153 |  | 0 | 0.000 |
