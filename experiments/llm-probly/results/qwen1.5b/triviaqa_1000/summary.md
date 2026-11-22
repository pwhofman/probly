Run `qwen1.5b/triviaqa_1000`, 1000 questions, precision float32

| method | family | access | cost | status | auroc | ci_low | ci_high | aurc | prr | ece | n_failed | seconds_per_question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| msp | baseline | logprobs | 1 | ok | 0.739 | 0.706 | 0.771 | 0.465 | 0.547 |  | 0 | 0.000 |
| likelihood_mean | baseline | logprobs | 1 | ok | 0.789 | 0.759 | 0.818 | 0.444 | 0.605 |  | 0 | 0.000 |
| mean_token_entropy | baseline | white_box | 1 | ok | 0.799 | 0.770 | 0.827 | 0.436 | 0.632 |  | 0 | 0.000 |
| lexical_similarity | agreement | text_only | 11 | ok | 0.773 | 0.743 | 0.802 | 0.454 | 0.514 |  | 0 | 0.000 |
| dummy_random | baseline | text_only | 1 | ok | 0.503 | 0.467 | 0.540 | 0.619 | 0.047 |  | 0 | 0.000 |
