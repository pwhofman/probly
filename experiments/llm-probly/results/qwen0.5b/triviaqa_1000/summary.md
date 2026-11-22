Run `qwen0.5b/triviaqa_1000`, 1000 questions, precision float32

| method | family | access | cost | status | auroc | ci_low | ci_high | aurc | prr | ece | n_failed | seconds_per_question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| msp | baseline | logprobs | 1 | ok | 0.639 | 0.595 | 0.683 | 0.720 | 0.302 |  | 0 | 0.000 |
| likelihood_mean | baseline | logprobs | 1 | ok | 0.712 | 0.672 | 0.747 | 0.689 | 0.401 |  | 0 | 0.000 |
| mean_token_entropy | baseline | white_box | 1 | ok | 0.696 | 0.656 | 0.735 | 0.698 | 0.372 |  | 0 | 0.000 |
| lexical_similarity | agreement | text_only | 11 | ok | 0.681 | 0.642 | 0.717 | 0.714 | 0.325 |  | 0 | 0.000 |
| dummy_random | baseline | text_only | 1 | ok | 0.515 | 0.465 | 0.561 | 0.780 | 0.065 |  | 0 | 0.000 |
