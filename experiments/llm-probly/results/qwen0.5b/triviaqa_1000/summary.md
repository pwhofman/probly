Run `qwen0.5b/triviaqa_1000`, 1000 questions, precision float32

| method | family | access | cost | status | auroc | ci_low | ci_high | vs_best | d_auroc | d_ci_low | d_ci_high | aurc | prr | ece | n_failed | seconds_per_question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| msp | baseline | logprobs | 1 | ok | 0.639 | 0.595 | 0.683 | worse | -0.073 | -0.114 | -0.029 | 0.720 | 0.302 |  | 0 | 0.000 |
| likelihood_mean | baseline | logprobs | 1 | ok | 0.712 | 0.672 | 0.747 | best | 0.000 | 0.000 | 0.000 | 0.689 | 0.401 |  | 0 | 0.000 |
| mean_token_entropy | baseline | white_box | 1 | ok | 0.696 | 0.656 | 0.735 | worse | -0.016 | -0.028 | -0.003 | 0.698 | 0.372 |  | 0 | 0.000 |
| lexical_similarity | agreement | text_only | 11 | ok | 0.681 | 0.642 | 0.717 | tied | -0.031 | -0.066 | 0.003 | 0.714 | 0.325 |  | 0 | 0.000 |
| dummy_random | baseline | text_only | 1 | ok | 0.515 | 0.465 | 0.561 | worse | -0.197 | -0.259 | -0.135 | 0.780 | 0.065 |  | 0 | 0.000 |
