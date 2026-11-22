Run `smollm2-1.7b/triviaqa_200`, 200 questions, precision float32

| method | family | access | cost | status | auroc | ci_low | ci_high | aurc | prr | ece | n_failed | seconds_per_question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| msp | baseline | logprobs | 1 | ok | 0.795 | 0.726 | 0.855 | 0.431 | 0.671 |  | 0 | 0.000 |
| likelihood_mean | baseline | logprobs | 1 | ok | 0.809 | 0.738 | 0.871 | 0.423 | 0.685 |  | 0 | 0.000 |
| mean_token_entropy | baseline | white_box | 1 | ok | 0.816 | 0.741 | 0.879 | 0.424 | 0.694 |  | 0 | 0.000 |
| lexical_similarity | agreement | text_only | 11 | ok | 0.796 | 0.726 | 0.854 | 0.445 | 0.585 |  | 0 | 0.000 |
| dummy_random | baseline | text_only | 1 | ok | 0.525 | 0.440 | 0.612 | 0.613 | 0.098 |  | 0 | 0.000 |
