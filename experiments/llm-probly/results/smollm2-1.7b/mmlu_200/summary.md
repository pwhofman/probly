Run `smollm2-1.7b/mmlu_200`, 200 questions, precision float32

| method | family | access | cost | status | auroc | ci_low | ci_high | aurc | prr | ece | n_failed | seconds_per_question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| msp | baseline | logprobs | 1 | ok | 0.627 | 0.546 | 0.702 | 0.452 | 0.366 |  | 0 | 0.000 |
| likelihood_mean | baseline | logprobs | 1 | ok | 0.583 | 0.510 | 0.668 | 0.476 | 0.310 |  | 0 | 0.000 |
| mean_token_entropy | baseline | white_box | 1 | ok | 0.564 | 0.488 | 0.650 | 0.498 | 0.221 |  | 0 | 0.000 |
| lexical_similarity | agreement | text_only | 11 | ok | 0.625 | 0.548 | 0.701 | 0.460 | 0.289 |  | 0 | 0.000 |
| dummy_random | baseline | text_only | 1 | ok | 0.527 | 0.440 | 0.605 | 0.523 | 0.103 |  | 0 | 0.000 |
