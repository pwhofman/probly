Run `qwen0.5b/mmlu_200`, 200 questions, precision float32

| method | family | access | cost | status | auroc | ci_low | ci_high | aurc | prr | ece | n_failed | seconds_per_question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| msp | baseline | logprobs | 1 | ok | 0.501 | 0.394 | 0.605 | 0.798 | 0.094 |  | 0 | 0.000 |
| likelihood_mean | baseline | logprobs | 1 | ok | 0.543 | 0.436 | 0.648 | 0.776 | 0.168 |  | 0 | 0.000 |
| mean_token_entropy | baseline | white_box | 1 | ok | 0.580 | 0.471 | 0.685 | 0.759 | 0.203 |  | 0 | 0.000 |
| lexical_similarity | agreement | text_only | 11 | ok | 0.507 | 0.403 | 0.611 | 0.794 | 0.068 |  | 0 | 0.000 |
| dummy_random | baseline | text_only | 1 | ok | 0.516 | 0.420 | 0.611 | 0.808 | -0.062 |  | 0 | 0.000 |
