Run `smollm2-1.7b/triviaqa_5`, 5 questions, precision float32

| method | family | access | cost | status | auroc | ci_low | ci_high | aurc | prr | ece | n_failed | seconds_per_question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| msp | baseline | logprobs | 1 | ok | 0.667 | 0.000 | 1.000 | 0.258 | 0.568 |  | 0 | 0.000 |
| likelihood_mean | baseline | logprobs | 1 | ok | 0.667 | 0.000 | 1.000 | 0.321 | 0.383 |  | 0 | 0.000 |
| mean_token_entropy | baseline | white_box | 1 | ok | 0.833 | 0.250 | 1.000 | 0.196 | 0.753 |  | 0 | 0.000 |
| lexical_similarity | agreement | text_only | 11 | ok | 0.833 | 0.250 | 1.000 | 0.196 | 0.753 |  | 0 | 0.000 |
| dummy_random | baseline | text_only | 1 | ok | 0.500 | 0.000 | 1.000 | 0.383 | 0.198 |  | 0 | 0.000 |
