Run `qwen7b/triviaqa_200`, 200 questions, precision float16

| method | family | status | auroc | ci_low | ci_high | aurc | prr | ece | n_failed | seconds_per_question |
|---|---|---|---|---|---|---|---|---|---|---|
| msp | baseline | ok | 0.824 | 0.764 | 0.877 | 0.207 | 0.719 |  | 0 | 0.000 |
| perplexity | baseline | ok | 0.817 | 0.752 | 0.871 | 0.209 | 0.704 |  | 0 | 0.000 |
| mean_token_entropy | baseline | ok | 0.833 | 0.772 | 0.885 | 0.204 | 0.724 |  | 0 | 0.000 |
| lexical_similarity | agreement | ok | 0.735 | 0.668 | 0.800 | 0.275 | 0.418 |  | 0 | 0.000 |
| likelihood_mean | baseline | ok | 0.817 | 0.752 | 0.871 | 0.209 | 0.704 |  | 0 | 0.000 |
