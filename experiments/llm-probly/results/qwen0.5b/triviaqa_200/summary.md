Run `qwen0.5b/triviaqa_200`, 200 questions, precision float32

| method | family | status | auroc | ci_low | ci_high | aurc | prr | ece | n_failed | seconds_per_question |
|---|---|---|---|---|---|---|---|---|---|---|
| dummy_random | baseline | ok | 0.567 | 0.467 | 0.669 | 0.771 | 0.160 |  | 0 | 0.000 |
| likelihood_mean | baseline | ok | 0.791 | 0.701 | 0.877 | 0.671 | 0.576 |  | 0 | 0.000 |
| p_true_brainstorm | self_eval | ok | 0.685 | 0.588 | 0.769 | 0.730 | 0.291 | 0.662 | 0 | 0.293 |
| p_true_empirical | self_eval | ok | 0.518 | 0.421 | 0.611 | 0.805 | -0.000 | 0.651 | 0 | 0.847 |
| p_true_zeroshot | self_eval | ok | 0.669 | 0.581 | 0.751 | 0.744 | 0.275 | 0.690 | 0 | 0.113 |
| verbalized_2s | self_eval | ok | 0.584 | 0.505 | 0.658 | 0.777 | 0.090 | 0.722 | 0 | 0.194 |
