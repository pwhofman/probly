Run `qwen3b/triviaqa_200`, 200 questions, precision float32

| method | family | status | auroc | ci_low | ci_high | aurc | prr | ece | n_failed | seconds_per_question |
|---|---|---|---|---|---|---|---|---|---|---|
| msp | baseline | ok | 0.753 | 0.682 | 0.818 | 0.342 | 0.573 |  | 0 | 0.000 |
| mean_token_entropy | baseline | ok | 0.817 | 0.757 | 0.871 | 0.299 | 0.684 |  | 0 | 0.000 |
| lexical_similarity | agreement | ok | 0.757 | 0.692 | 0.816 | 0.363 | 0.457 |  | 0 | 0.000 |
| likelihood_mean | baseline | ok | 0.786 | 0.719 | 0.843 | 0.321 | 0.631 |  | 0 | 0.000 |
| p_true_zeroshot | self_eval | ok | 0.877 | 0.830 | 0.921 | 0.262 | 0.793 | 0.176 | 0 | 0.244 |
| p_true_brainstorm | self_eval | ok | 0.820 | 0.764 | 0.873 | 0.299 | 0.689 | 0.259 | 0 | 0.471 |
| verbalized_2s | self_eval | ok | 0.713 | 0.659 | 0.766 | 0.395 | 0.350 | 0.412 | 0 | 0.771 |
| p_true_empirical | self_eval | ok | 0.830 | 0.775 | 0.880 | 0.298 | 0.650 | 0.181 | 0 | 3.157 |
