Run `qwen7b/triviaqa_200`, 200 questions, precision float16

| method | family | access | cost | status | auroc | ci_low | ci_high | aurc | prr | ece | n_failed | seconds_per_question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dummy_random | baseline | text_only | 1 | ok | 0.560 | 0.477 | 0.637 | 0.372 | 0.168 |  | 0 | 0.000 |
| discrete_semantic_entropy | agreement | text_only | 12 | ok | 0.734 | 0.672 | 0.793 | 0.271 | 0.410 |  | 0 | 0.201 |
| lexical_similarity | agreement | text_only | 11 | ok | 0.735 | 0.668 | 0.800 | 0.275 | 0.418 |  | 0 | 0.000 |
| likelihood_mean | baseline | logprobs | 1 | ok | 0.817 | 0.752 | 0.871 | 0.209 | 0.704 |  | 0 | 0.000 |
| mean_token_entropy | baseline | white_box | 1 | ok | 0.833 | 0.772 | 0.885 | 0.204 | 0.724 |  | 0 | 0.000 |
| msp | baseline | logprobs | 1 | ok | 0.824 | 0.764 | 0.877 | 0.207 | 0.719 |  | 0 | 0.000 |
| p_true_brainstorm | self_eval | logprobs | 12 | ok | 0.845 | 0.785 | 0.896 | 0.196 | 0.693 | 0.255 | 0 | 0.679 |
| p_true_empirical | self_eval | text_only | 11 | ok | 0.824 | 0.763 | 0.886 | 0.218 | 0.633 | 0.171 | 14 | 5.927 |
| p_true_zeroshot | self_eval | logprobs | 2 | ok | 0.870 | 0.815 | 0.920 | 0.189 | 0.767 | 0.220 | 0 | 0.344 |
| semantic_entropy | agreement | logprobs | 12 | ok | 0.735 | 0.673 | 0.792 | 0.271 | 0.410 |  | 0 | 0.196 |
| verbalized_2s | self_eval | text_only | 2 | ok | 0.734 | 0.672 | 0.796 | 0.279 | 0.409 | 0.320 | 0 | 0.905 |
