Run `qwen3b/triviaqa_200`, 200 questions, precision float32

| method | family | access | cost | status | auroc | ci_low | ci_high | aurc | prr | ece | n_failed | seconds_per_question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dummy_random | baseline | text_only | 1 | ok | 0.500 | 0.417 | 0.584 | 0.504 | 0.093 |  | 0 | 0.000 |
| discrete_semantic_entropy | agreement | text_only | 12 | ok | 0.765 | 0.704 | 0.822 | 0.353 | 0.466 |  | 0 | 0.271 |
| lexical_similarity | agreement | text_only | 11 | ok | 0.757 | 0.692 | 0.816 | 0.363 | 0.457 |  | 0 | 0.000 |
| likelihood_mean | baseline | logprobs | 1 | ok | 0.786 | 0.719 | 0.843 | 0.321 | 0.631 |  | 0 | 0.000 |
| mean_token_entropy | baseline | white_box | 1 | ok | 0.817 | 0.757 | 0.871 | 0.299 | 0.684 |  | 0 | 0.000 |
| msp | baseline | logprobs | 1 | ok | 0.753 | 0.682 | 0.818 | 0.342 | 0.573 |  | 0 | 0.000 |
| p_true_brainstorm | self_eval | logprobs | 12 | ok | 0.820 | 0.764 | 0.873 | 0.299 | 0.689 | 0.259 | 0 | 0.471 |
| p_true_empirical | self_eval | text_only | 11 | ok | 0.830 | 0.775 | 0.880 | 0.298 | 0.650 | 0.181 | 0 | 3.157 |
| p_true_zeroshot | self_eval | logprobs | 2 | ok | 0.877 | 0.830 | 0.921 | 0.262 | 0.793 | 0.176 | 0 | 0.244 |
| semantic_entropy | agreement | logprobs | 12 | ok | 0.759 | 0.696 | 0.815 | 0.356 | 0.460 |  | 0 | 0.276 |
| verbalized_2s | self_eval | text_only | 2 | ok | 0.713 | 0.659 | 0.766 | 0.395 | 0.350 | 0.412 | 0 | 0.771 |
