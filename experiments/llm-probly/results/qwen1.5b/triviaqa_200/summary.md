Run `qwen1.5b/triviaqa_200`, 200 questions, precision float32

| method | family | access | cost | status | auroc | ci_low | ci_high | aurc | prr | ece | n_failed | seconds_per_question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dummy_random | baseline | text_only | 1 | ok | 0.521 | 0.439 | 0.610 | 0.597 | 0.131 |  | 0 | 0.000 |
| discrete_semantic_entropy | agreement | text_only | 12 | ok | 0.783 | 0.719 | 0.838 | 0.448 | 0.520 |  | 0 | 0.768 |
| lexical_similarity | agreement | text_only | 11 | ok | 0.752 | 0.680 | 0.817 | 0.460 | 0.479 |  | 0 | 0.000 |
| likelihood_mean | baseline | logprobs | 1 | ok | 0.790 | 0.723 | 0.846 | 0.433 | 0.618 |  | 0 | 0.000 |
| mean_token_entropy | baseline | white_box | 1 | ok | 0.805 | 0.747 | 0.860 | 0.421 | 0.653 |  | 0 | 0.000 |
| msp | baseline | logprobs | 1 | ok | 0.748 | 0.673 | 0.817 | 0.462 | 0.552 |  | 0 | 0.000 |
| p_true_brainstorm | self_eval | logprobs | 12 | ok | 0.806 | 0.738 | 0.864 | 0.427 | 0.577 | 0.389 | 0 | 0.505 |
| p_true_empirical | self_eval | text_only | 11 | ok | 0.770 | 0.708 | 0.826 | 0.467 | 0.472 | 0.314 | 0 | 2.252 |
| p_true_zeroshot | self_eval | logprobs | 2 | ok | 0.825 | 0.764 | 0.883 | 0.411 | 0.650 | 0.335 | 0 | 0.313 |
| semantic_entropy | agreement | logprobs | 12 | ok | 0.781 | 0.716 | 0.837 | 0.447 | 0.517 |  | 0 | 0.725 |
| verbalized_2s | self_eval | text_only | 2 | ok | 0.538 | 0.488 | 0.587 | 0.606 | 0.081 | 0.563 | 0 | 0.560 |
