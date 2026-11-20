# triviaqa, 200 questions: methods x model size

## Models

| model | questions | accuracy | precision |
|---|---|---|---|
| qwen0.5b | 200 | 0.190 | float32 |
| qwen1.5b | 200 | 0.375 | float32 |
| qwen3b | 200 | 0.485 | float32 |
| qwen7b | 200 | 0.590 | float16 |


## AUROC (higher is better, 0.5 = random)

| method | qwen0.5b | qwen1.5b | qwen3b | qwen7b |
|---|---|---|---|---|
| discrete_semantic_entropy | 0.744 | 0.783 |  |  |
| dummy_random | 0.567 |  |  |  |
| lexical_similarity | 0.757 | 0.752 | 0.757 | 0.735 |
| likelihood_mean | 0.791 | 0.790 | 0.786 | 0.817 |
| mean_token_entropy | 0.773 | 0.805 | 0.817 | 0.833 |
| msp | 0.680 | 0.748 | 0.753 | 0.824 |
| p_true_brainstorm | 0.685 | 0.806 | 0.820 |  |
| p_true_empirical | 0.518 | 0.770 | 0.830 |  |
| p_true_zeroshot | 0.669 | 0.825 | 0.877 |  |
| semantic_entropy | 0.740 | 0.781 |  |  |
| verbalized_2s | 0.584 | 0.538 | 0.713 |  |


## PRR (higher is better, 1 = perfect, 0 = random)

| method | qwen0.5b | qwen1.5b | qwen3b | qwen7b |
|---|---|---|---|---|
| discrete_semantic_entropy | 0.497 | 0.520 |  |  |
| dummy_random | 0.160 |  |  |  |
| lexical_similarity | 0.467 | 0.479 | 0.457 | 0.418 |
| likelihood_mean | 0.576 | 0.618 | 0.631 | 0.704 |
| mean_token_entropy | 0.548 | 0.653 | 0.684 | 0.724 |
| msp | 0.400 | 0.552 | 0.573 | 0.719 |
| p_true_brainstorm | 0.291 | 0.577 | 0.689 |  |
| p_true_empirical | -0.000 | 0.472 | 0.650 |  |
| p_true_zeroshot | 0.275 | 0.650 | 0.793 |  |
| semantic_entropy | 0.490 | 0.517 |  |  |
| verbalized_2s | 0.090 | 0.081 | 0.350 |  |


## AURC (lower is better)

| method | qwen0.5b | qwen1.5b | qwen3b | qwen7b |
|---|---|---|---|---|
| discrete_semantic_entropy | 0.683 | 0.448 |  |  |
| dummy_random | 0.771 |  |  |  |
| lexical_similarity | 0.692 | 0.460 | 0.363 | 0.275 |
| likelihood_mean | 0.671 | 0.433 | 0.321 | 0.209 |
| mean_token_entropy | 0.680 | 0.421 | 0.299 | 0.204 |
| msp | 0.719 | 0.462 | 0.342 | 0.207 |
| p_true_brainstorm | 0.730 | 0.427 | 0.299 |  |
| p_true_empirical | 0.805 | 0.467 | 0.298 |  |
| p_true_zeroshot | 0.744 | 0.411 | 0.262 |  |
| semantic_entropy | 0.689 | 0.447 |  |  |
| verbalized_2s | 0.777 | 0.606 | 0.395 |  |


## ECE, only probability methods (lower is better)

| method | qwen0.5b | qwen1.5b | qwen3b | qwen7b |
|---|---|---|---|---|
| p_true_brainstorm | 0.662 | 0.389 | 0.259 |  |
| p_true_empirical | 0.651 | 0.314 | 0.181 |  |
| p_true_zeroshot | 0.690 | 0.335 | 0.176 |  |
| verbalized_2s | 0.722 | 0.563 | 0.412 |  |
