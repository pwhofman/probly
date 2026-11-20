# Methods

All methods are plugins in `methods/` (see CONTRACT.md). Each one has a family, and each one reads the same cached generations.

Family, model access (`text_only` / `logprobs` / `white_box`) and cost per question (generation passes,
greedy answer included) are kept in one place, `common/registry.py`; `uv run python -m cli methods` prints them.
The selector reads the same registry.

| Method | Family | Needs logits? | Model calls per question |
|---|---|---|---|
| semantic_entropy | agreement | yes (sample log-likelihoods, `samples_ll_mean`) | 0 LLM calls (reads the 10 cached samples) + NLI model on sample pairs (up to 45) |
| discrete_semantic_entropy | agreement | no (cluster sizes only) | 0 LLM calls + NLI model on sample pairs (up to 45) |
| lexical_similarity | agreement | no (text only, Rouge-L between samples) | 0 |
| p_true_zeroshot | self-eval | yes (probability of "True") | 1 extra LLM call |
| p_true_brainstorm | self-eval | yes (probability of the answer option "A" vs "B") | 1 extra LLM call (the 10 cached samples are shown in the prompt) |
| p_true_empirical | self-eval | yes (probability of "True") | 10 extra generations + scoring |
| verbalized_2s | self-eval | no (the model writes a confidence number, read from the text) | 1 short extra LLM call |
| msp | baseline | yes (cached greedy log-probs) | 0 |
| likelihood_mean / perplexity | baseline | yes (cached greedy log-likelihood) | 0 |
| mean_token_entropy | baseline | yes (cached token entropies) | 0 |

## How the cost works

Generation happens once per model: 1 greedy answer and 10 samples per question are cached. Every method reads that cache, so the methods with 0 LLM calls only do arithmetic on saved data. The self-evaluation methods need the LLM again at scoring time, because they ask the model to judge its own answer.

The agreement methods (SE and discrete SE) also run an NLI model (`microsoft/deberta-base-mnli`) on the samples to group them by meaning. This is a small classifier, not the LLM. The clusterer is greedy: each new sample is compared with the existing clusters, so a question needs between 9 and 45 NLI pair checks, depending on how many clusters there are.

## Known limitations

- SE and discrete SE each run the NLI model separately, so the NLI work is done twice per question. The `.nli.npz` cache is meant to remove this; it is planned for next week.
- SE differs from discrete SE only in the likelihood weights, so the two give almost the same AUROC (0.740 vs 0.744 at 0.5B, 0.781 vs 0.783 at 1.5B).
