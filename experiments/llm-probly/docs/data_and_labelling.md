# Data & labelling

*Owner: DATA. Numbers as of Friday 13 Nov; 3B and the second half of the hand check are added when they are in.*

## Questions

We use the **TriviaQA** validation split (`mandarjoshi/trivia_qa`, `rc.nocontext`), short factual questions
with a list of accepted answers ("aliases"). 200 questions are drawn with a fixed seed (42), so every model
answers exactly the same questions in the same order.

## Generation: one cache per model, shared by every method

Each model answers every question once, and all methods score that same cache. This keeps the comparison
fair and saves compute: no method generates its own answers.

For every question the cache stores

| Field | What it is | Used by |
|---|---|---|
| `greedy_answer` | the answer every method judges (no randomness) | everyone |
| `greedy_ll_sum`, `greedy_ll_mean` | log likelihood of that answer, summed and per token | likelihood baselines |
| `greedy_token_logprobs`, `greedy_token_entropies` | log probability and entropy of every answer token | MSP, perplexity, mean token entropy |
| `samples` (10), `samples_ll_mean`, `samples_ll_sum` | ten answers with temperature 0.7, with log likelihoods | agreement methods (SE, discrete SE, …), brainstorm p(True) |
| `correct`, `samples_correct` | correctness labels (below) | evaluation |

Settings, identical for all models: prompt *"Answer the following question with as few words as possible."*,
at most 32 new tokens, **generation stops at the first line break** (the models tend to answer and then keep
talking; only the tokens before the stop are scored, so the stored text and log likelihood describe the same
answer). Every cache has a `.meta.json` with the exact settings, library versions and the model's own
generation defaults (for Qwen2.5: `top_p=0.8`, `top_k=20`, `repetition_penalty=1.1`).

The token statistics are computed in `common/generation.py`, which reproduces the text, log likelihoods and
samples of probly's `HFTextGenerationSampler` exactly (`tests/test_generation.py`).

Re-create a cache (it saves after every question and resumes after an interruption):

    uv run python -m scripts.generate --model qwen1.5b --n 200
    uv run python -m scripts.label    --model qwen1.5b --n 200

**Precision matters.** Moving the 0.5B model from bfloat16 to float32 changed 9 of the first 50 greedy answers
(correct answers 12 → 8). Where a small model is torn between two answers, rounding decides. We fixed float32
beforehand rather than choosing by accuracy; the precision of every run is in its meta file.

## Correctness labels

An answer counts as **correct** if, after lowercasing and removing punctuation and articles, one of the
aliases appears in it as whole words (`common/correctness.py`). This is the usual TriviaQA evaluation and
handles full-sentence answers ("The Red Badge of Courage was set during the American Civil War").

| Model | Correct (greedy) | Correct (samples) |
|---|---|---|
| Qwen2.5-0.5B-Instruct | 38 / 200 = **19 %** | 20 % |
| Qwen2.5-1.5B-Instruct | 75 / 200 = **38 %** | 35 % |
| Qwen2.5-3B-Instruct | *pending* | *pending* |

## Label error rate (hand check)

Per model, 30 answers are checked by hand (15 labelled correct, 15 labelled wrong; split between two
people, `scripts/manual_check.py`). Because labels "correct" are over-sampled, the error rate is weighted
by how often each label occurs in the whole file.

| Model | Checked | Labelled correct, actually wrong | Labelled wrong, actually correct | Label error rate |
|---|---|---|---|---|
| 0.5B | 15 of 30 | 0 % | 0 % | 0 % |
| 1.5B | 15 of 30 | 0 % | 17 % | 10 % |

*Second half pending; with 15 answers each number rests on one or two answers.*

## Known weaknesses of alias matching

1. **Loose aliases accept wrong answers.** TriviaQA lists "Nixon pardon" as an answer for Gerald Ford, and
   "Earth pig" for *"What animal's name means earth pig in Dutch?"* (answer: aardvark). A model that just
   repeats the question can then be labelled correct.
2. **Short correct answers are missed.** *"The Maine."* is correct for the battleship sunk in Havana in 1898,
   but every alias is longer ("USS Maine", "Battleship Maine"), so none appears inside the answer. This
   happens more often with the larger model, whose answers are shorter, so its accuracy is probably
   somewhat underestimated.
3. **Too general answers** ("a coastal town in California" for Bodega Bay) are judged wrong by hand; the
   rule agrees here, but would not if an alias were just "California".

Turning the rule around (also accept answers contained in an alias) would fix 2 but accept too general
answers like "California", so we keep the rule and treat these as open questions: an NLI or model-judge
check against the canonical answer, compared with the hand labels, is planned.
