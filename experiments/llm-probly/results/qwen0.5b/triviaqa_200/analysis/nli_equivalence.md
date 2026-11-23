# NLI code paths with the same NLI model (qwen0.5b, TriviaQA-200, first 20 questions)

NLI model `microsoft/deberta-base-mnli`, device `mps`, float32. Made by `scripts/check_nli_equivalence.py`.

| path | NLI pairs / question | seconds / question | clusters = old | max abs SE diff | max abs DSE diff |
|---|---|---|---|---|---|
| old | 43.1 | 0.426 | 20/20 | 0.0e+00 | 0.0e+00 |
| A1 | 110.0 | 0.429 | 20/20 | 0.0e+00 | 0.0e+00 |
| A1+A2 | 49.9 | 0.263 | 20/20 | 0.0e+00 | 0.0e+00 |
| A1-A3 | 49.9 | 0.257 | 20/20 | 0.0e+00 | 0.0e+00 |

A1 vs A1-A3 over 1732 pairs of different texts: largest probability difference 3.0e-06, label flips 0. Pairs of identical texts: 468, of which the model (A1) did not say entailment for 0 (A2 sets them to entailment without asking the model).

- `old`: week-2 lazy greedy clustering (only the pairs it needs, but per method and per question).
- `A1`: full matrix per question; costs more pairs than `old` alone, but SE, DSE and the graph
  measures share it (one NLI pass instead of one per method).
- `A1+A2`: identical answers scored once. `A1-A3`: plus 16 questions per model batch (the benchmark).
- Times exclude loading the model; the model is warmed up before the first path.
