# Score contract (B2)

Every method saves one file: `results/qwen0.5b/<method>.json`

1. Direction: higher = more uncertain. For reflexive methods, uncertainty = 1 - confidence.
2. Raw values: store unnormalized scores. No rescaling.
3. No missing values: if parsing fails, uncertainty = 1.0 and failed = true.
4. Same answer: all methods score the main (greedy) answer.
5. Same question ids: `ids` in the same order as `id` in `data/qwen0.5b/triviaqa_50.jsonl`.

Example:

    {"method": "p_true_zeroshot_norm", "model": "Qwen/Qwen2.5-0.5B-Instruct",
     "answer": "greedy", "ids": [0, 1, 2], "uncertainty": [0.12, 0.87, 0.40],
     "failed": [false, false, true], "seconds_per_question": 0.8}
