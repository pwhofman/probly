# Cost table: reflexive methods

| Method | Who | Model calls / question | Generated tokens | Needs logits? | s / question (Qwen 0.5B) |
|---|---|---|---|---|---|
| p_true_zeroshot_norm | R1 | 1 forward pass | 0 | yes | 0.79 |
| p_true_brainstorm | R2 | 1 forward pass + 10 samples* | 0 | yes | pending |
| p_true_empirical | R3 | 10 generations | <= 3 x 10 | no | 0.50 |
| verbalized_2s | R3 | 1 generation | <= 8 | no | 0.11 |

*Brainstorm reuses the 10 samples in B1's data file; they are generated for semantic entropy anyway, so they add no extra cost here. Run from scratch: +10 generations.
Methods without logits (R3) only see text, so they also work with API models.
