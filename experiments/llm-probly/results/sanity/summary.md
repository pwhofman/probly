# Sanity test: tum refleksif yontemler x 3 model

20 ornek (10 soru x dogru/yanlis cevap). AUROC >= 0.8 calisiyor, 0.6-0.8 zayif sinyal, < 0.6 rastgele.

| Yontem | Kim | Model | AUROC | Gap | n_distinct | Failed / parse | s/ornek | Yorum |
|---|---|---|---|---|---|---|---|---|
| p_true_zeroshot_norm | R1 | smollm2-135m | 0.40 | -0.02 | 20 | - | 0.19 | rastgele |
| p_true_zeroshot_norm | R1 | qwen0.5b | 0.70 | +0.06 | 20 | - | 0.57 | zayif sinyal |
| p_true_zeroshot_norm | R1 | qwen1.5b | 0.97 | +0.68 | 17 | - | 5.91 | calisiyor |
| p_true_zeroshot_raw | R1 | smollm2-135m | 0.47 | -0.07 | 6 | - | 0.17 | rastgele |
| p_true_zeroshot_raw | R1 | qwen0.5b | 0.57 | +0.22 | 14 | - | 0.64 | rastgele |
| p_true_zeroshot_raw | R1 | qwen1.5b | 0.91 | +2.05 | 17 | - | 5.86 | calisiyor |
| p_true_brainstorm | R2 | smollm2-135m | - | - | - | - | - | bekleniyor |
| p_true_brainstorm | R2 | qwen0.5b | - | - | - | - | - | bekleniyor |
| p_true_brainstorm | R2 | qwen1.5b | - | - | - | - | - | bekleniyor |
| p_true_empirical | R3 | smollm2-135m | 0.52 | -0.02 | 7 | parse edilemeyen %42 | 0.18 | rastgele |
| p_true_empirical | R3 | qwen0.5b | 0.58 | +0.06 | 7 | parse edilemeyen %8 | 0.41 | rastgele |
| p_true_empirical | R3 | qwen1.5b | 0.94 | +0.88 | 4 | parse edilemeyen %0 | 1.27 | calisiyor |
| verbalized_2s | R3 | smollm2-135m | 0.45 | -0.10 | 6 | failed 0/20 | 0.09 | rastgele |
| verbalized_2s | R3 | qwen0.5b | 0.54 | +0.01 | 3 | failed 0/20 | 0.08 | rastgele |
| verbalized_2s | R3 | qwen1.5b | 0.60 | +0.01 | 2 | failed 0/20 | 0.21 | zayif sinyal |
| verbalized_2s_10 | R3 | smollm2-135m | 0.50 | +0.00 | 2 | failed 0/20 | 0.08 | rastgele |
| verbalized_2s_10 | R3 | qwen0.5b | 0.45 | -0.01 | 2 | failed 0/20 | 0.06 | rastgele |
| verbalized_2s_10 | R3 | qwen1.5b | 0.59 | +0.02 | 3 | failed 0/20 | 0.15 | rastgele |
| r2_letter_brainstorm | R2 | smollm2-135m | - | - | - | - | - | bekleniyor |
| r2_letter_brainstorm | R2 | qwen0.5b | 1.00 | +0.51 | 16 | - | - | calisiyor |
| r2_letter_brainstorm | R2 | qwen1.5b | - | - | - | - | - | bekleniyor |
| r2_letter_nobrainstorm | R2 | smollm2-135m | - | - | - | - | - | bekleniyor |
| r2_letter_nobrainstorm | R2 | qwen0.5b | 0.76 | +0.15 | 15 | - | - | zayif sinyal |
| r2_letter_nobrainstorm | R2 | qwen1.5b | - | - | - | - | - | bekleniyor |
