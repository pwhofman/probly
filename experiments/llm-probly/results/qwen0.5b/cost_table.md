# Maliyet tablosu: refleksif yontemler

| Yontem | Kim | Model cagrisi / soru | Uretilen token | Logit gerekli? | s/soru (Qwen 0.5B) |
|---|---|---|---|---|---|
| p_true_zeroshot_norm | R1 | 1 forward pass | 0 | evet | 0.72 (R1 notebook, Gun 3) |
| p_true_brainstorm | R2 | 1 forward pass + 10 ornek* | 0 | evet | bekleniyor |
| p_true_empirical | R3 | 10 uretim | <= 3 x 10 | hayir | 0.50 |
| verbalized_2s | R3 | 1 uretim | <= 8 | hayir | 0.11 |

*Brainstorm, B1'in veri dosyasindaki 10 ornegi tekrar kullanir; bu ornekler semantic entropy icin zaten uretildigi icin ek maliyet sayilmaz. Sifirdan calistirilirsa +10 uretim.
Logit gerektirmeyen yontemler (R3) sadece metin gordugu icin API modellerine de tasinabilir.
