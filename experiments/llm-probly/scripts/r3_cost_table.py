"""Gun 3 (R3): refleksif yontemlerin maliyet tablosu (soru basina sure ve model cagrisi).

results/qwen0.5b/<method>.json dosyalarindaki seconds_per_question degerini okur.
Dosyasi henuz olmayan yontemler icin bilinen degeri (kaynagiyla) ya da "bekleniyor" yazar.
Cikti: ekrana Markdown tablo + results/qwen0.5b/cost_table.md
"""

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results" / "qwen0.5b"

# method: (kim, soru basina model cagrisi, uretilen token, logit gerekli mi, bilinen sure + kaynak)
METHODS = {
    "p_true_zeroshot_norm": ("R1", "1 forward pass", "0", "evet", (0.721, "R1 notebook, Gun 3")),
    "p_true_brainstorm": ("R2", "1 forward pass + 10 ornek*", "0", "evet", None),
    "p_true_empirical": ("R3", "10 uretim", "<= 3 x 10", "hayir", None),
    "verbalized_2s": ("R3", "1 uretim", "<= 8", "hayir", None),
}


def main():
    lines = [
        "| Yontem | Kim | Model cagrisi / soru | Uretilen token | Logit gerekli? | s/soru (Qwen 0.5B) |",
        "|---|---|---|---|---|---|",
    ]
    for method, (who, calls, tokens, logits, known) in METHODS.items():
        f = RESULTS / f"{method}.json"
        if f.exists() and "seconds_per_question" in json.loads(f.read_text()):
            sec = f"{json.loads(f.read_text())['seconds_per_question']:.2f}"
        elif known is not None:
            sec = f"{known[0]:.2f} ({known[1]})"
        else:
            sec = "bekleniyor"
        lines.append(f"| {method} | {who} | {calls} | {tokens} | {logits} | {sec} |")
    note = (
        "\n\n*Brainstorm, B1'in veri dosyasindaki 10 ornegi tekrar kullanir; bu ornekler semantic "
        "entropy icin zaten uretildigi icin ek maliyet sayilmaz. Sifirdan calistirilirsa +10 uretim.\n"
        "Logit gerektirmeyen yontemler (R3) sadece metin gordugu icin API modellerine de tasinabilir.\n"
    )
    table = "\n".join(lines)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "cost_table.md").write_text("# Maliyet tablosu: refleksif yontemler\n\n" + table + note)
    print(table + note)
    print(f"Kaydedildi: {RESULTS / 'cost_table.md'}")


if __name__ == "__main__":
    main()
