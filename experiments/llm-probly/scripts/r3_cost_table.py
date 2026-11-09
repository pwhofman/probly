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
    "p_true_zeroshot_norm": ("R1", "1 forward pass", "0", "yes", (0.721, "R1 notebook, Day 3")),
    "p_true_brainstorm": ("R2", "1 forward pass + 10 samples*", "0", "yes", None),
    "p_true_empirical": ("R3", "10 generations", "<= 3 x 10", "no", None),
    "verbalized_2s": ("R3", "1 generation", "<= 8", "no", None),
}


def main():
    lines = [
        "| Method | Who | Model calls / question | Generated tokens | Needs logits? | s / question (Qwen 0.5B) |",
        "|---|---|---|---|---|---|",
    ]
    for method, (who, calls, tokens, logits, known) in METHODS.items():
        f = RESULTS / f"{method}.json"
        if f.exists() and "seconds_per_question" in json.loads(f.read_text()):
            sec = f"{json.loads(f.read_text())['seconds_per_question']:.2f}"
        elif known is not None:
            sec = f"{known[0]:.2f} ({known[1]})"
        else:
            sec = "pending"
        lines.append(f"| {method} | {who} | {calls} | {tokens} | {logits} | {sec} |")
    note = (
        "\n\n*Brainstorm reuses the 10 samples in B1's data file; they are generated for semantic entropy "
        "anyway, so they add no extra cost here. Run from scratch: +10 generations.\n"
        "Methods without logits (R3) only see text, so they also work with API models.\n"
    )
    table = "\n".join(lines)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "cost_table.md").write_text("# Cost table: reflexive methods\n\n" + table + note)
    print(table + note)
    print(f"Saved: {RESULTS / 'cost_table.md'}")


if __name__ == "__main__":
    main()
