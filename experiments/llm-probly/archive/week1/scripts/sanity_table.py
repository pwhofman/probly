"""Gun 2 aksami: R1, R2 ve R3'un sanity test sonuclarini tek tabloda birlestirir.

results/sanity/<method>_<model>.json dosyalarinin hepsini okur.
Henuz gelmemis yontem/model kombinasyonlari "bekleniyor" olarak gosterilir.
Cikti: ekrana Markdown tablo + results/sanity/summary.md
"""

import json
from pathlib import Path

SANITY = Path(__file__).resolve().parents[1] / "results" / "sanity"
MODELS = ["smollm2-135m", "qwen0.5b", "qwen1.5b"]
# Plandaki 4 refleksif yontem (+ R3'un 0-10 denemesi). Baska isimle gelen dosyalar da tabloya eklenir.
PLANNED = [
    ("p_true_zeroshot_norm", "R1"),
    ("p_true_zeroshot_raw", "R1"),
    ("p_true_brainstorm", "R2"),
    ("p_true_empirical", "R3"),
    ("verbalized_2s", "R3"),
    ("verbalized_2s_10", "R3"),
]


def verdict(auroc):
    if auroc >= 0.8:
        return "calisiyor"
    if auroc >= 0.6:
        return "zayif sinyal"
    return "rastgele"


# Dosya adlarinda model iki sekilde yaziliyor: kisa ad (R3: qwen0.5b) ya da
# HF adinin son kismi (R2'nin save_sanity'si: Qwen2.5-0.5B-Instruct). Ikisini de tani.
MODEL_ALIASES = {
    "smollm2-135m": "smollm2-135m",
    "smollm2-135m-instruct": "smollm2-135m",
    "qwen0.5b": "qwen0.5b",
    "qwen2.5-0.5b-instruct": "qwen0.5b",
    "qwen1.5b": "qwen1.5b",
    "qwen2.5-1.5b-instruct": "qwen1.5b",
}


def load():
    found = {}
    for f in sorted(SANITY.glob("*.json")):
        stem = f.stem
        hit = next((a for a in MODEL_ALIASES if stem.lower().endswith("_" + a)), None)
        if hit is None:
            continue
        method = stem[: -len(hit) - 1]
        found[(method, MODEL_ALIASES[hit])] = json.loads(f.read_text())
    return found


def owner(method, owners):
    if method in owners:
        return owners[method]
    prefix = method.split("_")[0].upper()  # r2_letter_brainstorm -> R2
    return prefix if prefix in {"R1", "R2", "R3"} else "?"


def fmt_extra(r):
    if "failed" in r:
        return f"failed {int(r['failed'])}/20"
    if "unparsed_share" in r:
        return f"parse edilemeyen %{100 * r['unparsed_share']:.0f}"
    return "-"


def main():
    found = load()
    owners = dict(PLANNED)
    methods = [m for m, _ in PLANNED] + sorted({m for m, _ in found} - set(owners))

    lines = [
        "| Yontem | Kim | Model | AUROC | Gap | n_distinct | Failed / parse | s/ornek | Yorum |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for method in methods:
        for model in MODELS:
            r = found.get((method, model))
            who = owner(method, owners)
            if r is None:
                lines.append(f"| {method} | {who} | {model} | - | - | - | - | - | bekleniyor |")
                continue
            sec = r.get("seconds_per_example")
            if sec is None and "seconds_total" in r:  # R1 toplam sureyi kaydediyor (20 ornek)
                sec = r["seconds_total"] / 20
            sec_s = f"{sec:.2f}" if sec is not None else "-"
            lines.append(
                f"| {method} | {who} | {model} | {r['auroc']:.2f} | {r['gap']:+.2f} "
                f"| {int(r['n_distinct'])} | {fmt_extra(r)} | {sec_s} | {verdict(r['auroc'])} |"
            )

    table = "\n".join(lines)
    header = (
        "# Sanity test: tum refleksif yontemler x 3 model\n\n"
        "20 ornek (10 soru x dogru/yanlis cevap). AUROC >= 0.8 calisiyor, "
        "0.6-0.8 zayif sinyal, < 0.6 rastgele.\n\n"
    )
    (SANITY / "summary.md").write_text(header + table + "\n")
    print(table)
    print(f"\nKaydedildi: {SANITY / 'summary.md'}")


if __name__ == "__main__":
    main()
