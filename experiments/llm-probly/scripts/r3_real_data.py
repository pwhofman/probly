"""Gun 3 (R3): empirical p(True) ve verbalized 2S, 50 TriviaQA sorusunda (Qwen2.5-0.5B).

- B1'in data/qwen0.5b/triviaqa_50.jsonl dosyasindaki ana (greedy) cevaplari puanlar
- B2'nin contract'ina gore results/qwen0.5b/<method>.json olarak kaydeder
  (uncertainty = 1 - guven, parse edilemezse uncertainty 1.0 ve failed = true)
- Ek olarak <method>_details.json: her sorunun ham cevaplari (hata ayiklama icin)
- B2'nin evaluate()'i gelene kadar AUROC ve AURC'yi kabaca kendisi hesaplar (onizleme)

Calistirma (repo kokunden):
    uv run python experiments/llm-probly/scripts/r3_real_data.py
    uv run python experiments/llm-probly/scripts/r3_real_data.py --methods verbalized_2s
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from probly.representer.sampler.huggingface import load_model
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r3_methods as r3  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]  # experiments/llm-probly
DATA = ROOT / "data" / "qwen0.5b" / "triviaqa_50.jsonl"
OUT = ROOT / "results" / "qwen0.5b"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def score_empirical(model, tok, q, a):
    p, texts, unparsed = r3.empirical_p_true(model, tok, q, a)
    # 10 cevabin hicbiri True/False diye okunamadiysa olcum yok -> failed
    failed = unparsed == len(texts)
    return (0.0 if failed else p), failed, {"raw": texts, "unparsed_of_10": unparsed}


def score_verbalized(model, tok, q, a):
    conf, failed, reply = r3.verbalized_2s(model, tok, q, a, scale=100)
    return conf, failed, {"raw": reply}


METHODS = {
    "p_true_empirical": (score_empirical, 10),  # (fonksiyon, soru basina model cagrisi)
    "verbalized_2s": (score_verbalized, 1),
}


def aurc(uncertainty, correct):
    """Area under the risk-coverage curve: en emin oldugu sorulardan baslayip
    sirayla ekle, her adimda hata oranini al, ortalamasini al. Dusuk = iyi."""
    order = np.argsort(uncertainty, kind="stable")
    errors = 1 - np.asarray(correct, dtype=float)[order]
    risks = np.cumsum(errors) / np.arange(1, len(errors) + 1)
    return float(risks.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    args = ap.parse_args()

    rows = [json.loads(line) for line in DATA.read_text().splitlines() if line.strip()]
    correct = np.array([r["correct"] for r in rows], dtype=int)
    print(f"{len(rows)} soru, greedy cevap dogruluk orani: {correct.mean():.2f}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, tok = load_model(MODEL)
    model.to(device)
    OUT.mkdir(parents=True, exist_ok=True)

    for method in args.methods:
        fn, calls = METHODS[method]
        torch.manual_seed(42)
        unc, failed, details, times = [], [], [], []
        for i, r in enumerate(rows):
            t0 = time.perf_counter()
            conf, f, info = fn(model, tok, r["question"], r["greedy_answer"])
            times.append(time.perf_counter() - t0)
            unc.append(1.0 - conf)
            failed.append(bool(f))
            details.append({"id": r["id"], "answer": r["greedy_answer"], "correct": bool(r["correct"]),
                            "confidence": conf, "failed": bool(f), **info})
            print(f"\r{method}: {i + 1}/{len(rows)}", end="", flush=True)
        print()

        result = {
            "method": method,
            "model": MODEL,
            "answer": "greedy",
            "ids": [r["id"] for r in rows],
            "uncertainty": unc,
            "failed": failed,
            "seconds_per_question": float(np.mean(times)),
            "model_calls_per_question": calls,
        }
        (OUT / f"{method}.json").write_text(json.dumps(result, indent=2) + "\n")
        (OUT / f"{method}_details.json").write_text(json.dumps(details, indent=2, ensure_ascii=False) + "\n")

        # Onizleme: hata = pozitif sinif, yuksek uncertainty -> hata bekliyoruz
        auroc = roc_auc_score(1 - correct, unc) if 0 < correct.sum() < len(correct) else float("nan")
        print(f"  AUROC={auroc:.3f}  AURC={aurc(unc, correct):.3f}  "
              f"(rastgele AURC ~ {1 - correct.mean():.3f})  failed={sum(failed)}/{len(rows)}  "
              f"n_distinct={len(np.unique(np.round(unc, 3)))}  {np.mean(times):.2f} s/soru")


if __name__ == "__main__":
    main()
