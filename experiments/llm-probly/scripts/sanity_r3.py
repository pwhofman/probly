import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from probly.representer.sampler.huggingface import load_model

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r3_methods as r3  # noqa: E402

# ---------------- Plan Bolum 5: 10 soru (dogru, yanlis) ----------------
SANITY = [
    ("What is the capital of France?", "Paris", "Berlin"),
    ("What is the chemical symbol for water?", "H2O", "CO2"),
    ("How many planets are in our solar system?", "8", "12"),
    ("Who wrote Romeo and Juliet?", "William Shakespeare", "Charles Dickens"),
    ("What is the largest ocean on Earth?", "Pacific Ocean", "Atlantic Ocean"),
    ("At what temperature in Celsius does water boil at sea level?", "100", "50"),
    ("Who was the first person to walk on the Moon?", "Neil Armstrong", "Yuri Gagarin"),
    ("What is the currency of Japan?", "Yen", "Won"),
    ("How many legs does a spider have?", "8", "6"),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci", "Pablo Picasso"),
]

# ---------------- Plandaki run_sanity (R2'ninki gelince onunla degistir) ----------------
def run_sanity(confidence_fn, model, tok):
    """confidence_fn(model, tok, question, answer) -> guven, 0..1"""
    conf, label = [], []
    for q, right, wrong in SANITY:
        conf += [confidence_fn(model, tok, q, right), confidence_fn(model, tok, q, wrong)]
        label += [1, 0]
    conf, label = np.array(conf), np.array(label)
    return {"auroc": roc_auc_score(label, conf),
            "gap": conf[label == 1].mean() - conf[label == 0].mean(),
            "n_distinct": len(np.unique(np.round(conf, 3)))}

LOG = []   # her ornegin detaylari buraya yazilir (ham cevaplar, failed vs.)

# ---------------- Yontemler (kod: r3_methods.py) ----------------
def empirical_p_true(model, tok, q, a, n=10):
    p, texts, unparsed = r3.empirical_p_true(model, tok, q, a, n)
    LOG.append({"q": q, "a": a, "conf": p, "unparsed_of_10": unparsed, "raw": texts})
    return p


def make_verbalized(scale):
    def verbalized(model, tok, q, a):
        conf, failed, reply = r3.verbalized_2s(model, tok, q, a, scale)
        LOG.append({"q": q, "a": a, "conf": conf, "failed": failed, "raw": reply})
        return conf

    return verbalized


verbalized_2s = make_verbalized(100)
verbalized_2s_10 = make_verbalized(10)


# ---------------- Calistir: 2 yontem x 3 model ----------------
MODELS = {"smollm2-135m": "HuggingFaceTB/SmolLM2-135M-Instruct",
          "qwen0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
          "qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct"}
ALL_METHODS = {"p_true_empirical": empirical_p_true,
               "verbalized_2s": verbalized_2s,        # 0-100 skala
               "verbalized_2s_10": verbalized_2s_10}  # 0-10 skala (plan: "hep ayni sayi" ise dene)

ap = argparse.ArgumentParser()
ap.add_argument("--methods", nargs="+", default=list(ALL_METHODS), choices=list(ALL_METHODS))
ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
args = ap.parse_args()
METHODS = {m: ALL_METHODS[m] for m in args.methods}

# Sonuclar her zaman experiments/llm-probly/results/sanity/ altina (nereden calistirirsan calistir)
OUT = Path(__file__).resolve().parents[1] / "results" / "sanity"
OUT.mkdir(parents=True, exist_ok=True)
device = "mps" if torch.backends.mps.is_available() else "cpu"

table = []
for short in args.models:
    name = MODELS[short]
    model, tok = load_model(name)
    model.to(device)
    for method, fn in METHODS.items():
        LOG.clear()
        torch.manual_seed(42)
        t0 = time.perf_counter()
        res = run_sanity(fn, model, tok)
        sec = (time.perf_counter() - t0) / 20
        if method.startswith("verbalized"):
            res["failed"] = sum(x["failed"] for x in LOG)
        else:
            res["unparsed_share"] = sum(x["unparsed_of_10"] for x in LOG) / 200
        res = {"method": method, "model": name, **{k: float(v) for k, v in res.items()},
               "seconds_per_example": sec, "examples": list(LOG)}
        with open(OUT / f"{method}_{short}.json", "w") as f:
            json.dump(res, f, indent=2)
            f.write("\n")
        table.append(res)
        print(f"{method:<18} {short:<13} AUROC={res['auroc']:.2f}  gap={res['gap']:+.2f}  "
              f"n_distinct={res['n_distinct']:.0f}  {sec:.2f} s/ornek")
    del model, tok
    if device == "mps":
        torch.mps.empty_cache()

print("\n| Yontem | Model | AUROC | Gap | n_distinct | Failed / parse edilemeyen | s/ornek |")
print("|---|---|---|---|---|---|---|")
for r in table:
    extra = f"{r['failed']:.0f}/20" if "failed" in r else f"%{100*r['unparsed_share']:.0f}"
    print(f"| {r['method']} | {r['model'].split('/')[-1]} | {r['auroc']:.2f} | {r['gap']:+.2f} "
          f"| {r['n_distinct']:.0f} | {extra} | {r['seconds_per_example']:.2f} |")
