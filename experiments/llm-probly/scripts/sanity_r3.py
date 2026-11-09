import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from probly.representer.sampler.huggingface import HFTextGenerationSampler, load_model

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

# ---------------- Yontem 1: Empirical p(True) ----------------
PT_PROMPT = ("Question: {q}\nPossible answer: {a}\nIs the possible answer:\n"
             " (A) True\n (B) False\nThe possible answer is:")

def is_true(t):
    t = t.strip().lower()
    return t.startswith("true") or re.match(r"^\(?a\)?(\W|$)", t) is not None

def is_false(t):
    t = t.strip().lower()
    return t.startswith("false") or re.match(r"^\(?b\)?(\W|$)", t) is not None

def empirical_p_true(model, tok, q, a, n=10):
    s = HFTextGenerationSampler(model, tok, num_samples=n, temperature=1.0,
                                max_new_tokens=3, with_log_likelihood=False)
    texts = [str(t) for t in s([PT_PROMPT.format(q=q, a=a)]).tensor.text[0]]
    p = float(np.mean([is_true(t) for t in texts]))
    unparsed = sum(not is_true(t) and not is_false(t) for t in texts)  # ne True ne False
    LOG.append({"q": q, "a": a, "conf": p, "unparsed_of_10": unparsed, "raw": texts})
    return p

# ---------------- Yontem 2: Verbalized 2S (0-100 ve 0-10 skala) ----------------
ASK_100 = "How confident are you that your answer is correct? Reply with only a number between 0 and 100."
ASK_10 = "How confident are you that your answer is correct? Reply with only a number between 0 and 10."

def parse_confidence(reply, scale):
    """Cevaptaki ilk sayiyi 0..1 araligina cevirir. Sayi yoksa None.
    "0.99" gibi ondalikli ve <= 1 olan sayilar zaten oran kabul edilir
    (eskiden 0.99 / 100 = 0.0099 oluyordu)."""
    m = re.search(r"(\d+(?:\.\d+)?)", reply)
    if m is None:
        return None
    x = float(m.group(1))
    if "." in m.group(1) and x <= 1.0:
        return x
    return min(max(x, 0.0), scale) / scale

def make_verbalized(ask, scale):
    def verbalized(model, tok, q, a):
        chat = [{"role": "user", "content": q},
                {"role": "assistant", "content": a},
                {"role": "user", "content": ask}]
        sv = HFTextGenerationSampler(model, tok, num_samples=1, do_sample=False,
                                     max_new_tokens=8, with_log_likelihood=False)
        reply = str(sv([chat]).tensor.text[0, 0])
        c = parse_confidence(reply, scale)
        failed = c is None
        conf = 0.0 if failed else c   # contract: parse edilemezse guven 0 -> uncertainty 1.0
        LOG.append({"q": q, "a": a, "conf": conf, "failed": failed, "raw": reply})
        return conf
    return verbalized

verbalized_2s = make_verbalized(ASK_100, 100.0)
verbalized_2s_10 = make_verbalized(ASK_10, 10.0)

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
