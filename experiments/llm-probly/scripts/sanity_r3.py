"""R3 sanity testi: empirical p(True) ve verbalized 2S, 3 model.

Grubun ortak sanity kodunu kullanir (R2: common/sanity.py -> run_sanity, save_sanity),
boylece R1, R2 ve R3'un sonuclari ayni sorular, ayni AUROC/gap/n_distinct hesabi
ve ayni dosya adi bicimiyle kaydedilir: results/sanity/<method>_<HF model adi>.json

Calistirma (repo kokunden):
    uv run python experiments/llm-probly/scripts/sanity_r3.py
    uv run python experiments/llm-probly/scripts/sanity_r3.py --methods verbalized_2s --models qwen0.5b
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from probly.representer.sampler.huggingface import load_model

ROOT = Path(__file__).resolve().parents[1]  # experiments/llm-probly
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import r3_methods as r3  # noqa: E402
from common.sanity import run_sanity, save_sanity  # noqa: E402

OUT = ROOT / "results" / "sanity"
MODELS = {
    "smollm2-135m": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "qwen0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
}
METHODS = ["p_true_empirical", "verbalized_2s", "verbalized_2s_10"]


def make_confidence_fn(method, model, tok, log):
    """R2'nin run_sanity'si confidence_fn(question, answer, samples) bekliyor.
    model ve tok'u iceride tutan bir fonksiyon dondururuz; ham cevaplar log'a yazilir."""

    def empirical(q, a, samples=None):
        p, texts, unparsed = r3.empirical_p_true(model, tok, q, a)
        log.append({"q": q, "a": a, "conf": p, "unparsed_of_10": unparsed, "raw": texts})
        return p

    def verbalized(scale):
        def fn(q, a, samples=None):
            conf, failed, reply = r3.verbalized_2s(model, tok, q, a, scale)
            log.append({"q": q, "a": a, "conf": conf, "failed": failed, "raw": reply})
            return conf

        return fn

    return {
        "p_true_empirical": empirical,
        "verbalized_2s": verbalized(100),
        "verbalized_2s_10": verbalized(10),
    }[method]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    args = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    for short in args.models:
        name = MODELS[short]
        model, tok = load_model(name)
        model.to(device)
        for method in args.methods:
            log = []
            torch.manual_seed(42)
            t0 = time.perf_counter()
            res = run_sanity(make_confidence_fn(method, model, tok, log))
            res["seconds_per_example"] = (time.perf_counter() - t0) / 20
            if method.startswith("verbalized"):
                res["failed"] = sum(x["failed"] for x in log)
            else:
                res["unparsed_share"] = sum(x["unparsed_of_10"] for x in log) / 200
            res["examples"] = log
            path = save_sanity(res, method, name, folder=OUT)
            path.write_text(path.read_text() + "\n")  # pre-commit "end of file" kontrolu icin
            print(f"{method:<18} {short:<13} AUROC={res['auroc']:.2f}  gap={res['gap']:+.2f}  "
                  f"n_distinct={res['n_distinct']}  {res['seconds_per_example']:.2f} s/ornek")
        del model, tok
        if device == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
