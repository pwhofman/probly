import gc
import time

import torch
from probly.representer.sampler.huggingface import HFTextGenerationSampler, load_model

MODELS = [
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
]
PROMPT = "Answer the following question with as few words as possible.\nQuestion: What is the capital of France?"
DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])

rows = []
for name in MODELS:
    print(f"\n===== {name} =====")
    t0 = time.perf_counter()
    model, tok = load_model(name)
    load_s = time.perf_counter() - t0
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    mem_gb = model.get_memory_footprint() / 1e9
    print(f"Parametre: {params_m:.0f}M | Bellek: {mem_gb:.2f} GB | Yukleme: {load_s:.1f} s")

    times = {}
    for dev in DEVICES:
        try:
            model.to(dev)
            torch.manual_seed(42)
            sampler = HFTextGenerationSampler(model, tok, num_samples=10, temperature=0.7, max_new_tokens=32)
            sampler([PROMPT])  # isinma turu (ilk calistirma hep yavas olur, saymiyoruz)
            t = time.perf_counter()
            out = sampler([PROMPT])
            times[dev] = time.perf_counter() - t
            print(f"[{dev}] 10 cevap: {times[dev]:.1f} s")
            print("   Ilk 3 cevap:", [str(a).strip() for a in out.tensor.text[0][:3]])
        except Exception as e:
            times[dev] = None
            print(f"[{dev}] HATA: {type(e).__name__}: {e}")

    rows.append((name, params_m, mem_gb, load_s, times))
    del model, tok
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

# Gruba atilacak tablo
fmt = lambda x: f"{x:.1f} s" if x is not None else "hata"
print("\n\n| Model | Parametre | Bellek (fp32) | Yukleme | 10 cevap CPU | 10 cevap MPS |")
print("|---|---|---|---|---|---|")
for name, p, m, l, t in rows:
    print(f"| {name} | {p:.0f}M | {m:.2f} GB | {l:.1f} s | {fmt(t.get('cpu'))} | {fmt(t.get('mps'))} |")
