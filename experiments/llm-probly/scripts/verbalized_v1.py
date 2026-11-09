import re

import numpy as np
import torch
from probly.representer.sampler.huggingface import HFTextGenerationSampler, load_model

ASK = "How confident are you that your answer is correct? Reply with only a number between 0 and 10."

def verbalized_2s(model, tok, question, answer):
    chat = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},   # 1. asama: cevap (biz veriyoruz)
        {"role": "user", "content": ASK},           # 2. asama: ne kadar eminsin?
    ]
    sv = HFTextGenerationSampler(model, tok, num_samples=1, do_sample=False,
                                 max_new_tokens=8, with_log_likelihood=False)
    reply = str(sv([chat]).tensor.text[0, 0])
    m = re.search(r"(\d+(?:\.\d+)?)", reply)
    if m is None:
        return 0.0, True, reply           # sayi yok -> guven 0, failed=True (contract)
    conf = min(max(float(m.group(1)), 0.0), 10.0) / 10.0   # 0..1 araligina cek
    return conf, False, reply

TESTS = [
    ("What is the capital of France?", "Paris", "Berlin"),
    ("Who wrote Romeo and Juliet?", "William Shakespeare", "Charles Dickens"),
    ("What is the currency of Japan?", "Yen", "Won"),
]

device = "mps" if torch.backends.mps.is_available() else "cpu"

for name in ["HuggingFaceTB/SmolLM2-135M-Instruct",
             "Qwen/Qwen2.5-0.5B-Instruct",
             "Qwen/Qwen2.5-1.5B-Instruct"]:
    print(f"\n==================== {name} ====================")
    model, tok = load_model(name)
    model.to(device)
    confs, fails = [], 0
    for q, right, wrong in TESTS:
        for label, a in [("DOGRU ", right), ("YANLIS", wrong)]:
            conf, failed, reply = verbalized_2s(model, tok, q, a)
            confs.append(conf)
            fails += failed
            print(f"[{label}] {a:<20} guven={conf:.2f}  failed={failed}  ham cevap: {reply!r}")
    print(f"--> farkli sayi sayisi (n_distinct): {len(np.unique(np.round(confs, 3)))} / 6   |   failed: {fails} / 6")
