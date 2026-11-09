import re

import numpy as np
import torch
from probly.representer.sampler.huggingface import HFTextGenerationSampler, load_model

# LM-Polygraph'taki p(True) prompt formati
PROMPT = (
    "Question: {q}\n"
    "Possible answer: {a}\n"
    "Is the possible answer:\n"
    " (A) True\n"
    " (B) False\n"
    "The possible answer is:"
)

def is_true_loose(t):
    # Plandaki kural: "true", "(a)" veya "a" ile basliyorsa True say
    return t.strip().lower().startswith(("true", "(a)", "a"))

def is_true_strict(t):
    # Daha siki kural: "true" ile basliyor ya da tek basina "A" / "(A)" / "A)"
    t = t.strip().lower()
    return t.startswith("true") or re.match(r"^\(?a\)?(\W|$)", t) is not None

def empirical_p_true(model, tok, question, answer, n=10):
    prompt = PROMPT.format(q=question, a=answer)
    s = HFTextGenerationSampler(model, tok, num_samples=n, temperature=1.0,
                                max_new_tokens=3, with_log_likelihood=False)
    texts = [str(t) for t in s([prompt]).tensor.text[0]]
    p_loose = np.mean([is_true_loose(t) for t in texts])
    p_strict = np.mean([is_true_strict(t) for t in texts])
    return p_loose, p_strict, texts

TESTS = [
    ("What is the capital of France?", "Paris", "Berlin"),
    ("Who wrote Romeo and Juliet?", "William Shakespeare", "Charles Dickens"),
    ("What is the currency of Japan?", "Yen", "Won"),
]

device = "mps" if torch.backends.mps.is_available() else "cpu"

for name in ["Qwen/Qwen2.5-1.5B-Instruct"]:
    print(f"\n==================== {name} ====================")
    model, tok = load_model(name)
    model.to(device)
    torch.manual_seed(42)
    for q, right, wrong in TESTS:
        for label, a in [("DOGRU ", right), ("YANLIS", wrong)]:
            pl, ps, texts = empirical_p_true(model, tok, q, a)
            print(f"\n[{label}] {q} -> {a}")
            print(f"   p_loose={pl:.1f}  p_strict={ps:.1f}")
            print(f"   ham cevaplar: {texts}")
