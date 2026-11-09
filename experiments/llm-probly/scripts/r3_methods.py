"""R3'un logit gerektirmeyen refleksif yontemleri (sanity testi ve 50 soru ortak kullanir).

- empirical p(True): modele 10 kez "(A) True / (B) False" sordurup True oranini sayar
- verbalized 2S: cevabi verdikten sonra "ne kadar eminsin?" diye sorar, sayiyi okur
"""

import re

import numpy as np
from probly.representer.sampler.huggingface import HFTextGenerationSampler

# ---------------- Empirical p(True) ----------------
PT_PROMPT = (
    "Question: {q}\nPossible answer: {a}\nIs the possible answer:\n"
    " (A) True\n (B) False\nThe possible answer is:"
)


def is_true(t):
    t = t.strip().lower()
    return t.startswith("true") or re.match(r"^\(?a\)?(\W|$)", t) is not None


def is_false(t):
    t = t.strip().lower()
    return t.startswith("false") or re.match(r"^\(?b\)?(\W|$)", t) is not None


def empirical_p_true(model, tok, q, a, n=10):
    """Return (p_true, ham cevaplar, ne True ne False olan cevap sayisi)."""
    s = HFTextGenerationSampler(
        model, tok, num_samples=n, temperature=1.0, max_new_tokens=3, with_log_likelihood=False
    )
    texts = [str(t) for t in s([PT_PROMPT.format(q=q, a=a)]).tensor.text[0]]
    p = float(np.mean([is_true(t) for t in texts]))
    unparsed = sum(not is_true(t) and not is_false(t) for t in texts)
    return p, texts, unparsed


# ---------------- Verbalized 2S ----------------
ASK = {
    100: "How confident are you that your answer is correct? Reply with only a number between 0 and 100.",
    10: "How confident are you that your answer is correct? Reply with only a number between 0 and 10.",
}


def parse_confidence(reply, scale):
    """Cevaptaki ilk sayiyi 0..1 araligina cevirir. Sayi yoksa None.
    "0.99" gibi ondalikli ve <= 1 olan sayilar zaten oran kabul edilir."""
    m = re.search(r"(\d+(?:\.\d+)?)", reply)
    if m is None:
        return None
    x = float(m.group(1))
    if "." in m.group(1) and x <= 1.0:
        return x
    return min(max(x, 0.0), scale) / scale


def verbalized_2s(model, tok, q, a, scale=100):
    """Return (guven 0..1, failed, ham cevap). Parse edilemezse guven 0 (contract: uncertainty 1.0)."""
    chat = [
        {"role": "user", "content": q},
        {"role": "assistant", "content": a},
        {"role": "user", "content": ASK[scale]},
    ]
    sv = HFTextGenerationSampler(
        model, tok, num_samples=1, do_sample=False, max_new_tokens=8, with_log_likelihood=False
    )
    reply = str(sv([chat]).tensor.text[0, 0])
    c = parse_confidence(reply, scale)
    return (0.0, True, reply) if c is None else (c, False, reply)
