"""Shared p(True) for reflexive methods (R1 zero-shot, R2 brainstorm).

Same prompt for both methods; the only difference is the brainstorm line.
Two modes:
- "chat_groups" (R1): chat template, next token compared over token groups
  True/A/(A vs False/B/(B (only strings that are single tokens).
- "plain_ab" (R2): no chat template, prompt ends with " (", next token A vs B.
Raw baseline: true_logit() (logit of the single token "True", as LM-Polygraph does).
"""

import torch

DEFAULT_MODE = "plain_ab"   # chosen by the sanity comparison (notebooks/reflexive_mode_comparison.ipynb)

GROUPS = {
    "chat_groups": (["True", " True", "A", " A", "(A"],
                    ["False", " False", "B", " B", "(B"]),
    "plain_ab": (["A"], ["B"]),
}


def build_text(question, answer, brainstorm=None):
    ideas = (f"Here are some ideas that were brainstormed: {', '.join(brainstorm)}\n"
             if brainstorm else "")
    return (f"Question: {question}\n"
            f"{ideas}"
            f"Possible answer: {answer}\n"
            "Is the possible answer:\n"
            " (A) True\n"
            " (B) False\n"
            "The possible answer is:")


def build_prompt(tok, question, answer, brainstorm=None, mode=DEFAULT_MODE):
    text = build_text(question, answer, brainstorm)
    if mode == "chat_groups":
        return tok.apply_chat_template([{"role": "user", "content": text}],
                                       tokenize=False, add_generation_prompt=True)
    if mode == "plain_ab":
        return text + " ("
    raise ValueError(f"unknown mode: {mode}")


def single_token_ids(tok, strings):
    """Token ids of the strings that are exactly one token for this tokenizer."""
    ids = set()
    for s in strings:
        enc = tok.encode(s, add_special_tokens=False)
        if len(enc) == 1:
            ids.add(enc[0])
    return sorted(ids)


@torch.no_grad()
def next_token_logits(model, tok, question, answer, brainstorm=None, mode=DEFAULT_MODE):
    text = build_prompt(tok, question, answer, brainstorm, mode)
    # The chat template already contains its special tokens, so none are added again.
    enc = tok(text, return_tensors="pt",
              add_special_tokens=(mode != "chat_groups")).to(model.device)
    return model(**enc).logits[0, -1].float()


def p_true_details(model, tok, question, answer, brainstorm=None, mode=DEFAULT_MODE):
    """Normalized p(True) plus coverage (probability mass on both groups)."""
    logp = torch.log_softmax(next_token_logits(model, tok, question, answer, brainstorm, mode), -1)
    true_ids, false_ids = (single_token_ids(tok, g) for g in GROUPS[mode])
    if not true_ids or not false_ids:
        raise ValueError(f"empty token group for mode {mode!r} "
                         f"(true={true_ids}, false={false_ids}); check the tokenizer")
    lt = torch.logsumexp(logp[true_ids], dim=0)
    lf = torch.logsumexp(logp[false_ids], dim=0)
    return {
        "p_true": torch.sigmoid(lt - lf).item(),      # = P(T) / (P(T) + P(F))
        "coverage": (lt.exp() + lf.exp()).item(),
        "mode": mode,
    }


def p_true(model, tok, question, answer, brainstorm=None, mode=DEFAULT_MODE):
    """Confidence in [0, 1]. Uncertainty for the score contract = 1 - p_true."""
    return p_true_details(model, tok, question, answer, brainstorm, mode)["p_true"]


def true_logit(model, tok, question, answer, brainstorm=None):
    """Raw baseline: logit of the single token 'True' (LM-Polygraph), chat-template prompt.
    High = more confident. For the score contract use uncertainty = -true_logit."""
    logits = next_token_logits(model, tok, question, answer, brainstorm, mode="chat_groups")
    true_id = tok.encode("True", add_special_tokens=False)[0]
    return logits[true_id].item()