"""Gemma 4 chat model wrapper used by the FastAPI backend.

Mirrors the inference loop from ``experiments/gemma/interaction.py``: load the
tokenizer and model once from a local HuggingFace cache, then apply the chat
template and run greedy generation per request.
"""

from __future__ import annotations

import os

# Force offline mode *before* importing transformers / huggingface_hub so no
# code path tries to contact the hub (transformers 5.x still does a model_info
# lookup from the tokenizer init path even when local_files_only=True).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if TYPE_CHECKING:
    from collections.abc import Iterator

MODEL_ID = "google/gemma-4-E2B-it"
# web/backend/app/gemma.py -> repo root is three parents up.
CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "model_cache"
MAX_NEW_TOKENS = 512


def _resolve_snapshot_dir() -> Path:
    """Return the on-disk snapshot directory for the cached model.

    Passing a local path (rather than the hub model id) to ``from_pretrained``
    keeps transformers on its "is_local" branch and avoids any remaining hub
    lookups triggered deep inside tokenizer initialization.
    """
    model_dir = CACHE_DIR / f"models--{MODEL_ID.replace('/', '--')}"
    snapshots = model_dir / "snapshots"
    if not snapshots.is_dir():
        msg = f"No snapshots directory under {model_dir}. Populate the cache via experiments/gemma/download.py."
        raise FileNotFoundError(msg)
    candidates = [p for p in snapshots.iterdir() if p.is_dir()]
    if not candidates:
        msg = f"No snapshot revisions found in {snapshots}."
        raise FileNotFoundError(msg)
    # Prefer the snapshot pointed to by refs/main if present, otherwise take
    # the only (or newest) one.
    ref_main = model_dir / "refs" / "main"
    if ref_main.is_file():
        revision = ref_main.read_text().strip()
        preferred = snapshots / revision
        if preferred.is_dir():
            return preferred
    return max(candidates, key=lambda p: p.stat().st_mtime)


class GemmaChat:
    """Load Gemma 4 once and serve chat completions."""

    def __init__(self) -> None:
        """Load the tokenizer and model from the local HF cache snapshot."""
        if not CACHE_DIR.exists():
            msg = (
                f"Gemma model cache not found at {CACHE_DIR}. "
                "Populate it by copying experiments/gemma/model_cache or "
                "running experiments/gemma/download.py."
            )
            raise FileNotFoundError(msg)
        snapshot_dir = _resolve_snapshot_dir()
        # transformers' auto-class return types are unions that include None
        # and backend-specific objects ty can't narrow. Cast to Any so the
        # downstream duck-typed calls (apply_chat_template / generate / decode)
        # type-check cleanly — the experiments REPL validates actual behavior.
        self.tokenizer = cast("Any", AutoTokenizer.from_pretrained(snapshot_dir, local_files_only=True))
        self.model = cast(
            "Any",
            AutoModelForCausalLM.from_pretrained(snapshot_dir, local_files_only=True),
        )

    def _prepare_inputs(self, messages: list[dict[str, str]]) -> Any:  # noqa: ANN401
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)

    def reply(self, messages: list[dict[str, str]]) -> str:
        """Generate a single assistant reply for the given chat history."""
        inputs = self._prepare_inputs(messages)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
        prompt_len = inputs["input_ids"].shape[-1]
        decoded: str = self.tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
        return decoded.strip()

    def reply_stream(
        self,
        messages: list[dict[str, str]],
    ) -> Iterator[tuple[str, float]]:
        """Yield ``(text, confidence)`` pairs as the model produces each token.

        Runs a manual greedy decoding loop instead of ``model.generate`` so
        we can read the top-1 softmax probability at every step and use it
        as a per-token confidence score. KV caching via ``past_key_values``
        keeps this roughly comparable in speed to the previous
        ``TextIteratorStreamer`` path — we just also get the logits.

        Under ``do_sample=False`` (greedy), the reported confidence is the
        probability the model assigned to the token it actually picked,
        which is a cheap but sensible "how sure was it here?" signal for
        the frontend's highlighting UI.
        """
        inputs = self._prepare_inputs(messages)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        device = input_ids.device
        eos_id = self.tokenizer.eos_token_id
        past: Any = None
        with torch.no_grad():
            for _ in range(MAX_NEW_TOKENS):
                step_ids = input_ids if past is None else input_ids[:, -1:]
                outputs = self.model(
                    input_ids=step_ids,
                    attention_mask=attention_mask,
                    past_key_values=past,
                    use_cache=True,
                )
                past = outputs.past_key_values
                logits = outputs.logits[:, -1, :]
                probs = torch.softmax(logits, dim=-1)
                next_id = int(torch.argmax(probs, dim=-1).item())
                confidence = float(probs[0, next_id].item())
                if eos_id is not None and next_id == eos_id:
                    break
                text = self.tokenizer.decode([next_id], skip_special_tokens=True)
                input_ids = torch.cat(
                    [input_ids, torch.tensor([[next_id]], device=device)],
                    dim=-1,
                )
                if attention_mask is not None:
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones((1, 1), device=device, dtype=attention_mask.dtype)],
                        dim=-1,
                    )
                if text:
                    yield text, confidence
