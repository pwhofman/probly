"""Interactive streaming chat REPL for the locally cached Gemma 4 model."""

from __future__ import annotations

import argparse

from core import CACHE_DIR, suppress_hf_noise
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

from gemma import MODEL_ID

EXIT_WORDS = {"exit", "quit", "/exit", "/quit", ":q"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--system",
        default=None,
        help="Optional system message prepended to the conversation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate per reply.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. 0 disables sampling (greedy decoding).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Nucleus sampling threshold (e.g. 0.9). Requires temperature > 0.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Keep only the K most likely tokens (e.g. 40). 0 disables.",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=None,
        help="Min-p sampling threshold (e.g. 0.05).",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Penalty for repeated tokens (>1.0 discourages, e.g. 1.1).",
    )
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=None,
        help="Forbid repeating any n-gram of this size.",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=None,
        help="Beam search width. >1 enables beam search (deterministic).",
    )
    parser.add_argument(
        "--penalty-alpha",
        type=float,
        default=None,
        help="Contrastive search alpha (pair with small --top-k, e.g. 0.6 + 4).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling.",
    )
    return parser.parse_args()


def build_gen_kwargs(args: argparse.Namespace) -> dict:
    """Collect only the generation kwargs the user actually set."""
    do_sample = args.temperature > 0.0
    kwargs: dict = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        kwargs["temperature"] = args.temperature
    for name in (
        "top_p",
        "top_k",
        "min_p",
        "repetition_penalty",
        "no_repeat_ngram_size",
        "num_beams",
        "penalty_alpha",
    ):
        value = getattr(args, name)
        if value is not None:
            kwargs[name] = value
    return kwargs


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    suppress_hf_noise()

    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = build_gen_kwargs(args)

    messages: list[dict[str, str]] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    print("Gemma 4 chat ready. Type a message and press Enter.")
    print("Commands: 'exit'/'quit' to leave, '/reset' to clear history.")
    print(f"Generation settings: {gen_kwargs}\n")

    while True:
        try:
            user = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in EXIT_WORDS:
            break
        if user.lower() == "/reset":
            messages = [{"role": "system", "content": args.system}] if args.system else []
            print("[history cleared]\n")
            continue

        messages.append({"role": "user", "content": user})
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)

        print("gemma> ", end="", flush=True)
        outputs = model.generate(
            **inputs,
            **gen_kwargs,
            streamer=streamer,
        )
        prompt_len = inputs["input_ids"].shape[-1]
        reply = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()
        messages.append({"role": "assistant", "content": reply})
        print()


if __name__ == "__main__":
    main()
