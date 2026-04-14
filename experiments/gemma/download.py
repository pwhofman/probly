"""Download the smallest Gemma 4 model into the shared data cache."""

from __future__ import annotations

from core import CACHE_DIR
from transformers import AutoModelForCausalLM, AutoTokenizer

from gemma import MODEL_ID


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Starting download of {MODEL_ID} into {CACHE_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    print(f"Downloaded {MODEL_ID} to {CACHE_DIR}")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Model parameters: {model.num_parameters():,}")


if __name__ == "__main__":
    main()
