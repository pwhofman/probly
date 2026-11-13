"""Generate the shared answer cache (cache v2) for any model and dataset (DATA).

For every question we store one greedy answer (the answer that all methods
judge) and 10 sampled answers, with their log likelihoods. Cache v2 also
stores, for the greedy answer, the log probability and the entropy of every
generated token, so the MSP / perplexity / mean token entropy baselines need
no extra model run.

Run from ``experiments/llm-probly``::

    uv run python -m scripts.generate --model qwen0.5b --n 5
    uv run python -m scripts.generate --model qwen0.5b --n 200
    caffeinate -i uv run python -m scripts.generate --model qwen3b --n 200 --device mps

Rows are appended to ``data/<model>/<dataset>_<n>.partial.jsonl`` after every
question, so an interrupted run continues where it stopped (run the same
command again). At the end the file is renamed to ``<dataset>_<n>.jsonl``.
The generation settings are written to ``<dataset>_<n>.meta.json``; a resumed
run refuses to continue if the settings differ.

The same seed gives the same questions in the same order, and answers are
seeded per question, so the first 50 rows of a 200-question file equal the
50-question file made with the same model, precision and seed.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import platform
import time
from typing import TYPE_CHECKING

import torch
import transformers
from transformers import set_seed

from common.data import DATASETS, DEFAULT_SEED, cache_path, load_jsonl, load_questions
from common.generation import generate_answers, stop_at_newline_config
from probly.representer.sampler.huggingface import load_model

if TYPE_CHECKING:
    from pathlib import Path

MODELS = {
    "smollm2": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "qwen0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
}
PRECISIONS = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

INSTRUCTION = "Answer the following question with as few words as possible."
NUM_SAMPLES = 10
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 32
CACHE_VERSION = 2


def build_prompt(question: str) -> str:
    """Return the user message sent to the model (the chat template is applied in generate_answers)."""
    return f"{INSTRUCTION}\nQuestion: {question}"


def parse_args() -> argparse.Namespace:
    """Read the command line options."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODELS), default="qwen0.5b")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="triviaqa")
    parser.add_argument("--n", type=int, default=5, help="number of questions")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="cpu", help="cpu or mps (Apple GPU)")
    parser.add_argument("--precision", choices=sorted(PRECISIONS), default="float32")
    return parser.parse_args()


def settings(args: argparse.Namespace) -> dict:
    """The settings that define a cache file; a resumed run must use the same ones."""
    return {
        "cache_version": CACHE_VERSION,
        "model": MODELS[args.model],
        "dataset": args.dataset,
        "n": args.n,
        "seed": args.seed,
        "precision": args.precision,
        "instruction": INSTRUCTION,
        "num_samples": NUM_SAMPLES,
        "temperature": TEMPERATURE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "stop": "first newline or EOS; only tokens before the stop are scored",
    }


def check_resume(meta_path: Path, wanted: dict) -> None:
    """Stop if an unfinished run with different settings would be continued."""
    if not meta_path.exists():
        return
    old = json.loads(meta_path.read_text())["settings"]
    changed = {k: (old.get(k), v) for k, v in wanted.items() if old.get(k) != v}
    if changed:
        msg = f"{meta_path} was written with other settings {changed}; delete the .partial.jsonl to start over."
        raise SystemExit(msg)


def write_meta(meta_path: Path, run_settings: dict, extra: dict) -> None:
    """Write the meta file next to the cache."""
    meta = {"settings": run_settings, **extra}
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Generate greedy answers and samples and write them as JSONL."""
    args = parse_args()
    model_name = MODELS[args.model]
    out_path = cache_path(args.model, args.n, args.dataset)
    partial_path = cache_path(args.model, args.n, args.dataset, suffix=".partial.jsonl")
    meta_path = cache_path(args.model, args.n, args.dataset, suffix=".meta.json")
    partial_path.parent.mkdir(parents=True, exist_ok=True)

    run_settings = settings(args)
    if partial_path.exists():
        check_resume(meta_path, run_settings)
    done = {row["id"] for row in load_jsonl(partial_path)} if partial_path.exists() else set()

    questions = load_questions(args.dataset, args.n, seed=args.seed)
    print(f"{model_name} on {args.dataset}: {len(questions)} questions, {len(done)} already done", flush=True)

    model, tok = load_model(model_name, model_kwargs={"dtype": PRECISIONS[args.precision]})
    model = model.to(args.device)
    model.eval()
    gen_config = stop_at_newline_config(model, tok)

    environment = {
        "device": args.device,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "machine": platform.platform(),
        "model_generation_config": {
            k: getattr(gen_config, k, None) for k in ("top_p", "top_k", "repetition_penalty", "do_sample")
        },
    }
    started = datetime.now(UTC).isoformat(timespec="seconds")
    write_meta(meta_path, run_settings, {"environment": environment, "started": started, "finished": None})

    start = time.time()
    n_new = 0
    for q in questions:
        if q["id"] in done:
            continue
        set_seed(args.seed + q["id"])  # reproducible samples per question
        prompt = build_prompt(q["question"])
        common = {"max_new_tokens": MAX_NEW_TOKENS, "generation_config": gen_config}
        greedy = generate_answers(model, tok, prompt, num_return_sequences=1, do_sample=False, **common)[0]
        samples = generate_answers(
            model, tok, prompt, num_return_sequences=NUM_SAMPLES, do_sample=True, temperature=TEMPERATURE, **common
        )

        row = {
            "id": q["id"],
            "dataset": args.dataset,
            "question": q["question"],
            "aliases": q["aliases"],
            "greedy_answer": greedy["text"],
            "greedy_ll_sum": greedy["ll_sum"],
            "greedy_ll_mean": greedy["ll_mean"],
            "greedy_token_logprobs": greedy["token_logprobs"],
            "greedy_token_entropies": greedy["token_entropies"],
            "samples": [s["text"] for s in samples],
            "samples_ll_mean": [s["ll_mean"] for s in samples],
            "samples_ll_sum": [s["ll_sum"] for s in samples],
            "model": model_name,
        }
        with partial_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n_new += 1
        print(f"[{q['id'] + 1}/{args.n}] {q['question'][:60]!r} -> {row['greedy_answer']!r}", flush=True)

    elapsed = time.time() - start
    n_rows = len(load_jsonl(partial_path))
    partial_path.replace(out_path)
    write_meta(
        meta_path,
        run_settings,
        {
            "environment": environment,
            "started": started,
            "finished": datetime.now(UTC).isoformat(timespec="seconds"),
            "seconds_per_question_last_run": round(elapsed / n_new, 2) if n_new else None,
        },
    )
    print(f"Done: {n_rows} rows -> {out_path} ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
