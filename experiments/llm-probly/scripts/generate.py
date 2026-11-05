"""Generate the shared answer data for the LLM-probly experiments (B1).

For every TriviaQA question we store one greedy answer (the answer that all
methods judge) and 10 sampled answers (used by semantic entropy and the
brainstorm variant of p(True)), together with their log likelihoods.

Run from ``experiments/llm-probly``::

    uv run python -m scripts.generate --model smollm2 --n 5
    uv run python -m scripts.generate --model qwen0.5b --n 50

Rows are written to ``data/<model>/triviaqa_<n>.partial.jsonl`` as soon as
each question is done, so an interrupted run continues where it stopped.
At the end the file is renamed to ``triviaqa_<n>.jsonl``.
"""

from __future__ import annotations

import argparse
import copy
import json
import time

from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizerBase, set_seed

from common.data import DATA_DIR, DEFAULT_SEED, load_jsonl, load_trivia_questions
from probly.representer.sampler.huggingface import HFTextGenerationSampler, load_model

MODELS = {
    "smollm2": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "qwen0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
}

INSTRUCTION = "Answer the following question with as few words as possible."
NUM_SAMPLES = 10
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 32


def build_prompt(question: str) -> str:
    """Return the user message sent to the model (the chat template is applied by probly)."""
    return f"{INSTRUCTION}\nQuestion: {question}"


def stop_at_newline_config(model: PreTrainedModel, tok: PreTrainedTokenizerBase) -> GenerationConfig:
    """Copy the model's generation config and make every newline token a stop token.

    Small models often answer and then keep talking on the next line. Stopping at
    the first line break (as LM-Polygraph does with its newline stop string) keeps only the
    answer, and probly then scores only the tokens before the stop token, so the
    stored text and log likelihood describe the same answer.
    """
    config = copy.deepcopy(model.generation_config)
    eos = config.eos_token_id
    eos_ids = set(eos) if isinstance(eos, list) else ({eos} if eos is not None else set())
    vocab = tok.batch_decode([[i] for i in range(len(tok))])
    newline_ids = {i for i, piece in enumerate(vocab) if "\n" in piece}
    config.eos_token_id = sorted(eos_ids | newline_ids)
    return config


def parse_args() -> argparse.Namespace:
    """Read the command line options."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODELS), default="smollm2")
    parser.add_argument("--n", type=int, default=5, help="number of questions")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="cpu", help="cpu or mps (Apple GPU)")
    return parser.parse_args()


def main() -> None:
    """Generate greedy answers and samples and write them as JSONL."""
    args = parse_args()
    model_name = MODELS[args.model]
    out_path = DATA_DIR / args.model / f"triviaqa_{args.n}.jsonl"
    partial_path = out_path.with_suffix(".partial.jsonl")
    partial_path.parent.mkdir(parents=True, exist_ok=True)

    questions = load_trivia_questions(args.n, seed=args.seed)
    done = {row["id"] for row in load_jsonl(partial_path)} if partial_path.exists() else set()
    print(f"{model_name}: {len(questions)} questions, {len(done)} already done")

    model, tok = load_model(model_name)
    model = model.to(args.device)
    model.eval()
    gen_config = stop_at_newline_config(model, tok)

    # Greedy decoding is deterministic, so running it twice gives the summed and
    # the length-normalised log likelihood of exactly the same answer.
    greedy_kwargs = {
        "num_samples": 1,
        "do_sample": False,
        "max_new_tokens": MAX_NEW_TOKENS,
        "generation_config": gen_config,
    }
    greedy_sum = HFTextGenerationSampler(model, tok, length_normalization=False, **greedy_kwargs)
    greedy_mean = HFTextGenerationSampler(model, tok, length_normalization=True, **greedy_kwargs)
    sampler = HFTextGenerationSampler(
        model,
        tok,
        num_samples=NUM_SAMPLES,
        temperature=TEMPERATURE,
        max_new_tokens=MAX_NEW_TOKENS,
        length_normalization=True,
        generation_config=gen_config,
    )

    start = time.time()
    for q in questions:
        if q["id"] in done:
            continue
        set_seed(args.seed + q["id"])  # reproducible samples per question
        prompt = [build_prompt(q["question"])]

        g_sum = greedy_sum(prompt).tensor
        g_mean = greedy_mean(prompt).tensor
        s = sampler(prompt).tensor

        row = {
            "id": q["id"],
            "question": q["question"],
            "aliases": q["aliases"],
            "greedy_answer": str(g_sum.text[0, 0]).strip(),
            "greedy_ll_sum": float(g_sum.log_likelihood[0, 0]),
            "greedy_ll_mean": float(g_mean.log_likelihood[0, 0]),
            "samples": [str(t).strip() for t in s.text[0]],
            "samples_ll_mean": [float(x) for x in s.log_likelihood[0]],
            "model": model_name,
        }
        with partial_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[{q['id'] + 1}/{args.n}] {q['question'][:60]!r} -> {row['greedy_answer']!r}", flush=True)

    n_rows = len(load_jsonl(partial_path))
    partial_path.replace(out_path)
    print(f"Done: {n_rows} rows -> {out_path} ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
