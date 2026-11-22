"""One command line entry point for the LLM-probly benchmark (E2).

Run from ``experiments/llm-probly``::

    uv run python -m cli run --config configs/qwen0.5b_triviaqa_200.yaml   # whole chain for one config
    uv run python -m cli methods                                            # the method registry
    uv run python -m cli methods --access text_only --max-cost 11           # what an API-only user can run
    uv run python -m cli generate --model qwen0.5b --n 1000                 # one step on its own
    uv run python -m cli label --model qwen0.5b --n 1000
    uv run python -m cli compare --dataset triviaqa --n 200                 # methods x models table
    uv run python -m cli plot --dataset triviaqa --n 200
    uv run python -m cli check-prefix --model qwen0.5b --small 200 --large 1000
    uv run python -m cli diff --run qwen0.5b/triviaqa_1000 likelihood_mean msp   # is A better than B?

``uv run python -m cli <command> --help`` shows the options of each command.
The step commands forward their options unchanged to the scripts in
``scripts/``, so they behave exactly like ``python -m scripts.<name>``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys

from common.registry import ACCESS_LEVELS, FAMILIES, candidates

# command -> module whose main() reads sys.argv; imported only when used, because
# generate and label import torch and transformers.
STEP_COMMANDS = {
    "generate": "scripts.generate",
    "label": "scripts.label",
    "compare": "scripts.compare_models",
    "plot": "scripts.plot_results",
    "check-prefix": "scripts.check_prefix",
    "diff": "scripts.diff_methods",
}


def methods_command(args: argparse.Namespace) -> None:
    """Print the registered methods, cheapest first, optionally filtered."""
    specs = candidates(access=args.access, max_cost=args.max_cost, family=args.family)
    if args.json:
        print(json.dumps([spec.to_dict() for spec in specs], indent=2))
        return
    header = f"{'method':<28}{'family':<11}{'access':<11}{'cost':>6}  {'model':<6}{'nli':<5}note"
    print(header)
    print("-" * len(header))
    for spec in specs:
        model = "yes" if spec.needs_model else ""
        nli = "yes" if spec.uses_nli else ""
        print(f"{spec.name:<28}{spec.family:<11}{spec.access:<11}{spec.cost:>6g}  {model:<6}{nli:<5}{spec.note}")


def build_parser() -> argparse.ArgumentParser:
    """Parser with one subcommand per action."""
    from scripts import run_benchmark  # noqa: PLC0415 - keeps `cli methods` free of pandas/yaml

    parser = argparse.ArgumentParser(
        prog="python -m cli", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="generate -> label -> methods -> evaluate -> table for one config")
    run_benchmark.add_arguments(run)
    run.set_defaults(handler=run_benchmark.run)

    methods = sub.add_parser("methods", help="list the method registry (family, access, cost)")
    methods.add_argument("--access", default="white_box", choices=ACCESS_LEVELS, help="model access you have")
    methods.add_argument("--max-cost", type=float, default=float("inf"), help="generation passes per question")
    methods.add_argument("--family", choices=FAMILIES)
    methods.add_argument("--json", action="store_true", help="print JSON instead of a table")
    methods.set_defaults(handler=methods_command)

    for name, module in STEP_COMMANDS.items():
        sub.add_parser(name, help=f"same as python -m {module}", add_help=False)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Dispatch to the chosen command."""
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in STEP_COMMANDS:
        module = importlib.import_module(STEP_COMMANDS[argv[0]])
        sys.argv = [f"python -m cli {argv[0]}", *argv[1:]]  # the step scripts parse sys.argv
        module.main()
        return
    args = build_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
