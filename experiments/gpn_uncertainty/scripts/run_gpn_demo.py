"""Run the GPN uncertainty demo."""

from __future__ import annotations

import argparse
from pathlib import Path

from gpn_uq.demo import run_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GPN variants on a small synthetic node-classification graph.")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--nodes-per-class", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()
    run_demo(args.output_dir, seed=args.seed, nodes_per_class=args.nodes_per_class, epochs=args.epochs, lr=args.lr)
    print(f"Wrote demo outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
