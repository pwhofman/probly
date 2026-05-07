"""Run the GPN uncertainty demo."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from gpn_uq.demo import DEFAULT_AMAZON_LAYOUT_PATH, run_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GPN variants on synthetic and Amazon Photos node-classification graphs.")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--experiment", choices=("synthetic", "amazon-photo", "all"), default="synthetic")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--nodes-per-class", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--amazon-epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--forceatlas2-iterations", type=int, default=50)
    parser.add_argument("--amazon-layout", choices=("forceatlas2", "precomputed"), default="precomputed")
    parser.add_argument("--amazon-layout-path", type=Path, default=DEFAULT_AMAZON_LAYOUT_PATH)
    parser.add_argument(
        "--figure-dpi",
        type=int,
        default=200,
        help="DPI for rasterized layers embedded in generated PDF figures.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device for model training and inference, e.g. 'cpu' or 'cuda'.")
    parser.add_argument("--retrain", action="store_true", help="Retrain models even when compatible checkpoints exist.")
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help="Skip training and only use existing compatible checkpoints; missing models are plotted as gray placeholders.",
    )
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    run_demo(
        args.output_dir,
        seed=args.seed,
        nodes_per_class=args.nodes_per_class,
        epochs=args.epochs,
        lr=args.lr,
        experiment=args.experiment,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        cache_dir=args.cache_dir,
        amazon_epochs=args.amazon_epochs,
        forceatlas2_iterations=args.forceatlas2_iterations,
        amazon_layout=args.amazon_layout,
        amazon_layout_path=args.amazon_layout_path,
        figure_dpi=args.figure_dpi,
        device=args.device,
        retrain=args.retrain,
        inference_only=args.inference_only,
    )
    print(f"Wrote demo outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
