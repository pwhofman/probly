# experiments/rl_uncertainty/scripts/plot_figure.py
"""Generate triptych figure from evaluation results.

Usage:
    python -m experiments.rl_uncertainty.scripts.plot_figure \
        --eval-dir results/rl_uncertainty/continuous_nav/ensemble/vanilla/seed42/eval \
        --env continuous_nav \
        --output figures/triptych.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.rl_uncertainty.envs.continuous_nav import ContinuousNavEnv
from experiments.rl_uncertainty.viz.triptych import make_triptych


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate triptych figure")
    parser.add_argument("--eval-dir", type=str, required=True)
    parser.add_argument("--env", choices=["continuous_nav", "racetrack"], required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--layout", type=str, default="default", help="Named env layout (default, gauntlet)")
    parser.add_argument("--train-log-dir", type=str, default=None)
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)

    # Default output: save triptych into the eval directory
    output_path = Path(args.output) if args.output else eval_dir / "triptych.png"

    if args.env == "continuous_nav":
        env = ContinuousNavEnv(layout=args.layout)
        obstacles = list(env.obstacles)
        goal = env.goal
        goal_radius = env.goal_radius
        start = env.start
    else:
        obstacles = None
        goal = None
        goal_radius = 0.05
        start = None

    make_triptych(
        eval_dir=eval_dir,
        obstacles=obstacles,
        goal=goal,
        goal_radius=goal_radius,
        start=start,
        train_log_dir=Path(args.train_log_dir) if args.train_log_dir else None,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
