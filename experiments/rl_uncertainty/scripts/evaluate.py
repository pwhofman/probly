# experiments/rl_uncertainty/scripts/evaluate.py
"""Evaluation script: rollout trajectories + compute EU heatmap grid.

Usage:
    python -m experiments.rl_uncertainty.scripts.evaluate \
        --checkpoint results/rl_uncertainty/continuous_nav/ensemble/vanilla/seed42/ \
        --method ensemble --K 5 --env continuous_nav \
        --n-rollouts 20 --grid-resolution 50 \
        --beta 0.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from experiments.rl_uncertainty.agents.dqn import DQNAgent
from experiments.rl_uncertainty.envs import make_env
from experiments.rl_uncertainty.risk.lcb import lcb_action
from experiments.rl_uncertainty.uncertainty.ensemble import EnsembleEstimator

if TYPE_CHECKING:
    from experiments.rl_uncertainty.uncertainty.interface import UncertaintyEstimator


def load_ensemble(
    checkpoint_dir: Path,
    env_name: str,
    k: int,
    hidden: int = 64,
    env_kwargs: dict[str, Any] | None = None,
) -> list[DQNAgent]:
    """Load K trained DQN agents from checkpoint directory."""
    env = make_env(env_name, **(env_kwargs or {}))
    agents = []
    for i in range(k):
        agent = DQNAgent(state_dim=env.state_dim, n_actions=env.n_actions, seed=i, hidden=hidden)
        agent.load(str(checkpoint_dir / f"agent_{i}.pt"))
        agents.append(agent)
    return agents


def rollout(
    env_name: str,
    estimator: UncertaintyEstimator,
    beta: float,
    seed: int,
    env_kwargs: dict[str, Any] | None = None,
) -> dict:
    """Single rollout. Returns trajectory dict."""
    env = make_env(env_name, **(env_kwargs or {}))
    state = env.reset(seed=seed)
    states = [state.copy()]
    actions = []
    rewards = []

    while True:
        action = lcb_action(state, estimator, beta=beta)
        result = env.step(action)
        states.append(result.next_state.copy())
        actions.append(action)
        rewards.append(result.reward)
        if result.done:
            return {
                "states": np.array(states).tolist(),
                "actions": actions,
                "rewards": rewards,
                "total_reward": sum(rewards),
                "event": result.info.get("event", ""),
                "length": len(actions),
                "seed": seed,
            }
        state = result.next_state


def rollout_single_agent(
    env_name: str,
    agent: DQNAgent,
    seed: int,
    env_kwargs: dict[str, Any] | None = None,
) -> dict:
    """Rollout using a single agent's greedy policy (no ensemble)."""
    env = make_env(env_name, **(env_kwargs or {}))
    state = env.reset(seed=seed)
    states = [state.copy()]
    actions = []
    rewards = []

    while True:
        action = agent.select_action(state, epsilon=0.0)
        result = env.step(action)
        states.append(result.next_state.copy())
        actions.append(action)
        rewards.append(result.reward)
        if result.done:
            return {
                "states": np.array(states).tolist(),
                "actions": actions,
                "rewards": rewards,
                "total_reward": sum(rewards),
                "event": result.info.get("event", ""),
                "length": len(actions),
                "seed": seed,
            }
        state = result.next_state


def compute_eu_grid(
    env_name: str,
    estimator: UncertaintyEstimator,
    resolution: int = 50,
    env_kwargs: dict[str, Any] | None = None,
) -> dict:
    """Compute epistemic uncertainty heatmap over state space.

    For racetrack (4D state), fixes velocity to zero and grids over (x,y).
    """
    env = make_env(env_name, **(env_kwargs or {}))
    low, high = env.bounds
    xs = np.linspace(low[0], high[0], resolution)
    ys = np.linspace(low[1], high[1], resolution)
    xx, yy = np.meshgrid(xs, ys)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    if env.state_dim > 2:
        padding = np.zeros((grid_points.shape[0], env.state_dim - 2))
        grid_points = np.concatenate([grid_points, padding], axis=1)

    grid_points = grid_points.astype(np.float32)
    result = estimator.estimate(grid_points)

    return {
        "xs": xs.tolist(),
        "ys": ys.tolist(),
        "epistemic": result.epistemic.reshape(resolution, resolution).tolist(),
        "aleatoric": result.aleatoric.reshape(resolution, resolution).tolist(),
        "total": result.total.reshape(resolution, resolution).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained RL agents")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--method", choices=["ensemble", "credal"], default="ensemble")
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--env", choices=["continuous_nav", "racetrack"], required=True)
    parser.add_argument("--n-rollouts", type=int, default=20)
    parser.add_argument("--grid-resolution", type=int, default=50)
    parser.add_argument("--beta", type=float, default=0.0, help="LCB beta (0=greedy)")
    parser.add_argument("--beta-risky", type=float, default=0.0, help="Beta for risk-neutral baseline")
    parser.add_argument("--beta-safe", type=float, default=2.0, help="Beta for risk-averse agent")
    parser.add_argument("--hidden", type=int, default=64, help="Q-network hidden size (must match training)")
    parser.add_argument("--layout", type=str, default="default", help="Named env layout (default, gauntlet)")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    out_dir = ckpt / "eval"
    out_dir.mkdir(exist_ok=True)

    env_kwargs: dict[str, Any] = {}
    if args.layout != "default":
        env_kwargs["layout"] = args.layout

    print(f"Loading {args.method} from {ckpt} ...")

    if args.method == "ensemble":
        agents = load_ensemble(ckpt, args.env, args.K, hidden=args.hidden, env_kwargs=env_kwargs)
        estimator = EnsembleEstimator(agents)
    elif args.method == "credal":
        from experiments.rl_uncertainty.uncertainty.credal import CredalEstimator  # noqa: PLC0415

        env = make_env(args.env, **env_kwargs)
        agent = DQNAgent(state_dim=env.state_dim, n_actions=env.n_actions, seed=0)
        agent.load(str(ckpt / "agent.pt"))
        estimator = CredalEstimator(agent, num_members=args.K)

    # Rollouts with risk-neutral (beta=0) and risk-averse (beta>0)
    for label, beta in [("risky", args.beta_risky), ("safe", args.beta_safe)]:
        print(f"  Rolling out {args.n_rollouts} episodes ({label}, beta={beta}) ...")
        trajectories = [
            rollout(args.env, estimator, beta=beta, seed=i, env_kwargs=env_kwargs) for i in range(args.n_rollouts)
        ]
        events = [t["event"] for t in trajectories]
        n_crashes = events.count("collision") + events.count("wall")
        n_goals = events.count("goal") + events.count("finish")
        print(f"    Crashes: {n_crashes}/{len(trajectories)}, Goals: {n_goals}/{len(trajectories)}")
        (out_dir / f"trajectories_{label}.json").write_text(json.dumps(trajectories, indent=2))

    # Per-member ensemble trajectories (spaghetti plot)
    if args.method == "ensemble":
        print(f"  Rolling out per-member trajectories ({args.K} members) ...")
        member_trajs = []
        for i, agent in enumerate(agents):
            traj = rollout_single_agent(args.env, agent, seed=0, env_kwargs=env_kwargs)
            traj["member"] = i
            member_trajs.append(traj)
            event = traj["event"]
            print(f"    Member {i}: len={traj['length']}, event={event}")
        (out_dir / "trajectories_members.json").write_text(json.dumps(member_trajs, indent=2))

    # EU heatmap grid
    print(f"  Computing EU grid ({args.grid_resolution}x{args.grid_resolution}) ...")
    grid = compute_eu_grid(args.env, estimator, resolution=args.grid_resolution, env_kwargs=env_kwargs)
    (out_dir / "eu_grid.json").write_text(json.dumps(grid))

    # Save gallery figures alongside JSON
    _save_eval_figures(args.env, out_dir, args.method, env_kwargs=env_kwargs)

    print(f"Saved to {out_dir}")


def _save_eval_figures(
    env_name: str,
    eval_dir: Path,
    method: str,
    env_kwargs: dict[str, Any] | None = None,
) -> None:
    """Save standalone PNG figures from evaluation results."""
    from experiments.rl_uncertainty.viz.gallery import save_eu_snapshot  # noqa: PLC0415

    # Heatmap
    grid_data = json.loads((eval_dir / "eu_grid.json").read_text())
    from experiments.rl_uncertainty.envs.continuous_nav import ContinuousNavEnv  # noqa: PLC0415

    obstacles: list[tuple[np.ndarray, float]] | None = None
    env = make_env(env_name, **(env_kwargs or {}))
    if isinstance(env, ContinuousNavEnv):
        obstacles = list(env.obstacles)

    save_eu_snapshot(
        np.array(grid_data["epistemic"]),
        np.array(grid_data["xs"]),
        np.array(grid_data["ys"]),
        eval_dir / "heatmap.png",
        obstacles=obstacles,
        title="Epistemic Uncertainty (eval)",
    )

    # Spaghetti plot (ensemble only)
    member_path = eval_dir / "trajectories_members.json"
    if member_path.exists() and method == "ensemble":
        import matplotlib.pyplot as plt  # noqa: PLC0415

        from experiments.rl_uncertainty.viz.trajectories import plot_member_trajectories  # noqa: PLC0415

        member_trajs = json.loads(member_path.read_text())
        goal = getattr(env, "goal", None)
        goal_radius = getattr(env, "goal_radius", 0.05)
        start = getattr(env, "start", None)

        fig, ax = plt.subplots(1, 1, figsize=(5, 5), constrained_layout=True)
        plot_member_trajectories(
            ax,
            member_trajs,
            obstacles=obstacles,
            goal=goal,
            goal_radius=goal_radius,
            start=start,
            title="Ensemble Member Trajectories",
        )
        fig.savefig(eval_dir / "spaghetti.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  [gallery] Saved eval figures to {eval_dir}")


if __name__ == "__main__":
    main()
