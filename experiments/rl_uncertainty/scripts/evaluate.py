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

import numpy as np

from experiments.rl_uncertainty.agents.dqn import DQNAgent
from experiments.rl_uncertainty.envs.continuous_nav import ContinuousNavEnv
from experiments.rl_uncertainty.envs.racetrack import RacetrackEnv
from experiments.rl_uncertainty.risk.lcb import lcb_action
from experiments.rl_uncertainty.uncertainty.ensemble import EnsembleEstimator
from experiments.rl_uncertainty.uncertainty.interface import UncertaintyEstimator


def make_env(name: str) -> ContinuousNavEnv | RacetrackEnv:
    if name == "continuous_nav":
        return ContinuousNavEnv()
    if name == "racetrack":
        return RacetrackEnv()
    msg = f"Unknown env: {name}"
    raise ValueError(msg)


def load_ensemble(checkpoint_dir: Path, env_name: str, K: int) -> list[DQNAgent]:
    env = make_env(env_name)
    agents = []
    for i in range(K):
        agent = DQNAgent(state_dim=env.state_dim, n_actions=env.n_actions, seed=i)
        agent.load(str(checkpoint_dir / f"agent_{i}.pt"))
        agents.append(agent)
    return agents


def rollout(
    env_name: str,
    estimator: UncertaintyEstimator,
    beta: float,
    seed: int,
) -> dict:
    """Single rollout. Returns trajectory dict."""
    env = make_env(env_name)
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
) -> dict:
    """Rollout using a single agent's greedy policy (no ensemble)."""
    env = make_env(env_name)
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
) -> dict:
    """Compute epistemic uncertainty heatmap over state space.

    For racetrack (4D state), fixes velocity to zero and grids over (x,y).
    """
    env = make_env(env_name)
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
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    out_dir = ckpt / "eval"
    out_dir.mkdir(exist_ok=True)

    print(f"Loading {args.method} from {ckpt} ...")

    if args.method == "ensemble":
        agents = load_ensemble(ckpt, args.env, args.K)
        estimator = EnsembleEstimator(agents)
    elif args.method == "credal":
        from experiments.rl_uncertainty.uncertainty.credal import CredalEstimator
        env = make_env(args.env)
        agent = DQNAgent(state_dim=env.state_dim, n_actions=env.n_actions, seed=0)
        agent.load(str(ckpt / "agent.pt"))
        estimator = CredalEstimator(agent, num_members=args.K)

    # Rollouts with risk-neutral (beta=0) and risk-averse (beta>0)
    for label, beta in [("risky", args.beta_risky), ("safe", args.beta_safe)]:
        print(f"  Rolling out {args.n_rollouts} episodes ({label}, beta={beta}) ...")
        trajectories = [rollout(args.env, estimator, beta=beta, seed=i) for i in range(args.n_rollouts)]
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
            traj = rollout_single_agent(args.env, agent, seed=0)
            traj["member"] = i
            member_trajs.append(traj)
            event = traj["event"]
            print(f"    Member {i}: len={traj['length']}, event={event}")
        (out_dir / "trajectories_members.json").write_text(json.dumps(member_trajs, indent=2))

    # EU heatmap grid
    print(f"  Computing EU grid ({args.grid_resolution}x{args.grid_resolution}) ...")
    grid = compute_eu_grid(args.env, estimator, resolution=args.grid_resolution)
    (out_dir / "eu_grid.json").write_text(json.dumps(grid))

    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
