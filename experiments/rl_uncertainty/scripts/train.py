# experiments/rl_uncertainty/scripts/train.py
"""Training script for RL uncertainty experiments.

Usage:
    python -m experiments.rl_uncertainty.scripts.train \
        --env continuous_nav --method ensemble --K 5 \
        --seed 42 --steps 200000 --tag v6_ensemble_k5

Results land in experiments/rl_uncertainty/results/<tag>/ with intermediate
snapshots (training curve, EU heatmaps) saved during training.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import json
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from experiments.rl_uncertainty.agents.dqn import DQNAgent
from experiments.rl_uncertainty.envs import make_env

if TYPE_CHECKING:
    from experiments.rl_uncertainty.uncertainty.ensemble import EnsembleEstimator

# Callback type: (step, agents, logs, uncertainty_log, probe_eu_grid) -> None
_SnapshotCallback = Callable[
    [int, list[DQNAgent], list[list[dict]], list[dict], np.ndarray | None],
    None,
]


def train_single_agent(
    env_name: str,
    seed: int,
    n_steps: int,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_steps: int = 30_000,
    batch_size: int = 64,
    train_freq: int = 4,
    env_kwargs: dict[str, Any] | None = None,
) -> tuple[DQNAgent, list[dict]]:
    """Train a single DQN agent. Returns (agent, training_log)."""
    env = make_env(env_name, **(env_kwargs or {}))
    agent = DQNAgent(state_dim=env.state_dim, n_actions=env.n_actions, seed=seed)

    log: list[dict] = []
    state = env.reset(seed=seed)
    episode_reward = 0.0
    episode_len = 0
    episode_idx = 0

    for step in range(n_steps):
        frac = min(step / max(epsilon_decay_steps, 1), 1.0)
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * frac

        action = agent.select_action(state, epsilon=epsilon)
        result = env.step(action)
        agent.store(state, action, result.reward, result.next_state, result.done)

        episode_reward += result.reward
        episode_len += 1

        if step % train_freq == 0:
            agent.train_step(batch_size=batch_size)

        if result.done:
            log.append(
                {
                    "step": step,
                    "episode": episode_idx,
                    "reward": episode_reward,
                    "length": episode_len,
                    "event": result.info.get("event", ""),
                    "epsilon": epsilon,
                }
            )
            episode_idx += 1
            episode_reward = 0.0
            episode_len = 0
            state = env.reset(seed=seed + episode_idx)
        else:
            state = result.next_state

    return agent, log


def track_uncertainty_during_training(
    agents: list[DQNAgent],
    probe_states: np.ndarray,
    step: int,
) -> dict:
    """Evaluate uncertainty on fixed probe states during training."""
    from experiments.rl_uncertainty.uncertainty.ensemble import EnsembleEstimator  # noqa: PLC0415

    estimator = EnsembleEstimator(agents)
    result = estimator.estimate(probe_states)
    return {
        "step": step,
        "mean_epistemic": float(result.epistemic.mean()),
        "mean_aleatoric": float(result.aleatoric.mean()),
        "mean_total": float(result.total.mean()),
    }


def train_ensemble(  # noqa: PLR0915
    env_name: str,
    k: int,
    base_seed: int,
    n_steps: int,
    snapshot_callback: _SnapshotCallback | None = None,
    hidden: int = 64,
    env_kwargs: dict[str, Any] | None = None,
) -> tuple[list[DQNAgent], list[list[dict]], list[dict]]:
    """Train k independent DQN agents. Returns (agents, logs, uncertainty_log)."""
    _env_kwargs = env_kwargs or {}
    env_template = make_env(env_name, **_env_kwargs)
    low, high = env_template.bounds
    n_per_dim = 10
    xs = np.linspace(low[0], high[0], n_per_dim)
    ys = np.linspace(low[1], high[1], n_per_dim)
    xx, yy = np.meshgrid(xs, ys)
    probe_states = np.column_stack([xx.ravel(), yy.ravel()])
    if env_template.state_dim > 2:
        padding = np.zeros((probe_states.shape[0], env_template.state_dim - 2))
        probe_states = np.concatenate([probe_states, padding], axis=1)
    probe_states = probe_states.astype(np.float32)

    agents = []
    envs = []
    states = []
    logs: list[list[dict]] = [[] for _ in range(k)]
    episode_rewards = [0.0] * k
    episode_lens = [0] * k
    episode_idxs = [0] * k

    for i in range(k):
        seed = base_seed * 1000 + i
        agent = DQNAgent(state_dim=env_template.state_dim, n_actions=env_template.n_actions, seed=seed, hidden=hidden)
        env = make_env(env_name, **_env_kwargs)
        state = env.reset(seed=seed)
        agents.append(agent)
        envs.append(env)
        states.append(state)

    uncertainty_log: list[dict] = []
    eu_track_freq = 1000
    snapshot_freq = 50_000

    print(f"  Training {k} ensemble members in lockstep for {n_steps} steps ...")
    for step in range(n_steps):
        frac = min(step / max(30_000, 1), 1.0)
        epsilon = 1.0 + (0.05 - 1.0) * frac

        for i in range(k):
            action = agents[i].select_action(states[i], epsilon=epsilon)
            result = envs[i].step(action)
            agents[i].store(states[i], action, result.reward, result.next_state, result.done)
            episode_rewards[i] += result.reward
            episode_lens[i] += 1

            if step % 4 == 0:
                agents[i].train_step(batch_size=64)

            if result.done:
                seed_i = base_seed * 1000 + i
                logs[i].append(
                    {
                        "step": step,
                        "episode": episode_idxs[i],
                        "reward": episode_rewards[i],
                        "length": episode_lens[i],
                        "event": result.info.get("event", ""),
                        "epsilon": epsilon,
                    }
                )
                episode_idxs[i] += 1
                episode_rewards[i] = 0.0
                episode_lens[i] = 0
                states[i] = envs[i].reset(seed=seed_i + episode_idxs[i])
            else:
                states[i] = result.next_state

        if step % eu_track_freq == 0 and step > 0:
            entry = track_uncertainty_during_training(agents, probe_states, step)
            uncertainty_log.append(entry)
            if step % 10000 == 0:
                print(f"  Step {step}/{n_steps} -- EU: {entry['mean_epistemic']:.4f}")

        # Save gallery snapshots at regular intervals
        if snapshot_callback and step > 0 and step % snapshot_freq == 0:
            # Build a coarse EU grid for the snapshot
            estimator = _make_estimator(agents)
            eu_result = estimator.estimate(probe_states)
            eu_grid = eu_result.epistemic.reshape(n_per_dim, n_per_dim)
            snapshot_callback(step, agents, logs, uncertainty_log, eu_grid)

    # Final snapshot at end of training
    if snapshot_callback:
        estimator = _make_estimator(agents)
        eu_result = estimator.estimate(probe_states)
        eu_grid = eu_result.epistemic.reshape(n_per_dim, n_per_dim)
        snapshot_callback(n_steps, agents, logs, uncertainty_log, eu_grid)

    return agents, logs, uncertainty_log


def _make_estimator(agents: list[DQNAgent]) -> EnsembleEstimator:
    """Create an EnsembleEstimator from agents (avoids circular import at top level)."""
    from experiments.rl_uncertainty.uncertainty.ensemble import EnsembleEstimator  # noqa: PLC0415

    return EnsembleEstimator(agents)


def _build_snapshot_callback(
    results_dir: Path,
    obstacles: list[tuple[np.ndarray, float]] | None,
    xs: np.ndarray,
    ys: np.ndarray,
) -> _SnapshotCallback:
    """Build a callback that saves gallery snapshots during training."""
    from experiments.rl_uncertainty.viz.gallery import (  # noqa: PLC0415
        save_decomposition,
        save_eu_snapshot,
        save_training_curve,
    )

    snap_dir = results_dir / "eu_snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)

    def callback(
        step: int,
        _agents: list[DQNAgent],
        logs: list[list[dict]],
        uncertainty_log: list[dict],
        eu_grid: np.ndarray | None,
    ) -> None:
        label = f"{step // 1000:03d}k"
        print(f"  [gallery] Saving snapshot at step {step} ...")

        # EU heatmap snapshot
        if eu_grid is not None:
            save_eu_snapshot(
                eu_grid,
                xs,
                ys,
                snap_dir / f"step_{label}.png",
                obstacles=obstacles,
                title=f"EU at step {label}",
            )

        # Training curve (overwritten each time — always shows latest)
        save_training_curve(logs, results_dir / "training_curve.png")

        # Decomposition (overwritten each time)
        if uncertainty_log:
            save_decomposition(uncertainty_log, results_dir / "decomposition.png")

    return callback


def main() -> None:  # noqa: PLR0915
    parser = argparse.ArgumentParser(description="Train RL agents for uncertainty demo")
    parser.add_argument("--env", choices=["continuous_nav", "racetrack"], required=True)
    parser.add_argument("--method", choices=["ensemble", "credal"], default="ensemble")
    parser.add_argument("--K", type=int, default=5, help="Ensemble size")
    parser.add_argument("--risk", choices=["vanilla", "lcb", "penalized"], default="vanilla")
    parser.add_argument("--beta", type=float, default=1.0, help="LCB beta")
    parser.add_argument("--lambda_", type=float, default=0.5, help="Penalty lambda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--hidden", type=int, default=64, help="Q-network hidden size")
    parser.add_argument("--collision-reward", type=float, default=None, help="Override collision/wall reward")
    parser.add_argument("--layout", type=str, default="default", help="Named env layout (default, gauntlet)")
    parser.add_argument("--tag", type=str, default=None, help="Results gallery folder name")
    parser.add_argument("--out", type=str, default=None, help="Output dir (overrides --tag)")
    args = parser.parse_args()

    # Resolve output directory: --out overrides --tag
    if args.out:
        out_dir = Path(args.out) / args.env / args.method / args.risk / f"seed{args.seed}"
    elif args.tag:
        out_dir = Path(__file__).resolve().parent.parent / "results" / args.tag
    else:
        out_dir = Path("results/rl_uncertainty") / args.env / args.method / args.risk / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build env kwargs from CLI overrides
    env_kwargs: dict[str, Any] = {}
    if args.layout != "default":
        env_kwargs["layout"] = args.layout
    if args.collision_reward is not None:
        key = "wall_reward" if args.env == "racetrack" else "collision_reward"
        env_kwargs[key] = args.collision_reward

    print(f"Training: env={args.env} method={args.method} K={args.K} seed={args.seed} hidden={args.hidden}")
    if env_kwargs:
        print(f"  env overrides: {env_kwargs}")
    print(f"Results:  {out_dir}")
    t0 = time.time()

    # Save config upfront so we can see what's running
    config = {
        "env": args.env,
        "method": args.method,
        "risk": args.risk,
        "seed": args.seed,
        "steps": args.steps,
        "K": args.K,
        "hidden": args.hidden,
        "layout": args.layout,
        "env_kwargs": env_kwargs,
        "tag": args.tag,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    uncertainty_log: list[dict] = []
    if args.method == "ensemble":
        # Build snapshot callback for gallery
        env_template = make_env(args.env, **env_kwargs)
        low, high = env_template.bounds
        xs = np.linspace(low[0], high[0], 10)
        ys = np.linspace(low[1], high[1], 10)
        from experiments.rl_uncertainty.envs.continuous_nav import ContinuousNavEnv  # noqa: PLC0415

        obstacles = list(env_template.obstacles) if isinstance(env_template, ContinuousNavEnv) else None
        snapshot_cb = _build_snapshot_callback(out_dir, obstacles, xs, ys)

        agents, logs, uncertainty_log = train_ensemble(
            args.env,
            k=args.K,
            base_seed=args.seed,
            n_steps=args.steps,
            snapshot_callback=snapshot_cb,
            hidden=args.hidden,
            env_kwargs=env_kwargs,
        )
        for i, agent in enumerate(agents):
            agent.save(str(out_dir / f"agent_{i}.pt"))
    elif args.method == "credal":
        agent, log = train_single_agent(
            args.env,
            seed=args.seed,
            n_steps=args.steps,
            env_kwargs=env_kwargs,
        )
        agent.save(str(out_dir / "agent.pt"))
        logs = [log]

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s")

    # Update config with elapsed time
    config["elapsed_s"] = elapsed
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    for i, log in enumerate(logs):
        log_path = out_dir / f"train_log_{i}.json"
        log_path.write_text(json.dumps(log, indent=2))

    if uncertainty_log:
        (out_dir / "uncertainty_log.json").write_text(json.dumps(uncertainty_log, indent=2))

    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
