# experiments/rl_uncertainty/scripts/train.py
"""Training script for RL uncertainty experiments.

Usage:
    python -m experiments.rl_uncertainty.scripts.train \
        --env continuous_nav --method ensemble --K 5 \
        --risk vanilla --seed 42 --steps 50000 \
        --out results/rl_uncertainty/
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from experiments.rl_uncertainty.agents.dqn import DQNAgent
from experiments.rl_uncertainty.envs.continuous_nav import ContinuousNavEnv
from experiments.rl_uncertainty.envs.racetrack import RacetrackEnv


def make_env(name: str) -> ContinuousNavEnv | RacetrackEnv:
    if name == "continuous_nav":
        return ContinuousNavEnv()
    if name == "racetrack":
        return RacetrackEnv()
    msg = f"Unknown env: {name}"
    raise ValueError(msg)


def train_single_agent(
    env_name: str,
    seed: int,
    n_steps: int,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_steps: int = 30_000,
    batch_size: int = 64,
    train_freq: int = 4,
) -> tuple[DQNAgent, list[dict]]:
    """Train a single DQN agent. Returns (agent, training_log)."""
    env = make_env(env_name)
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
            log.append({
                "step": step,
                "episode": episode_idx,
                "reward": episode_reward,
                "length": episode_len,
                "event": result.info.get("event", ""),
                "epsilon": epsilon,
            })
            episode_idx += 1
            episode_reward = 0.0
            episode_len = 0
            state = env.reset(seed=seed + episode_idx)
        else:
            state = result.next_state

    return agent, log


def train_ensemble(
    env_name: str,
    K: int,
    base_seed: int,
    n_steps: int,
) -> tuple[list[DQNAgent], list[list[dict]]]:
    """Train K independent DQN agents. Returns (agents, logs)."""
    agents = []
    all_logs = []
    for i in range(K):
        print(f"  Training ensemble member {i + 1}/{K} ...")
        agent, log = train_single_agent(env_name, seed=base_seed * 1000 + i, n_steps=n_steps)
        agents.append(agent)
        all_logs.append(log)
    return agents, all_logs


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RL agents for uncertainty demo")
    parser.add_argument("--env", choices=["continuous_nav", "racetrack"], required=True)
    parser.add_argument("--method", choices=["ensemble", "credal"], default="ensemble")
    parser.add_argument("--K", type=int, default=5, help="Ensemble size")
    parser.add_argument("--risk", choices=["vanilla", "lcb", "penalized"], default="vanilla")
    parser.add_argument("--beta", type=float, default=1.0, help="LCB beta")
    parser.add_argument("--lambda_", type=float, default=0.5, help="Penalty lambda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--out", type=str, default="results/rl_uncertainty")
    args = parser.parse_args()

    out_dir = Path(args.out) / args.env / args.method / args.risk / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training: env={args.env} method={args.method} risk={args.risk} seed={args.seed}")
    t0 = time.time()

    if args.method == "ensemble":
        agents, logs = train_ensemble(args.env, K=args.K, base_seed=args.seed, n_steps=args.steps)
        for i, agent in enumerate(agents):
            agent.save(str(out_dir / f"agent_{i}.pt"))
    elif args.method == "credal":
        agent, log = train_single_agent(args.env, seed=args.seed, n_steps=args.steps)
        agent.save(str(out_dir / "agent.pt"))
        logs = [log]

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s")

    meta = {"env": args.env, "method": args.method, "risk": args.risk, "seed": args.seed,
            "steps": args.steps, "K": args.K, "elapsed_s": elapsed}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    for i, log in enumerate(logs):
        log_path = out_dir / f"train_log_{i}.json"
        log_path.write_text(json.dumps(log, indent=2))

    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
