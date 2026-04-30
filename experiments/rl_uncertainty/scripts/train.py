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


def track_uncertainty_during_training(
    agents: list[DQNAgent],
    env_name: str,
    probe_states: np.ndarray,
    step: int,
) -> dict:
    """Evaluate uncertainty on fixed probe states during training."""
    from experiments.rl_uncertainty.uncertainty.ensemble import EnsembleEstimator
    estimator = EnsembleEstimator(agents)
    result = estimator.estimate(probe_states)
    return {
        "step": step,
        "mean_epistemic": float(result.epistemic.mean()),
        "mean_aleatoric": float(result.aleatoric.mean()),
        "mean_total": float(result.total.mean()),
    }


def train_ensemble(
    env_name: str,
    K: int,
    base_seed: int,
    n_steps: int,
) -> tuple[list[DQNAgent], list[list[dict]], list[dict]]:
    """Train K independent DQN agents. Returns (agents, logs, uncertainty_log)."""
    env_template = make_env(env_name)
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
    logs: list[list[dict]] = [[] for _ in range(K)]
    episode_rewards = [0.0] * K
    episode_lens = [0] * K
    episode_idxs = [0] * K

    for i in range(K):
        seed = base_seed * 1000 + i
        agent = DQNAgent(state_dim=env_template.state_dim, n_actions=env_template.n_actions, seed=seed)
        env = make_env(env_name)
        state = env.reset(seed=seed)
        agents.append(agent)
        envs.append(env)
        states.append(state)

    uncertainty_log: list[dict] = []
    eu_track_freq = 1000

    print(f"  Training {K} ensemble members in lockstep for {n_steps} steps ...")
    for step in range(n_steps):
        frac = min(step / max(30_000, 1), 1.0)
        epsilon = 1.0 + (0.05 - 1.0) * frac

        for i in range(K):
            action = agents[i].select_action(states[i], epsilon=epsilon)
            result = envs[i].step(action)
            agents[i].store(states[i], action, result.reward, result.next_state, result.done)
            episode_rewards[i] += result.reward
            episode_lens[i] += 1

            if step % 4 == 0:
                agents[i].train_step(batch_size=64)

            if result.done:
                seed_i = base_seed * 1000 + i
                logs[i].append({
                    "step": step,
                    "episode": episode_idxs[i],
                    "reward": episode_rewards[i],
                    "length": episode_lens[i],
                    "event": result.info.get("event", ""),
                    "epsilon": epsilon,
                })
                episode_idxs[i] += 1
                episode_rewards[i] = 0.0
                episode_lens[i] = 0
                states[i] = envs[i].reset(seed=seed_i + episode_idxs[i])
            else:
                states[i] = result.next_state

        if step % eu_track_freq == 0 and step > 0:
            entry = track_uncertainty_during_training(agents, env_name, probe_states, step)
            uncertainty_log.append(entry)
            if step % 10000 == 0:
                print(f"  Step {step}/{n_steps} -- EU: {entry['mean_epistemic']:.4f}")

    return agents, logs, uncertainty_log


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

    uncertainty_log: list[dict] = []
    if args.method == "ensemble":
        agents, logs, uncertainty_log = train_ensemble(args.env, K=args.K, base_seed=args.seed, n_steps=args.steps)
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

    if uncertainty_log:
        (out_dir / "uncertainty_log.json").write_text(json.dumps(uncertainty_log, indent=2))

    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
