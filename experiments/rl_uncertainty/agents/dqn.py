"""Simple DQN agent for 2D RL experiments.

Minimal implementation: MLP Q-network, replay buffer, target network.
Designed to train on CPU in minutes for toy environments.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn


class QNetwork(nn.Module):
    """Small MLP Q-network."""

    def __init__(self, state_dim: int, n_actions: int, hidden: int = 64) -> None:
        """Initialize Q-network layers."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.net(x)


@dataclass
class Transition:
    """Single environment transition for replay buffer."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


@dataclass
class DQNAgent:
    """Simple DQN with replay buffer and target network.

    Args:
        state_dim: Dimensionality of state vector.
        n_actions: Number of discrete actions.
        hidden: Hidden layer size.
        lr: Learning rate.
        gamma: Discount factor.
        buffer_size: Replay buffer capacity.
        target_update_freq: Steps between target network syncs.
        seed: Random seed.
    """

    state_dim: int
    n_actions: int
    hidden: int = 64
    lr: float = 1e-3
    gamma: float = 0.99
    buffer_size: int = 50_000
    target_update_freq: int = 100
    seed: int = 0

    _q_net: QNetwork = field(init=False, repr=False)
    _target_net: QNetwork = field(init=False, repr=False)
    _optimizer: torch.optim.Adam = field(init=False, repr=False)
    _buffer: deque[Transition] = field(init=False, repr=False)
    _step_count: int = field(init=False, repr=False, default=0)
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize networks, optimizer, and replay buffer."""
        torch.manual_seed(self.seed)
        self._rng = np.random.default_rng(self.seed)
        self._q_net = QNetwork(self.state_dim, self.n_actions, self.hidden)
        self._target_net = QNetwork(self.state_dim, self.n_actions, self.hidden)
        self._target_net.load_state_dict(self._q_net.state_dict())
        self._optimizer = torch.optim.Adam(self._q_net.parameters(), lr=self.lr)
        self._buffer = deque(maxlen=self.buffer_size)

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """Epsilon-greedy action selection."""
        if self._rng.random() < epsilon:
            return int(self._rng.integers(self.n_actions))
        q = self.q_values(state)
        return int(np.argmax(q))

    def q_values(self, state: np.ndarray) -> np.ndarray:
        """Q-values for a single state. Returns shape (n_actions,)."""
        self._q_net.eval()
        with torch.no_grad():
            x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q = self._q_net(x).squeeze(0).numpy()
        return q

    def batch_q_values(self, states: np.ndarray) -> np.ndarray:
        """Q-values for a batch of states. Returns shape (batch, n_actions)."""
        self._q_net.eval()
        with torch.no_grad():
            x = torch.tensor(states, dtype=torch.float32)
            q = self._q_net(x).numpy()
        return q

    def store(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """Add transition to replay buffer."""
        self._buffer.append(Transition(state, action, reward, next_state, done))

    def train_step(self, batch_size: int = 64) -> float | None:
        """Sample batch and do one gradient step. Returns loss or None if buffer too small."""
        if len(self._buffer) < batch_size:
            return None

        indices = self._rng.integers(len(self._buffer), size=batch_size)
        batch = [self._buffer[i] for i in indices]

        states = torch.tensor(np.array([t.state for t in batch]), dtype=torch.float32)
        actions = torch.tensor([t.action for t in batch], dtype=torch.long)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32)
        next_states = torch.tensor(np.array([t.next_state for t in batch]), dtype=torch.float32)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32)

        self._q_net.train()
        q_pred = self._q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_next = self._target_net(next_states).max(dim=1).values
            q_target = rewards + self.gamma * q_next * (1.0 - dones)

        loss = nn.functional.mse_loss(q_pred, q_target)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        self._step_count += 1
        if self._step_count % self.target_update_freq == 0:
            self._target_net.load_state_dict(self._q_net.state_dict())

        return float(loss.detach())

    def get_network(self) -> QNetwork:
        """Return the Q-network (for probly wrapping)."""
        return self._q_net

    def save(self, path: str) -> None:
        """Save Q-network weights to file."""
        torch.save(self._q_net.state_dict(), path)

    def load(self, path: str) -> None:
        """Load Q-network weights from file."""
        self._q_net.load_state_dict(torch.load(path, weights_only=True))
        self._target_net.load_state_dict(self._q_net.state_dict())
