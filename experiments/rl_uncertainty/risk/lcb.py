"""Lower Confidence Bound action selection.

At test time, selects actions that are both high-value and low-uncertainty:
    action = argmax(Q_mean - beta * Q_std)

beta=0 is vanilla (risk-neutral), beta>0 is cautious.
"""

from __future__ import annotations

import numpy as np

from experiments.rl_uncertainty.uncertainty.interface import UncertaintyEstimator


def lcb_action(
    state: np.ndarray,
    estimator: UncertaintyEstimator,
    beta: float,
) -> int:
    """Select action using Lower Confidence Bound.

    Args:
        state: Single state vector, shape (state_dim,).
        estimator: Uncertainty estimator providing Q mean/std.
        beta: Risk-aversion coefficient (0 = risk-neutral).

    Returns:
        Selected action index.
    """
    q_mean, q_std = estimator.q_with_uncertainty(state[np.newaxis])
    adjusted = q_mean[0] - beta * q_std[0]
    return int(np.argmax(adjusted))
