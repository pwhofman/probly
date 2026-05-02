"""Torch-backend BO tests: surrogate posteriors, the loop, and acquisitions.

These tests instantiate every shipped surrogate and verify that:

* ``predict(surrogate, x)`` returns a probly :class:`Representation` (a
  :class:`TorchGaussianDistribution` for the GP, a :class:`TorchSample`
  for the ensemble / NN-based surrogates) and that ``posterior_mean_std``
  extracts ``(mean, std)`` from each with the right shape.
* The loop yields ``n_iterations + 1`` BO states and grows the
  observation tensor by one per iteration.
* The best observed value is monotonically non-increasing across BO
  iterations (running-min invariant of the loop).

Surrogate quality comparisons (GP baseline vs RF / MC-Dropout / BNN)
live in ``probly_benchmark.bayesian_optimization`` -- they need many
seeds and a real budget to be informative, which doesn't belong in unit
tests.
"""

from __future__ import annotations

import itertools

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("botorch")
pytest.importorskip("gpytorch")
pytest.importorskip("sklearn")

from probly.evaluation.bayesian_optimization import (  # noqa: E402
    BNNSurrogate,
    BotorchGPSurrogate,
    MCDropoutSurrogate,
    RandomAcquisition,
    RandomForestSurrogate,
    UpperConfidenceBound,
    bayesian_optimization_steps,
    forrester,
    posterior_mean_std,
    rosenbrock,
)
from probly.predictor import predict  # noqa: E402
from probly.representation import Representation  # noqa: E402
from probly.representation.distribution.torch_gaussian import TorchGaussianDistribution  # noqa: E402
from probly.representation.sample.torch import TorchSample  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _toy_xy(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Return small (n, 2) train inputs and (n,) outputs from a smooth function."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(12, 2, generator=g, dtype=torch.float64)
    y = (x[:, 0] - 0.5) ** 2 + (x[:, 1] - 0.3) ** 2 + 0.05 * torch.randn(12, generator=g, dtype=torch.float64)
    return x, y


# ---------------------------------------------------------------------------
# Surrogate -> Predictor[GaussianDistribution] integration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("surrogate_factory", "expected_repr_type"),
    [
        (BotorchGPSurrogate, TorchGaussianDistribution),
        (lambda: RandomForestSurrogate(n_estimators=20, seed=0), TorchSample),
        (lambda: MCDropoutSurrogate(epochs=20, num_samples=5, seed=0), TorchSample),
        (lambda: BNNSurrogate(epochs=20, num_samples=5, seed=0), TorchSample),
    ],
    ids=["gp", "rf", "dropout", "bnn"],
)
def test_predict_returns_native_representation(surrogate_factory, expected_repr_type):
    surrogate = surrogate_factory()
    x_train, y_train = _toy_xy()
    surrogate.fit(x_train, y_train)

    x_query = torch.rand(7, 2, dtype=torch.float64)
    rep = predict(surrogate, x_query)
    assert isinstance(rep, Representation)
    assert isinstance(rep, expected_repr_type)

    mean, std = posterior_mean_std(rep)
    assert mean.shape == (7,)
    assert std.shape == (7,)
    assert torch.all(std > 0)


def test_surrogate_fit_required_before_predict():
    surrogate = BotorchGPSurrogate()
    with pytest.raises(RuntimeError, match="must be fit"):
        predict(surrogate, torch.zeros(1, 2, dtype=torch.float64))


# ---------------------------------------------------------------------------
# Loop wiring
# ---------------------------------------------------------------------------


def test_loop_yields_correct_number_of_states_and_grows_observations():
    objective = forrester()
    surrogate = BotorchGPSurrogate()
    acquisition = RandomAcquisition(seed=0)

    states = list(
        bayesian_optimization_steps(
            objective,
            surrogate,
            acquisition,
            n_init=4,
            n_iterations=3,
            seed=0,
        )
    )
    assert len(states) == 4  # 1 initial + 3 acquisition rounds
    assert states[0].x.shape[0] == 4
    assert states[-1].x.shape[0] == 4 + 3
    assert all(states[i].iteration == i for i in range(len(states)))


def test_loop_with_random_forest_surrogate_runs_end_to_end():
    objective = forrester()
    surrogate = RandomForestSurrogate(n_estimators=20, seed=0)
    acquisition = UpperConfidenceBound(beta=2.0, seed=0, n_raw_samples=128)

    states = list(bayesian_optimization_steps(objective, surrogate, acquisition, n_init=4, n_iterations=3, seed=0))
    assert len(states) == 4
    # The non-differentiable RF must still propose distinct candidates inside bounds.
    proposed = states[-1].x[-3:]
    lo, hi = objective.bounds[0], objective.bounds[1]
    assert torch.all(proposed >= lo)
    assert torch.all(proposed <= hi)


# ---------------------------------------------------------------------------
# Loop invariant: running-min of best_y
# ---------------------------------------------------------------------------


def test_best_y_is_monotone_non_increasing_across_iterations():
    objective = forrester()
    surrogate = BotorchGPSurrogate()
    acquisition = UpperConfidenceBound(beta=2.0, seed=0, n_raw_samples=128)
    bests = [
        state.best_y
        for state in bayesian_optimization_steps(objective, surrogate, acquisition, n_init=4, n_iterations=6, seed=0)
    ]
    for previous, current in itertools.pairwise(bests):
        assert current <= previous


# ---------------------------------------------------------------------------
# 2-D objective sanity: surrogate posterior shape matches a flattened mesh
# ---------------------------------------------------------------------------


def test_surrogate_handles_2d_rosenbrock_grid():
    objective = rosenbrock(dim=2)
    surrogate = BotorchGPSurrogate()
    x = torch.tensor(
        [
            [-2.0, -1.0],
            [0.5, 0.5],
            [1.0, 1.0],
            [3.0, 8.0],
            [-3.0, 5.0],
        ],
        dtype=torch.float64,
    )
    y = objective(x)
    surrogate.fit(x, y)

    # Build a small flattened mesh and check the posterior shape.
    grid = torch.cartesian_prod(
        torch.linspace(-2.0, 2.0, 5, dtype=torch.float64),
        torch.linspace(-2.0, 2.0, 5, dtype=torch.float64),
    )
    mean, std = posterior_mean_std(predict(surrogate, grid))
    assert mean.shape == (25,)
    assert std.shape == (25,)
