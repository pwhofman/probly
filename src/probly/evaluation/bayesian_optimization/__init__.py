"""Bayesian optimization with composable surrogates and acquisitions.

Typical workflow::

    from probly.evaluation.bayesian_optimization import (
        BotorchGPSurrogate, UpperConfidenceBound, bayesian_optimization_steps,
        hartmann, regret_curve,
    )

    objective = hartmann()
    for state in bayesian_optimization_steps(
        objective,
        BotorchGPSurrogate(),
        UpperConfidenceBound(beta=2.0, seed=0),
        n_init=10,
        n_iterations=30,
        seed=0,
    ):
        regret = state.best_y - objective.optimal_value

Surrogates are probly representation predictors returning a
:class:`~probly.representation.distribution.TorchGaussianDistribution`, so
``predict(surrogate, x)`` yields a posterior with ``.mean`` / ``.var`` -- the
same shape any probly UQ method exposes for regression. Three surrogates
ship by default:

* :class:`BotorchGPSurrogate` -- exact GP via botorch.
* :class:`RandomForestSurrogate` -- per-tree mean/variance from sklearn's
  ``RandomForestRegressor``.
* :class:`MCDropoutSurrogate` -- a torch MLP made stochastic at inference
  via :func:`~probly.transformation.dropout.dropout`.
"""

from probly.evaluation.bayesian_optimization.acquisition import (
    Acquisition as Acquisition,
    RandomAcquisition as RandomAcquisition,
    UpperConfidenceBound as UpperConfidenceBound,
)
from probly.evaluation.bayesian_optimization.loop import (
    BOState as BOState,
    bayesian_optimization_steps as bayesian_optimization_steps,
)
from probly.evaluation.bayesian_optimization.metrics import (
    best_so_far as best_so_far,
    regret_curve as regret_curve,
    regret_nauc as regret_nauc,
    simple_regret as simple_regret,
)
from probly.evaluation.bayesian_optimization.objectives import (
    Objective as Objective,
    forrester as forrester,
    hartmann as hartmann,
    rosenbrock as rosenbrock,
)
from probly.evaluation.bayesian_optimization.surrogate import (
    BNNSurrogate as BNNSurrogate,
    BotorchGPSurrogate as BotorchGPSurrogate,
    MCDropoutSurrogate as MCDropoutSurrogate,
    RandomForestSurrogate as RandomForestSurrogate,
    Surrogate as Surrogate,
    posterior_mean_std as posterior_mean_std,
)
from probly.evaluation.bayesian_optimization.visualization import (
    plot_objective_1d as plot_objective_1d,
    plot_objective_2d as plot_objective_2d,
)

__all__ = [
    "Acquisition",
    "BNNSurrogate",
    "BOState",
    "BotorchGPSurrogate",
    "MCDropoutSurrogate",
    "Objective",
    "RandomAcquisition",
    "RandomForestSurrogate",
    "Surrogate",
    "UpperConfidenceBound",
    "bayesian_optimization_steps",
    "best_so_far",
    "forrester",
    "hartmann",
    "plot_objective_1d",
    "plot_objective_2d",
    "posterior_mean_std",
    "regret_curve",
    "regret_nauc",
    "rosenbrock",
    "simple_regret",
]
