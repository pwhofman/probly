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

The :class:`Surrogate` protocol lets the GP be swapped for any model that
exposes ``fit`` and ``posterior_mean_std`` -- the bundled
:class:`EnsembleSurrogate` is one such alternative that uses cross-member
spread as its uncertainty estimate.
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
    hartmann as hartmann,
    rosenbrock as rosenbrock,
)
from probly.evaluation.bayesian_optimization.surrogate import (
    BotorchGPSurrogate as BotorchGPSurrogate,
    EnsembleSurrogate as EnsembleSurrogate,
    Surrogate as Surrogate,
)

__all__ = [
    "Acquisition",
    "BOState",
    "BotorchGPSurrogate",
    "EnsembleSurrogate",
    "Objective",
    "RandomAcquisition",
    "Surrogate",
    "UpperConfidenceBound",
    "bayesian_optimization_steps",
    "best_so_far",
    "hartmann",
    "regret_curve",
    "regret_nauc",
    "rosenbrock",
    "simple_regret",
]
