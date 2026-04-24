"""Active learning module with composable pool, strategies, and iterator."""

from probly.evaluation.active_learning._torch_estimator import (
    MCDropoutEstimator as MCDropoutEstimator,
    TorchEnsembleEstimator as TorchEnsembleEstimator,
    TorchEstimator as TorchEstimator,
)
from probly.evaluation.active_learning.loop import (
    ALState as ALState,
    active_learning_steps as active_learning_steps,
)
from probly.evaluation.active_learning.metrics import (
    compute_accuracy as compute_accuracy,
    compute_ece as compute_ece,
    compute_nauc as compute_nauc,
)
from probly.evaluation.active_learning.pool import (
    ActiveLearningPool as ActiveLearningPool,
)
from probly.evaluation.active_learning.strategies import (
    BADGEQuery as BADGEQuery,
    EntropyQuery as EntropyQuery,
    Estimator as Estimator,
    MarginSampling as MarginSampling,
    MutualInfoQuery as MutualInfoQuery,
    QueryStrategy as QueryStrategy,
    RandomQuery as RandomQuery,
    UncertaintyQuery as UncertaintyQuery,
)
