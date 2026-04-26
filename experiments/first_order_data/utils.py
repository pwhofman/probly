import numpy as np
from typing import Any
from tqdm import tqdm
from scipy.optimize import linprog


# NOTE: this is directly taken from the legacy files: supporting_files/old_packages/probly/metrics.py
def coverage_convex_hull(probs: np.ndarray, targets: np.ndarray, **kwargs: Any) -> float:  # noqa: ANN401
    """Compute credal set coverage via convex hull :cite:`nguyenCredalEnsembling2025`.

    The coverage is defined as the proportion of instances whose true distribution is contained in the convex hull.
    This is computed using linear programming by checking whether the target distribution can be expressed as
    a convex combination of the predicted distributions.

    Args:
        probs: The predicted probabilities as an array of shape (n_instances, n_samples, n_classes).
        targets: The true labels as an array of shape (n_instances, n_classes).
        **kwargs: Additional keyword arguments for the linear programming solver, e.g. tolerance.

    Returns:
        cov: The coverage.

    """
    covered = 0
    n_extrema = probs.shape[1]
    c = np.zeros(n_extrema)  # we do not care about the coefficients in this case
    bounds = [(0, 1)] * n_extrema
    for i in tqdm(range(probs.shape[0]), desc="Instances"):
        a_eq = np.vstack((probs[i].T, np.ones(n_extrema)))
        b_eq = np.concatenate((targets[i], [1]), axis=0)
        res = linprog(c=c, A_eq=a_eq, b_eq=b_eq, bounds=bounds, **kwargs)
        covered += res.success
    cov = covered / probs.shape[0]
    return float(cov)


def coverage_convex_hull_relaxed(
    probs: np.ndarray,
    targets: np.ndarray,
    epsilon: float = 1e-3,
    **kwargs: Any,  # noqa: ANN401
) -> float:
    """Compute relaxed credal set coverage via convex hull.
    An instance is counted as covered if its target distribution lies within
    L1-distance epsilon of the convex hull of the predicted distributions.
    With epsilon=0 its equivalent to the strict convex hull coverage.

    Args:
        probs: The predicted probabilities as an array of shape (n_instances, n_samples, n_classes).
        targets: The true labels as an array of shape (n_instances, n_classes).
        epsilon: L1-distance tolerance for relaxed coverage.
        **kwargs: Additional keyword arguments for the linear programming solver.
    Returns:
        cov: The relaxed coverage.
    """
    covered = 0
    n_extrema = probs.shape[1]
    n_classes = probs.shape[2]
    # variables: [lambda_1, ..., lambda_M, s+_1, ..., s+_K, s-_1, ..., s-_K]
    # objective: minimize sum of slacks (zeros for lambdas, ones for slacks)
    c = np.concatenate([np.zeros(n_extrema), np.ones(2 * n_classes)])
    bounds = [(0, 1)] * n_extrema + [(0, None)] * (2 * n_classes)
    for i in tqdm(range(probs.shape[0]), desc="Instances"):
        # class constraints: V^T @ lambda + s+ - s- = t
        # normalization: sum(lambda) = 1
        a_eq_top = np.hstack([probs[i].T, np.eye(n_classes), -np.eye(n_classes)])
        a_eq_bot = np.concatenate([np.ones(n_extrema), np.zeros(2 * n_classes)])
        a_eq = np.vstack([a_eq_top, a_eq_bot])
        b_eq = np.concatenate([targets[i], [1]])
        res = linprog(c=c, A_eq=a_eq, b_eq=b_eq, bounds=bounds, **kwargs)
        covered += res.success and res.fun <= epsilon  # res.fun is optimal objective value c @ solution
    cov = covered / probs.shape[0]
    return float(cov)
