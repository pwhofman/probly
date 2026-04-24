"""Active learning benchmark on two-moons (classification) and sine regression.

Compares uncertainty sampling vs random sampling over multiple seeds,
plotting mean ± 95 % CI curves for each strategy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch
    from torch import nn

from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
from sklearn.datasets import make_moons
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from probly.evaluation.active_learning import active_learning_loop  # ty: ignore[unresolved-import]
from probly.evaluation.active_learning._torch_estimator import TorchEstimator
from probly.evaluation.active_learning._utils import badge_query, margin_sampling, total_entropy

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_TRAIN = 300
N_TEST = 200
POOL_SIZE = 5
N_ITERATIONS = 10
N_SEEDS = 10


# ---------------------------------------------------------------------------
# Query functions / wrappers
# ---------------------------------------------------------------------------
def random_query_fn(outputs: np.ndarray) -> np.ndarray:
    """Return uniform random scores — effectively random sampling."""
    return np.ones(outputs.shape[0])


class BADGEClassifier:
    """Sklearn-compatible wrapper that plugs BADGE selection into the active learning loop.

    BADGE selects instances by k-means++ over gradient embeddings.  For
    sklearn models without an explicit embedding layer the input features are
    used as a proxy embedding.  ``uncertainty_scores`` returns a binary array
    (1 for BADGE-selected, 0 otherwise) so the loop's top-n selection retrieves
    exactly the BADGE-chosen indices.

    Args:
        base_model: Any sklearn classifier with ``fit``, ``predict``, and
            ``predict_proba``.
        pool_size: Number of instances queried per iteration — must match the
            ``pool_size`` passed to ``active_learning_loop``.
    """

    def __init__(self, base_model: Any, pool_size: int = POOL_SIZE) -> None:  # noqa: ANN401
        """See class docstring."""
        self._model = base_model
        self._pool_size = pool_size

    def fit(self, x: np.ndarray, y: np.ndarray) -> BADGEClassifier:
        """Delegate to the base model's fit."""
        self._model.fit(x, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Delegate to the base model's predict."""
        return self._model.predict(x)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Delegate to the base model's predict_proba."""
        return self._model.predict_proba(x)

    def uncertainty_scores(self, x: np.ndarray) -> np.ndarray:
        """Return a binary mask with 1s at the BADGE-selected positions."""
        probs = self._model.predict_proba(x)
        n = min(self._pool_size, len(x))
        selected = badge_query(x, probs, n)
        scores = np.zeros(len(x))
        scores[selected] = 1.0
        return scores


class SimpleMLP:
    """Two-hidden-layer MLP with an ``embed`` method for penultimate-layer access.

    The architecture is ``Linear -> ReLU -> Linear -> ReLU -> Linear``.
    ``embed`` returns the activations after the second ReLU, which serve as
    gradient embeddings for BADGE.

    Args:
        in_features: Input dimensionality.
        hidden: Width of each hidden layer.
        out_features: Number of output classes.
    """

    def __new__(cls, in_features: int, hidden: int, out_features: int) -> nn.Module:
        """Return a freshly constructed ``_MLP`` instance."""
        from torch import nn  # noqa: PLC0415

        class _MLP(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.features = nn.Sequential(
                    nn.Linear(in_features, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                )
                self.head = nn.Linear(hidden, out_features)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.head(self.features(x))

            def embed(self, x: torch.Tensor) -> torch.Tensor:
                """Return penultimate-layer activations of shape ``(n, hidden)``."""
                return self.features(x)

        return _MLP()


class BADGEMLPEstimator(TorchEstimator):
    """``TorchEstimator`` subclass that uses real MLP embeddings for BADGE selection.

    Extends :class:`~probly.evaluation.active_learning._torch_estimator.TorchEstimator`
    with an ``uncertainty_scores`` method.  At query time the penultimate-layer
    activations are extracted via the model's ``embed`` method and passed to
    :func:`~probly.evaluation.active_learning._utils.badge_query` together with
    the softmax probabilities.

    Args:
        model: A :class:`SimpleMLP` instance (or any ``nn.Module`` that exposes
            an ``embed`` method returning penultimate-layer features).
        pool_size: Number of instances queried per iteration — must match the
            ``pool_size`` passed to ``active_learning_loop``.
        **kwargs: Forwarded to :class:`TorchEstimator`.
    """

    def __init__(self, model: nn.Module, *, pool_size: int = POOL_SIZE, **kwargs: Any) -> None:  # noqa: ANN401
        """See class docstring."""
        super().__init__(model, **kwargs)
        self._pool_size = pool_size

    def _embed(self, x: np.ndarray) -> np.ndarray:
        """Return penultimate-layer activations of shape ``(n_samples, hidden)``."""
        from typing import cast as _cast  # noqa: PLC0415

        import torch  # noqa: PLC0415

        model_any = _cast("Any", self.model)
        self.model.eval()
        x_t = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        parts: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(x_t), self.pred_batch_size):
                batch = x_t[start : start + self.pred_batch_size]
                emb = model_any.embed(batch)
                parts.append(emb.cpu().numpy())
        return np.concatenate(parts, axis=0)

    def uncertainty_scores(self, x: np.ndarray) -> np.ndarray:
        """Return a binary mask with 1s at the BADGE-selected positions."""
        embeddings = self._embed(x)
        probs = self._predict_proba(x)
        n = min(self._pool_size, len(x))
        selected = badge_query(embeddings, probs, n)
        scores = np.zeros(len(x))
        scores[selected] = 1.0
        return scores


# ---------------------------------------------------------------------------
# Strategy style registry
# ---------------------------------------------------------------------------
_STRATEGY_STYLES: dict[str, dict] = {
    "margin": {"color": "C0", "label": "Margin sampling", "linestyle": "-"},
    "entropy": {"color": "C2", "label": "Entropy sampling", "linestyle": "-."},
    "badge_gbm": {"color": "C3", "label": "BADGE (GBM, feature emb)", "linestyle": ":"},
    "badge_mlp": {"color": "C4", "label": "BADGE (MLP, layer emb)", "linestyle": (0, (3, 1, 1, 1))},
    "random": {"color": "C1", "label": "Random sampling", "linestyle": "--"},
}


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
def make_sine_regression(
    n_train: int = N_TRAIN,
    n_test: int = N_TEST,
    noise: float = 0.1,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a noisy sine regression dataset."""
    rng = np.random.default_rng(seed)
    x_train = rng.uniform(-np.pi, np.pi, (n_train, 1))
    y_train = np.sin(x_train[:, 0]) + rng.normal(0, noise, n_train)
    x_test = np.linspace(-np.pi, np.pi, n_test).reshape(-1, 1)
    y_test = np.sin(x_test[:, 0])
    return x_train, y_train, x_test, y_test


def make_two_moons_classification(
    n_train: int = N_TRAIN,
    n_test: int = N_TEST,
    noise: float = 0.1,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a two-moons binary classification dataset."""
    x_train, y_train = make_moons(n_samples=n_train, noise=noise, random_state=seed)
    x_test, y_test = make_moons(n_samples=n_test, noise=noise, random_state=seed + 1)
    return x_train, y_train, x_test, y_test


# ---------------------------------------------------------------------------
# Multi-seed runner
# ---------------------------------------------------------------------------
def run_seeds(
    make_dataset: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    make_model: Callable[[int], Any],
    metric: str,
    strategies: dict[str, Any],
    n_seeds: int = N_SEEDS,
) -> dict[str, np.ndarray]:
    """Return ``{strategy_name: (n_seeds, n_iterations)}`` score arrays.

    Args:
        make_dataset: Callable that returns (x_train, y_train, x_test, y_test) given a seed.
        make_model: Callable that returns a fresh model given a seed.
        metric: Metric name passed to ``active_learning_loop``.
        strategies: Mapping from strategy name to query function (``None`` for
            models that expose ``uncertainty_scores``).
        n_seeds: Number of random seeds to average over.
    """
    runs: dict[str, list[list[float]]] = {name: [] for name in strategies}

    for seed in range(n_seeds):
        x_train, y_train, x_test, y_test = make_dataset(seed=seed)

        for name, query_fn in strategies.items():
            _, _, scores, _ = active_learning_loop(
                make_model(seed),
                x_train,
                y_train,
                x_test,
                y_test,
                query_fn=query_fn,
                metric=metric,
                pool_size=POOL_SIZE,
                n_iterations=N_ITERATIONS,
                seed=seed,
            )
            runs[name].append(scores)

    return {name: np.array(v) for name, v in runs.items()}


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def _ci95(arr: np.ndarray) -> np.ndarray:
    """Return half-width of 95 % t-CI across axis=0."""
    n = arr.shape[0]
    se = arr.std(axis=0, ddof=1) / np.sqrt(n)
    return se * stats.t.ppf(0.975, df=n - 1)


def plot_comparison(
    ax: plt.Axes,
    runs: dict[str, np.ndarray],
    ylabel: str,
    title: str,
) -> None:
    """Plot mean ± 95 % CI curves for all strategies."""
    iterations = np.arange(1, next(iter(runs.values())).shape[1] + 1)
    handles = []

    for name, data in runs.items():
        style = _STRATEGY_STYLES.get(name, {"color": "grey", "label": name, "linestyle": "-"})
        mean = data.mean(axis=0)
        ci = _ci95(data)

        ax.plot(
            iterations,
            mean,
            color=style["color"],
            linestyle=style["linestyle"],
            marker="o",
            markersize=3,
            linewidth=1.5,
        )
        ax.fill_between(iterations, mean - ci, mean + ci, alpha=0.15, color=style["color"])

        nauc_vals = [float(np.trapezoid(row, x=np.arange(len(row))) / max(len(row) - 1, 1)) for row in data]
        label = style["label"] + f"  (NAUC={np.mean(nauc_vals):.3f})"
        handles.append(
            plt.Line2D(
                [0],
                [0],
                color=style["color"],
                linestyle=style["linestyle"],
                marker="o",
                markersize=3,
                linewidth=1.5,
                label=label,
            )
        )

    ax.legend(handles=handles, fontsize=8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def plot_results(
    clf_runs: dict[str, np.ndarray],
    reg_runs: dict[str, np.ndarray],
    n_seeds: int,
) -> None:
    """Render side-by-side comparison plots for classification and regression."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Active learning benchmark  ({n_seeds} seeds, mean +/- 95 % CI)", fontsize=11)

    plot_comparison(axes[0], clf_runs, ylabel="Accuracy", title="Two Moons -- Classification")
    plot_comparison(axes[1], reg_runs, ylabel="Negative MAE", title="Sine Regression")

    axes[0].set_ylim(0, 1)
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(n_seeds: int = N_SEEDS) -> None:
    """Run active learning benchmark comparing uncertainty vs random sampling."""
    print(f"=== Two Moons Classification  ({n_seeds} seeds) ===")
    clf_runs = run_seeds(
        make_two_moons_classification,
        lambda seed: GradientBoostingClassifier(n_estimators=50, random_state=seed),
        metric="accuracy",
        strategies={
            "margin": margin_sampling,
            "entropy": total_entropy,
            "random": random_query_fn,
        },
        n_seeds=n_seeds,
    )

    print(f"=== Two Moons Classification -- BADGE (GBM)  ({n_seeds} seeds) ===")
    badge_gbm_runs = run_seeds(
        make_two_moons_classification,
        lambda seed: BADGEClassifier(GradientBoostingClassifier(n_estimators=50, random_state=seed)),
        metric="accuracy",
        strategies={"badge_gbm": None},
        n_seeds=n_seeds,
    )
    clf_runs.update(badge_gbm_runs)

    print(f"=== Two Moons Classification -- BADGE (MLP)  ({n_seeds} seeds) ===")
    badge_mlp_runs = run_seeds(
        make_two_moons_classification,
        lambda _: BADGEMLPEstimator(
            SimpleMLP(in_features=2, hidden=64, out_features=2),
            pool_size=POOL_SIZE,
            n_epochs=50,
            optimizer_kwargs={"lr": 1e-3},
        ),
        metric="accuracy",
        strategies={"badge_mlp": None},
        n_seeds=n_seeds,
    )
    clf_runs.update(badge_mlp_runs)

    for name, data in clf_runs.items():
        print(f"  {name:>12}: mean accuracy @ last iter = {data[:, -1].mean():.4f}")

    print()
    print(f"=== Sine Regression  ({n_seeds} seeds) ===")
    reg_runs = run_seeds(
        make_sine_regression,
        lambda seed: GradientBoostingRegressor(n_estimators=50, random_state=seed),
        metric="mae",
        strategies={"random": random_query_fn},
        n_seeds=n_seeds,
    )
    for name, data in reg_runs.items():
        print(f"  {name:>10}: mean neg-MAE @ last iter = {data[:, -1].mean():.4f}")

    plot_results(clf_runs, reg_runs, n_seeds)


if __name__ == "__main__":
    main()
