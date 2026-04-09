"""Active learning benchmark on two-moons (classification) and sine regression.

Compares uncertainty sampling vs random sampling over multiple seeds,
plotting mean ± 95 % CI curves for each strategy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
from sklearn.datasets import make_moons
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from probly.evaluation.active_learning import active_learning_loop

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_TRAIN = 300
N_TEST = 200
POOL_SIZE = 5
N_ITERATIONS = 10
N_SEEDS = 10


# ---------------------------------------------------------------------------
# Query functions
# ---------------------------------------------------------------------------
def random_query_fn(outputs: np.ndarray) -> np.ndarray:
    """Return uniform random scores — effectively random sampling."""
    return np.ones(outputs.shape[0])


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
    n_seeds: int = N_SEEDS,
) -> dict[str, np.ndarray]:
    """Return ``{"uncertainty": (n_seeds, n_iterations), "random": ...}`` score arrays."""
    unc_runs: list[list[float]] = []
    rnd_runs: list[list[float]] = []

    for seed in range(n_seeds):
        x_train, y_train, x_test, y_test = make_dataset(seed=seed)

        _, _, unc_scores, _ = active_learning_loop(
            make_model(seed),
            x_train,
            y_train,
            x_test,
            y_test,
            metric=metric,
            pool_size=POOL_SIZE,
            n_iterations=N_ITERATIONS,
            seed=seed,
        )
        _, _, rnd_scores, _ = active_learning_loop(
            make_model(seed),
            x_train,
            y_train,
            x_test,
            y_test,
            query_fn=random_query_fn,
            metric=metric,
            pool_size=POOL_SIZE,
            n_iterations=N_ITERATIONS,
            seed=seed,
        )
        unc_runs.append(unc_scores)
        rnd_runs.append(rnd_scores)

    return {
        "uncertainty": np.array(unc_runs),
        "random": np.array(rnd_runs),
    }


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
    """Plot mean ± 95 % CI curves for uncertainty vs random sampling."""
    iterations = np.arange(1, runs["uncertainty"].shape[1] + 1)
    styles = {
        "uncertainty": {"color": "C0", "label": "Uncertainty sampling"},
        "random": {"color": "C1", "label": "Random sampling", "linestyle": "--"},
    }

    for key, data in runs.items():
        mean = data.mean(axis=0)
        ci = _ci95(data)
        kw = styles[key]

        ax.plot(
            iterations,
            mean,
            color=kw.get("color", "C0"),
            linestyle=kw.get("linestyle", "-"),
            marker=kw.get("marker", "o"),
            markersize=3,
            linewidth=1.5,
        )

        ax.fill_between(iterations, mean - ci, mean + ci, alpha=0.15, color=kw["color"])
    # --- NAUC summary in legend labels ---
    for key, data in runs.items():
        nauc_vals = [float(np.trapezoid(row, x=np.arange(len(row))) / max(len(row) - 1, 1)) for row in data]
        mean_nauc = np.mean(nauc_vals)
        styles[key]["label"] += f"  (NAUC={mean_nauc:.3f})"

    # --- build legend handles safely ---
    handles = [
        plt.Line2D(
            [0],
            [0],
            color=s["color"],
            linestyle=s.get("linestyle", "-"),
            marker=s.get("marker", "o"),
            markersize=3,
            linewidth=1.5,
            label=s["label"],
        )
        for s in styles.values()
    ]

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
    fig.suptitle(f"Active learning benchmark  ({n_seeds} seeds, mean ± 95 % CI)", fontsize=11)

    plot_comparison(axes[0], clf_runs, ylabel="Accuracy", title="Two Moons — Classification")
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
        n_seeds=n_seeds,
    )
    for strategy, data in clf_runs.items():
        print(f"  {strategy:>12}: mean accuracy @ last iter = {data[:, -1].mean():.4f}")

    print()
    print(f"=== Sine Regression  ({n_seeds} seeds) ===")
    reg_runs = run_seeds(
        make_sine_regression,
        lambda seed: GradientBoostingRegressor(n_estimators=50, random_state=seed),
        metric="mae",
        n_seeds=n_seeds,
    )
    for strategy, data in reg_runs.items():
        print(f"  {strategy:>12}: mean neg-MAE @ last iter = {data[:, -1].mean():.4f}")

    plot_results(clf_runs, reg_runs, n_seeds)


if __name__ == "__main__":
    main()
