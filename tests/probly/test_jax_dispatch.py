"""First calls inside JAX transformations must lazily load their backend."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("jax")


@pytest.mark.parametrize(
    ("module", "name", "expression", "backend"),
    [
        *[
            ("probly.metrics", name, f"{name}(p, jnp.array([0, 1]))", "probly.metrics.jax")
            for name in (
                "log_loss_per_instance",
                "brier_loss_per_instance",
                "spherical_loss_per_instance",
                "zero_one_loss_per_instance",
                "accuracy",
            )
        ],
        (
            "probly.metrics",
            "log_loss_per_instance",
            (
                "jax.grad(lambda q: log_loss_per_instance(create_categorical_distribution(q), jnp.array([0])).sum())"
                "(jnp.array([[1.0, 0.0]]))"
            ),
            "probly.metrics.jax",
        ),
        (
            "probly.metrics",
            "log_loss_per_instance",
            (
                "log_loss_per_instance(create_categorical_distribution_from_logits(jnp.array([[0.0, -1000.0]])), "
                "jnp.array([1]))"
            ),
            "probly.metrics.jax",
        ),
        (
            "probly.quantification",
            "entropy",
            "entropy(p)",
            "probly.quantification.measure.distribution.jax",
        ),
        *[
            ("probly.conformal_scores", name, f"{name}(p, jnp.array([0, 1]))", f"probly.conformal_scores.{backend}.jax")
            for name, backend in (
                ("lac_score", "lac"),
                ("dirichlet_rl_score_func", "dirichlet_relative_likelihood"),
                ("absolute_error_score", "absolute_error"),
            )
        ],
        *[
            (
                "probly.conformal_scores",
                name,
                f"{name}(jnp.sort(p, axis=-1), jnp.array([0, 1]))",
                f"probly.conformal_scores.{backend}.jax",
            )
            for name, backend in (("cqr_score", "cqr"), ("cqr_r_score", "cqr_r"))
        ],
        *[
            (
                "probly.conformal_scores",
                name,
                f"{name}(randomized=False)(p, jnp.array([0, 1]))",
                f"probly.conformal_scores.{backend}.jax",
            )
            for name, backend in (("APSScore", "aps"), ("RAPSScore", "raps"), ("SAPSScore", "saps"))
        ],
        (
            "probly.conformal_scores",
            "uacqr_score",
            "uacqr_score(jnp.sort(jnp.stack([p, p + 0.1]), axis=-1), jnp.array([0, 1]))",
            "probly.conformal_scores.uacqr.jax",
        ),
        *[
            (
                "probly.representation.distribution",
                name,
                expression,
                f"probly.representation.distribution.{backend}",
            )
            for name, expression, backend in (
                (
                    "create_categorical_distribution",
                    "create_categorical_distribution(p).probabilities",
                    "jax_categorical",
                ),
                (
                    "create_categorical_distribution_from_logits",
                    "create_categorical_distribution_from_logits(p).log_probabilities",
                    "jax_categorical",
                ),
                ("create_bernoulli_distribution", "create_bernoulli_distribution(p).probabilities", "jax_bernoulli"),
                (
                    "create_bernoulli_distribution_from_logits",
                    "create_bernoulli_distribution_from_logits(p).log_probabilities",
                    "jax_bernoulli",
                ),
                (
                    "create_dirichlet_distribution_from_alphas",
                    "create_dirichlet_distribution_from_alphas(p).mean.probabilities",
                    "jax_dirichlet",
                ),
                ("create_gaussian_distribution", "create_gaussian_distribution(p).var", "jax_gaussian"),
            )
        ],
        ("probly.representation.sample", "create_sample", "create_sample(p).array", "probly.representation.sample.jax"),
        (
            "probly.representation.conformal_set",
            "create_onehot_conformal_set",
            "create_onehot_conformal_set(p > 0.5).array",
            "probly.representation.conformal_set.jax",
        ),
        (
            "probly.representation.conformal_set",
            "create_interval_conformal_set",
            "create_interval_conformal_set(p, p + 0.1).array",
            "probly.representation.conformal_set.jax",
        ),
        (
            "probly.representation.credal_set",
            "create_distance_based_credal_set_from_center_and_radius",
            "create_distance_based_credal_set_from_center_and_radius(p, 0.1).lower()",
            "probly.representation.credal_set.jax",
        ),
        (
            "probly.evaluation.selective_prediction",
            "selective_prediction",
            "selective_prediction(p[:, 0], p[:, 1], n_bins=2)",
            "probly.evaluation.selective_prediction.jax",
        ),
        (
            "probly.utils.quantile",
            "calculate_quantile",
            "calculate_quantile(p[:, 0], 0.5)",
            "probly.utils.quantile.jax",
        ),
        (
            "probly.utils.quantile",
            "calculate_weighted_quantile",
            "calculate_weighted_quantile(p[:, 0], 0.5, p[:, 1])",
            "probly.utils.quantile.jax",
        ),
    ],
)
def test_first_call_under_jit_loads_backend(module, name, expression, backend):
    # Each case needs a fresh interpreter: importing another function from the same backend can hide a missing hook.
    script = textwrap.dedent(
        f"""\
        import sys
        import jax
        import jax.numpy as jnp
        from jax.experimental import checkify
        from {module} import {name}
        from probly.representation.distribution import (
            create_categorical_distribution,
            create_categorical_distribution_from_logits,
        )

        assert {backend!r} not in sys.modules

        def evaluate(p):
            return {expression}

        probabilities = jnp.array([[0.7, 0.3], [0.2, 0.8]])
        error, actual = jax.jit(checkify.checkify(evaluate))(probabilities)
        error.throw()
        assert {backend!r} in sys.modules
        expected = evaluate(probabilities)
        for a, e in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
            assert jnp.allclose(a, e, equal_nan=True), (a, e)
        """
    )
    result = subprocess.run(  # noqa: S603 - fixed imports and expressions from parametrized cases
        [sys.executable, "-c", script], capture_output=True, text=True, check=False, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr
