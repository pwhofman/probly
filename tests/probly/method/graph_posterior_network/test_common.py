"""Backend-agnostic tests for graph_posterior_network/_common.py.

Targets the small branches in ``graph_evidence_log_scale`` and the
generator dispatch fallbacks that previously had no coverage.
"""

from __future__ import annotations

from math import log, pi

import pytest

from probly.method.graph_posterior_network._common import (
    cuq_graph_neural_network_generator,
    graph_evidence_log_scale,
    graph_posterior_network_generator,
    lop_graph_posterior_network_generator,
)


class TestGraphEvidenceLogScale:
    """Cover every branch of ``graph_evidence_log_scale``."""

    def test_none_scale_returns_zero(self) -> None:
        """Passing ``None`` short-circuits to ``0.0``."""
        assert graph_evidence_log_scale(None, latent_dim=4, num_classes=3) == 0.0

    def test_latent_new_uses_4_pi_term(self) -> None:
        """The ``latent-new`` family uses ``0.5 * d * log(4 pi)`` (line 68)."""
        result = graph_evidence_log_scale("latent-new", latent_dim=4, num_classes=3)
        expected = 0.5 * 4 * log(4 * pi)
        assert result == pytest.approx(expected)

    def test_latent_old_uses_2_pi_with_dim_term(self) -> None:
        """The ``latent-old`` family adds ``log(d + 1)`` (lines 69-70)."""
        result = graph_evidence_log_scale("latent-old", latent_dim=4, num_classes=3)
        expected = 0.5 * (4 * log(2 * pi) + log(5))
        assert result == pytest.approx(expected)

    def test_unknown_scale_raises(self) -> None:
        """Unknown scale strings raise ValueError (lines 72-73)."""
        with pytest.raises(ValueError, match="Unknown graph posterior evidence scale"):
            graph_evidence_log_scale("not-a-scale", latent_dim=4, num_classes=3)  # ty: ignore[invalid-argument-type]

    def test_plus_classes_suffix_adds_log_num_classes(self) -> None:
        """The ``-plus-classes`` suffix adds ``log(num_classes)`` (line 75)."""
        base = graph_evidence_log_scale("latent-new", latent_dim=4, num_classes=3)
        with_classes = graph_evidence_log_scale("latent-new-plus-classes", latent_dim=4, num_classes=3)
        assert with_classes == pytest.approx(base + log(3))


class TestUnregisteredGeneratorsRaise:
    """The default branch of each generator raises NotImplementedError."""

    def test_graph_posterior_network_generator_unregistered(self) -> None:
        """Unregistered encoder type triggers the NotImplementedError (lines 94-95)."""
        with pytest.raises(NotImplementedError, match="No graph posterior network registered"):
            graph_posterior_network_generator("not-a-model", latent_dim=2, num_classes=2)

    def test_lop_graph_posterior_network_generator_unregistered(self) -> None:
        """Unregistered encoder type triggers the NotImplementedError (lines 115-116)."""
        with pytest.raises(NotImplementedError, match="No LOP graph posterior network registered"):
            lop_graph_posterior_network_generator("not-a-model", latent_dim=2, num_classes=2)

    def test_cuq_graph_neural_network_generator_unregistered(self) -> None:
        """Unregistered encoder type triggers the NotImplementedError (lines 135-136)."""
        with pytest.raises(NotImplementedError, match="No CUQ graph neural network registered"):
            cuq_graph_neural_network_generator("not-a-model", latent_dim=2, num_classes=2)
