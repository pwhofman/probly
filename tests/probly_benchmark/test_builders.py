"""Tests for probly_benchmark.builders BuildContext plumbing."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from probly_benchmark.builders import BuildContext, build_model


def test_build_context_in_features_defaults_to_none() -> None:
    """Existing callers that omit in_features see no behavior change."""
    ctx = BuildContext(
        base_model_name="resnet18",
        model_type="logit_classifier",
        num_classes=10,
        pretrained=False,
    )
    assert ctx.in_features is None


def test_build_model_threads_in_features_to_tabular_mlp() -> None:
    """build_model('dropout', ...) on a TabularMLP base receives in_features."""
    ctx = BuildContext(
        base_model_name="tabular_mlp",
        model_type="logit_classifier",
        num_classes=3,
        pretrained=False,
        in_features=5,
    )
    model = build_model("dropout", {"p": 0.1}, ctx)
    # Smoke test: a 1-row forward pass with the configured feature count must work.
    import torch  # noqa: PLC0415

    out = model(torch.zeros(1, 5))
    assert out.shape == (1, 3)


def test_build_model_threads_in_features_to_posterior_network() -> None:
    """The posterior_network builder also accepts in_features for tabular encoders."""
    ctx = BuildContext(
        base_model_name="tabular_mlp",
        model_type="probabilistic_classifier",
        num_classes=3,
        pretrained=False,
        in_features=5,
    )
    # Just constructing the model must not raise; we don't run a forward pass
    # because PosteriorNetwork needs class_counts and we'd need a real loader.
    # Pass a dummy loader-like object via train_loader=None — the builder uses
    # uniform class_counts in that branch.
    model = build_model("posterior_network", {"latent_dim": 6}, ctx)
    assert model is not None
