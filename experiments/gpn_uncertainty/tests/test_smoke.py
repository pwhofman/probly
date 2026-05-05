"""Smoke tests for the GPN demo."""

from __future__ import annotations

from gpn_uq.demo import build_model, evaluate_model, make_synthetic_graph, train_model


def test_demo_smoke() -> None:
    data = make_synthetic_graph(nodes_per_class=8, seed=0)
    model = build_model("GPN", data.x.shape[-1], 3)
    train_model(model, data, "GPN", epochs=1, lr=0.01)
    result = evaluate_model(model, data, n_bins=4)
    assert 0.0 <= result.accuracy <= 1.0
    assert result.selective_loss.shape == (4,)
