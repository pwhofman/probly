"""Smoke tests for the GPN demo."""

from __future__ import annotations

import json

from gpn_uq.demo import build_model, evaluate_model, make_synthetic_graph, run_demo, train_model


def test_demo_smoke() -> None:
    data = make_synthetic_graph(nodes_per_class=8, seed=0)
    model = build_model("GPN", data.x.shape[-1], 3)
    train_model(model, data, "GPN", epochs=1, lr=0.01, use_mixed_precision=False)
    result = evaluate_model(model, data, n_bins=4, use_mixed_precision=False)
    assert 0.0 <= result.accuracy <= 1.0
    assert result.selective_loss.shape == (4,)


def test_inference_only_missing_checkpoints_uses_placeholders(tmp_path) -> None:
    output_dir = tmp_path / "results"
    run_demo(
        output_dir,
        experiment="synthetic",
        nodes_per_class=8,
        checkpoint_dir=tmp_path / "checkpoints",
        inference_only=True,
    )

    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert (output_dir / "graph_uncertainty.pdf").exists()
    assert (output_dir / "selective_prediction.pdf").exists()
    assert set(metrics) == {"GPN", "LOP-GPN", "CUQ-GNN"}
    assert all(model_metrics["accuracy"] is None for model_metrics in metrics.values())
    assert all(model_metrics["selective_area"] is None for model_metrics in metrics.values())
