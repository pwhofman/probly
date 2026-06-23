"""Torch tests for graph posterior networks."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from torch import nn  # noqa: E402
from torch_geometric.data import Data  # noqa: E402
from torch_geometric.utils import add_remaining_self_loops  # noqa: E402

from probly.method.graph_posterior_network import (  # noqa: E402
    CUQGraphNeuralNetworkPredictor,
    GraphPosteriorNetworkPredictor,
    LOPGraphPosteriorNetworkPredictor,
    cuq_graph_neural_network,
    graph_posterior_network,
    lop_graph_posterior_network,
)
from probly.predictor import predict  # noqa: E402
from probly.representation.distribution.torch_dirichlet import TorchDirichletDistribution  # noqa: E402
from probly.train.evidential.torch import mixture_uce_loss, postnet_loss  # noqa: E402


def _tiny_graph() -> Data:
    x = torch.tensor(
        [
            [1.0, 0.0, 0.2, 0.0],
            [0.8, 0.1, 0.0, 0.1],
            [0.0, 1.0, 0.2, 0.1],
            [0.1, 0.8, 0.0, 0.2],
            [0.2, 0.1, 1.0, 0.0],
            [0.0, 0.2, 0.8, 0.1],
        ]
    )
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0],
            [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0, 5],
        ],
        dtype=torch.long,
    )
    y = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    train_mask = torch.tensor([True, False, True, False, True, False])
    test_mask = ~train_mask
    return Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask)


def _encoder() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 5))


def test_graph_posterior_network_forward_shape_and_positivity() -> None:
    data = _tiny_graph()
    model = graph_posterior_network(_encoder(), 3, 3, encoder_dim=5, num_flows=2, propagation_steps=2)

    alpha = model(data)

    assert alpha.shape == (data.num_nodes, 3)
    assert torch.isfinite(alpha).all()
    assert (alpha > 0).all()


def test_graph_posterior_network_predict_returns_dirichlet() -> None:
    data = _tiny_graph()
    model = graph_posterior_network(_encoder(), 3, 3, encoder_dim=5, num_flows=2, propagation_steps=2)

    distribution = predict(model, data)

    assert isinstance(distribution, TorchDirichletDistribution)
    assert distribution.alphas.shape == (data.num_nodes, 3)


def test_graph_posterior_network_registers_graph_predictor_protocol() -> None:
    model = graph_posterior_network(_encoder(), 3, 3, encoder_dim=5, num_flows=2, propagation_steps=2)

    assert isinstance(model, GraphPosteriorNetworkPredictor)


def test_graph_posterior_network_loss_backward() -> None:
    data = _tiny_graph()
    model = graph_posterior_network(_encoder(), 3, 3, encoder_dim=5, num_flows=2, propagation_steps=2)
    train_mask = torch.as_tensor(data.train_mask, dtype=torch.bool)
    y = torch.as_tensor(data.y, dtype=torch.long)

    alpha = model(data)
    loss = postnet_loss(alpha[train_mask], y[train_mask], reduction="mean")
    loss.backward()

    assert model.latent_encoder.weight.grad is not None
    assert torch.isfinite(model.latent_encoder.weight.grad).all()


def test_lop_graph_posterior_network_forward_raw_contains_mixture_terms() -> None:
    data = _tiny_graph()
    model = lop_graph_posterior_network(_encoder(), 3, 3, encoder_dim=5, num_flows=2, propagation_steps=2)
    assert isinstance(data.train_mask, torch.Tensor)
    assert isinstance(data.y, torch.Tensor)

    alpha_features, mixture_weights = model(data)

    assert alpha_features.shape == (data.num_nodes, 3)
    assert mixture_weights.shape == (data.num_nodes, data.num_nodes)
    assert torch.isfinite(alpha_features).all()
    assert torch.isfinite(mixture_weights).all()
    assert (alpha_features > 0).all()


def test_lop_graph_posterior_network_registers_lop_predictor_protocol() -> None:
    model = lop_graph_posterior_network(_encoder(), 3, 3, encoder_dim=5, num_flows=2, propagation_steps=2)

    assert isinstance(model, LOPGraphPosteriorNetworkPredictor)
    assert not isinstance(model, GraphPosteriorNetworkPredictor)


def test_lop_graph_posterior_network_sparse_propagation_matches_random_walk_ppr() -> None:
    data = _tiny_graph()
    data.edge_index = torch.cat(
        [
            data.edge_index,
            torch.tensor([[0, 0, 1], [2, 3, 4]], dtype=torch.long),
        ],
        dim=1,
    )
    model = lop_graph_posterior_network(_encoder(), 3, 3, encoder_dim=5, num_flows=2, propagation_steps=2)
    assert isinstance(data.edge_index, torch.Tensor)
    assert data.num_nodes is not None

    _, mixture_weights = model(data)
    edge_weight = torch.ones(data.edge_index.shape[1])
    edge_index, edge_weight = add_remaining_self_loops(data.edge_index, edge_weight, 1.0, data.num_nodes)
    adjacency = torch.zeros(data.num_nodes, data.num_nodes)
    adjacency[edge_index[0], edge_index[1]] = edge_weight
    degree = adjacency.sum(dim=1).clamp_min(1.0)
    propagation_matrix = adjacency / degree.view(-1, 1)
    expected = torch.eye(data.num_nodes)
    for _ in range(2):
        expected = 0.9 * propagation_matrix @ expected + 0.1 * torch.eye(data.num_nodes)

    torch.testing.assert_close(mixture_weights, expected)
    torch.testing.assert_close(mixture_weights.sum(dim=-1), torch.ones(data.num_nodes))


def test_lop_graph_posterior_network_reuses_cached_propagation_weights() -> None:
    data = _tiny_graph()
    model = lop_graph_posterior_network(_encoder(), 3, 3, encoder_dim=5, num_flows=2, propagation_steps=2)

    _, first_weights = model(data)
    _, second_weights = model(data)

    assert second_weights is first_weights


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for autocast dtype checks.")
def test_lop_graph_posterior_network_autocast_uses_float16_propagation_weights() -> None:
    data = _tiny_graph().to("cuda")
    model = lop_graph_posterior_network(_encoder(), 3, 3, encoder_dim=5, num_flows=2, propagation_steps=2).to("cuda")

    with torch.amp.autocast("cuda", dtype=torch.float16):
        _, mixture_weights = model(data)

    assert mixture_weights.dtype == torch.float16


def test_lop_mixture_uce_loss_backward() -> None:
    data = _tiny_graph()
    model = lop_graph_posterior_network(_encoder(), 3, 3, encoder_dim=5, num_flows=2, propagation_steps=2)
    train_mask = torch.as_tensor(data.train_mask, dtype=torch.bool)
    y = torch.as_tensor(data.y, dtype=torch.long)

    alpha_features, mixture_weights = model(data)
    loss = mixture_uce_loss(alpha_features, mixture_weights[train_mask], y[train_mask])
    loss.backward()

    assert model.latent_encoder.weight.grad is not None
    assert torch.isfinite(model.latent_encoder.weight.grad).all()


def test_cuq_graph_neural_network_forward_shape_and_positivity() -> None:
    data = _tiny_graph()
    model = cuq_graph_neural_network(
        _encoder(),
        3,
        3,
        encoder_dim=5,
        num_flows=2,
        propagation_steps=2,
        convolution_name="appnp",
    )

    alpha = model(data)

    assert alpha.shape == (data.num_nodes, 3)
    assert torch.isfinite(alpha).all()
    assert (alpha > 0).all()


def test_cuq_graph_neural_network_registers_cuq_predictor_protocol() -> None:
    model = cuq_graph_neural_network(
        _encoder(),
        3,
        3,
        encoder_dim=5,
        num_flows=2,
        propagation_steps=2,
        convolution_name="appnp",
    )

    assert isinstance(model, CUQGraphNeuralNetworkPredictor)
    assert not isinstance(model, GraphPosteriorNetworkPredictor)


def test_random_walk_norm_with_explicit_edge_weight() -> None:
    """``random_walk_norm`` casts an explicit ``edge_weight`` to ``dtype`` (line 47)."""
    from probly.method.graph_posterior_network.torch import random_walk_norm  # noqa: PLC0415

    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    edge_weight = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)

    new_edge_index, normalized = random_walk_norm(
        edge_index, edge_weight, num_nodes=3, add_self_loops=False, dtype=torch.float32
    )

    assert normalized.dtype == torch.float32
    # Without self-loops the indices remain unchanged.
    assert torch.equal(new_edge_index, edge_index)


def test_validate_data_rejects_non_data() -> None:
    """``_validate_data`` raises TypeError when ``data`` is not a PyG ``Data``."""
    model = graph_posterior_network(_encoder(), 3, 3, encoder_dim=5, num_flows=2, propagation_steps=2)

    with pytest.raises(TypeError, match="torch_geometric.data.Data"):  # noqa: RUF043
        model("not a data object")


def test_encode_returns_hidden_and_latent() -> None:
    """``encode`` exposes both hidden and latent representations (lines 119-123)."""
    data = _tiny_graph()
    model = graph_posterior_network(_encoder(), 3, 3, encoder_dim=5, num_flows=2, propagation_steps=2)

    hidden, latent = model.encode(data)

    assert hidden.shape == (data.num_nodes, 5)
    assert latent.shape == (data.num_nodes, 3)


def test_class_probabilities_uses_class_counts_when_provided() -> None:
    """``class_probabilities`` returns the configured prior when ``class_counts`` is set (line 136)."""
    data = _tiny_graph()
    model = graph_posterior_network(
        _encoder(),
        3,
        3,
        encoder_dim=5,
        num_flows=2,
        propagation_steps=2,
        class_counts=[1.0, 1.0, 2.0],
    )

    probs = model.class_probabilities(data)

    assert torch.allclose(probs, torch.tensor([0.25, 0.25, 0.5]))


def test_class_probabilities_requires_train_mask_when_no_class_counts() -> None:
    """``class_probabilities`` raises when neither ``class_counts`` nor labels are available (lines 139-140)."""
    model = graph_posterior_network(_encoder(), 3, 3, encoder_dim=5, num_flows=2, propagation_steps=2)
    data = Data(x=torch.zeros(3, 4), edge_index=torch.zeros((2, 0), dtype=torch.long))

    with pytest.raises(ValueError, match="data.y and data.train_mask"):  # noqa: RUF043
        model.class_probabilities(data)


def test_feature_evidence_returns_dict() -> None:
    """``feature_evidence`` exposes hidden, latent and beta_ft (lines 163-165)."""
    data = _tiny_graph()
    model = graph_posterior_network(_encoder(), 3, 3, encoder_dim=5, num_flows=2, propagation_steps=2)

    evidence = model.feature_evidence(data)

    assert set(evidence) == {"hidden", "latent", "beta_ft"}
    assert evidence["hidden"].shape == (data.num_nodes, 5)
    assert evidence["latent"].shape == (data.num_nodes, 3)
    assert evidence["beta_ft"].shape == (data.num_nodes, 3)
    assert torch.isfinite(evidence["beta_ft"]).all()


def test_lop_disabled_cache_recomputes_each_call() -> None:
    """When caching is disabled the cache early-returns ``None`` (line 268)."""
    data = _tiny_graph()
    model = lop_graph_posterior_network(
        _encoder(),
        3,
        3,
        encoder_dim=5,
        num_flows=2,
        propagation_steps=2,
        cache_propagation_weights=False,
    )

    _, first_weights = model(data)
    _, second_weights = model(data)

    assert second_weights is not first_weights
    assert torch.allclose(first_weights, second_weights)


def test_lop_zero_size_cache_recomputes_each_call() -> None:
    """A zero-size cache hits the early-return guards in store and lookup (lines 268, 288)."""
    data = _tiny_graph()
    model = lop_graph_posterior_network(
        _encoder(),
        3,
        3,
        encoder_dim=5,
        num_flows=2,
        propagation_steps=2,
        propagation_weight_cache_size=0,
    )

    _, first_weights = model(data)
    _, second_weights = model(data)

    assert second_weights is not first_weights


def test_lop_cache_invalidates_on_dtype_change() -> None:
    """Cached entries are dropped when the dtype query changes (lines 281-282)."""
    data = _tiny_graph()
    model = lop_graph_posterior_network(_encoder(), 3, 3, encoder_dim=5, num_flows=2, propagation_steps=2)

    # First call caches under data.x dtype (float32).
    _, _ = model(data)
    assert isinstance(data.edge_index, torch.Tensor)
    assert data.num_nodes is not None
    cached = model._cached_propagation_weights(data.edge_index, data.num_nodes, torch.float64)  # noqa: SLF001
    # Mismatched dtype must invalidate the cache (returns None).
    assert cached is None


def test_lop_cache_evicts_lru_when_full() -> None:
    """Caching extra ``edge_index`` tensors evicts the oldest entry (line 303)."""
    data_a = _tiny_graph()
    data_b = _tiny_graph()
    model = lop_graph_posterior_network(
        _encoder(),
        3,
        3,
        encoder_dim=5,
        num_flows=2,
        propagation_steps=2,
        propagation_weight_cache_size=1,
    )

    _, _ = model(data_a)
    _, _ = model(data_b)
    # ``data_b``'s edge_index is the only one cached now; ``data_a``'s entry got evicted.
    assert len(model._propagation_weight_cache) == 1  # noqa: SLF001


def test_lop_propagation_weight_dtype_falls_back_to_input() -> None:
    """On CPU the dtype helper returns the input tensor's dtype (line 308 covered, line 307 CUDA-only)."""
    model = lop_graph_posterior_network(_encoder(), 3, 3, encoder_dim=5, num_flows=2, propagation_steps=2)

    beta_ft = torch.zeros(2, 3, dtype=torch.float64)
    assert model._propagation_weight_dtype(beta_ft) == torch.float64  # noqa: SLF001


def test_lop_forward_requires_num_nodes() -> None:
    """``forward`` raises when ``data.num_nodes`` is unset (lines 363-364)."""
    # Pass an explicit class_counts so class_probabilities does not need ``train_mask``.
    model = lop_graph_posterior_network(
        _encoder(),
        3,
        3,
        encoder_dim=5,
        num_flows=2,
        propagation_steps=2,
        class_counts=[1, 1, 1],
    )

    class _DataWithoutNumNodes(Data):
        @property
        def num_nodes(self) -> None:
            return None

    bad_data = _DataWithoutNumNodes(
        x=torch.zeros(3, 4),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
    )

    with pytest.raises(ValueError, match="num_nodes"):
        model(bad_data)


def test_lop_predict_representation_returns_mixture() -> None:
    """``predict_representation`` wraps the LOP outputs in a mixture distribution (lines 373-374)."""
    from probly.representation.distribution.torch_mixture import TorchMixtureDistribution  # noqa: PLC0415

    data = _tiny_graph()
    model = lop_graph_posterior_network(_encoder(), 3, 3, encoder_dim=5, num_flows=2, propagation_steps=2)

    dist = model.predict_representation(data)

    assert isinstance(dist, TorchMixtureDistribution)
    # The mixture stores its components and per-node mixture weights.
    assert dist.components is not None
    assert dist.mixture_weights is not None


def test_cuq_with_gcn_convolution() -> None:
    """The GCN branch of ``CUQGraphNeuralNetwork`` exercises lines 431-432 and 439-440."""
    data = _tiny_graph()
    model = cuq_graph_neural_network(
        _encoder(),
        3,
        3,
        encoder_dim=5,
        num_flows=2,
        propagation_steps=2,
        convolution_name="gcn",
    )

    alpha = model(data)

    assert alpha.shape == (data.num_nodes, 3)
    assert (alpha > 0).all()


def test_cuq_unsupported_convolution_raises() -> None:
    """Unknown convolution names raise ValueError (lines 434-435)."""
    with pytest.raises(ValueError, match="Unsupported CUQ-GNN convolution"):
        cuq_graph_neural_network(
            _encoder(),
            3,
            3,
            encoder_dim=5,
            num_flows=2,
            propagation_steps=2,
            convolution_name="unknown",  # ty: ignore[invalid-argument-type]
        )


def test_cuq_predict_representation_returns_dirichlet() -> None:
    """``predict_representation`` wraps the alpha output in a Dirichlet (lines 445-446)."""
    data = _tiny_graph()
    model = cuq_graph_neural_network(
        _encoder(),
        3,
        3,
        encoder_dim=5,
        num_flows=2,
        propagation_steps=2,
        convolution_name="appnp",
    )

    dist = model.predict_representation(data)

    assert isinstance(dist, TorchDirichletDistribution)
    assert dist.alphas.shape == (data.num_nodes, 3)
