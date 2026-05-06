"""Torch tests for graph posterior networks."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from torch import nn  # noqa: E402
from torch_geometric.data import Data  # noqa: E402
from torch_geometric.utils import add_remaining_self_loops  # noqa: E402

from probly.method.graph_posterior_network import (  # noqa: E402
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
