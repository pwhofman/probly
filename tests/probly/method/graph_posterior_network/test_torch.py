"""Torch tests for graph posterior networks."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from torch import nn  # noqa: E402
from torch_geometric.data import Data  # noqa: E402

from probly.predictor import predict  # noqa: E402
from probly.representation.distribution.torch_dirichlet import TorchDirichletDistribution  # noqa: E402
from probly.train.evidential.torch import mixture_uce_loss, postnet_loss  # noqa: E402
from probly.transformation.graph_posterior_network import (  # noqa: E402
    cuq_graph_neural_network,
    graph_posterior_network,
    lop_graph_posterior_network,
)


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

    raw = model.forward_raw(data)

    assert raw.alpha.shape == (data.num_nodes, 3)
    assert raw.alpha_features.shape == (data.num_nodes, 3)
    assert raw.mixture_weights.shape == (data.num_nodes, data.num_nodes)
    assert torch.isfinite(raw.alpha).all()
    assert (raw.alpha > 0).all()


def test_lop_mixture_uce_loss_backward() -> None:
    data = _tiny_graph()
    model = lop_graph_posterior_network(_encoder(), 3, 3, encoder_dim=5, num_flows=2, propagation_steps=2)
    train_mask = torch.as_tensor(data.train_mask, dtype=torch.bool)
    y = torch.as_tensor(data.y, dtype=torch.long)

    raw = model.forward_raw(data)
    loss = mixture_uce_loss(raw.alpha_features, raw.mixture_weights[train_mask], y[train_mask])
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
