"""Tests for optional Hugging Face transformers bindings."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
import torch
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertModel,
    GLPNConfig,
    GLPNForDepthEstimation,
    PatchTSTConfig,
    PatchTSTForPrediction,
    ViTConfig,
    ViTForImageClassification,
)

from probly.method import dropout, ensemble
from probly.predictor import predict_raw
from probly.quantification import quantify
from probly.representer import representer


def _tiny_config(num_labels: int = 3) -> BertConfig:
    config = BertConfig(
        vocab_size=50,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=32,
    )
    config.num_labels = num_labels
    return config


@pytest.fixture
def model() -> BertForSequenceClassification:
    torch.manual_seed(0)
    return BertForSequenceClassification(_tiny_config())


@pytest.fixture
def input_ids() -> torch.Tensor:
    return torch.randint(0, 50, (4, 10))


def test_predict_raw_returns_logits(model: BertForSequenceClassification, input_ids: torch.Tensor) -> None:
    with torch.no_grad():
        out = predict_raw(model, input_ids)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (4, 3)


def test_predict_raw_rejects_headless_model(input_ids: torch.Tensor) -> None:
    base = BertModel(_tiny_config())
    with torch.no_grad(), pytest.raises(TypeError, match="prediction head"):
        predict_raw(base, input_ids)


def test_mc_dropout_end_to_end(model: BertForSequenceClassification, input_ids: torch.Tensor) -> None:
    dropout_model = dropout(model, p=0.1, predictor_type="logit_classifier")
    dropout_model.eval()
    rep = representer(dropout_model, num_samples=5)
    with torch.no_grad():
        sample = rep.represent(input_ids)
    decomposition = quantify(sample)
    assert decomposition["total"].shape == (4,)


def test_vision_model_end_to_end() -> None:
    torch.manual_seed(0)
    config = ViTConfig(
        image_size=8,
        patch_size=4,
        num_channels=3,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
    )
    config.num_labels = 2
    model = ViTForImageClassification(config)
    pixel_values = torch.randn(4, 3, 8, 8)

    with torch.no_grad():
        logits = predict_raw(model, pixel_values)
    assert logits.shape == (4, 2)

    dropout_model = dropout(model, p=0.1, predictor_type="logit_classifier")
    dropout_model.eval()
    with torch.no_grad():
        sample = representer(dropout_model, num_samples=3).represent(pixel_values)
    assert quantify(sample)["total"].shape == (4,)


def test_depth_model_end_to_end() -> None:
    torch.manual_seed(0)
    config = GLPNConfig(
        num_channels=3,
        num_encoder_blocks=2,
        depths=[1, 1],
        sr_ratios=[2, 1],
        hidden_sizes=[8, 16],
        patch_sizes=[7, 3],
        strides=[2, 2],
        num_attention_heads=[1, 2],
        mlp_ratios=[2, 2],
        decoder_hidden_size=16,
    )
    model = GLPNForDepthEstimation(config)
    pixel_values = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        depth = predict_raw(model, pixel_values)
    assert depth.shape == (2, 64, 64)

    members = ensemble(model, num_members=2)
    with torch.no_grad():
        sample = representer(members).represent(pixel_values)
    assert quantify(sample)["total"].shape == (2, 64, 64)


def test_time_series_model_end_to_end() -> None:
    torch.manual_seed(0)
    config = PatchTSTConfig(
        num_input_channels=1,
        context_length=64,
        prediction_length=16,
        patch_length=8,
        patch_stride=8,
        d_model=16,
        num_attention_heads=2,
        num_hidden_layers=1,
        ffn_dim=32,
        loss="mse",
    )
    model = PatchTSTForPrediction(config)
    past_values = torch.randn(2, 64, 1)

    with torch.no_grad():
        forecast = predict_raw(model, past_values=past_values)
    assert forecast.shape == (2, 16, 1)

    dropout_model = dropout(model, p=0.1)
    dropout_model.eval()
    with torch.no_grad():
        sample = representer(dropout_model, num_samples=3).represent(past_values=past_values)
    assert quantify(sample)["total"].shape == (2, 16, 1)


def test_swag_wrapper_end_to_end(model: BertForSequenceClassification, input_ids: torch.Tensor) -> None:
    from probly.method import swag  # noqa: PLC0415
    from probly.method.swag import collect_swag  # noqa: PLC0415

    swag_model = swag(model, max_rank=3, predictor_type="logit_classifier")
    for _ in range(3):
        with torch.no_grad():
            for param in swag_model.model.parameters():  # ty: ignore[unresolved-attribute]
                param.add_(torch.randn_like(param) * 0.01)
        collect_swag(swag_model)
    swag_model.eval()  # ty: ignore[call-non-callable]

    with torch.no_grad():
        sample = representer(swag_model, num_samples=3).represent(input_ids)
    assert quantify(sample)["total"].shape == (4,)


def test_ensemble_members_differ_and_predict(model: BertForSequenceClassification, input_ids: torch.Tensor) -> None:
    members = ensemble(model, num_members=3, predictor_type="logit_classifier")
    vectors = [torch.nn.utils.parameters_to_vector(member.parameters()) for member in members]
    assert not torch.equal(vectors[0], vectors[1])

    with torch.no_grad():
        sample = representer(members).represent(input_ids)
    decomposition = quantify(sample)
    assert decomposition["total"].shape == (4,)
