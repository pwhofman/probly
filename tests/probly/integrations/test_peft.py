"""Tests for optional PEFT bindings."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
from peft import LoraConfig, get_peft_model
import torch
from transformers import BertConfig, BertForSequenceClassification

from probly.method import ensemble
from probly.predictor import predict_raw
from probly.quantification import quantify
from probly.representer import representer


@pytest.fixture
def lora_model():
    torch.manual_seed(0)
    config = BertConfig(
        vocab_size=50,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=32,
    )
    config.num_labels = 3
    base = BertForSequenceClassification(config)
    return get_peft_model(base, LoraConfig(task_type="SEQ_CLS", r=2, target_modules=["query", "value"]))


@pytest.fixture
def input_ids() -> torch.Tensor:
    return torch.randint(0, 50, (4, 10))


def test_predict_raw_returns_logits(lora_model, input_ids: torch.Tensor) -> None:
    with torch.no_grad():
        out = predict_raw(lora_model, input_ids)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (4, 3)


def test_only_adapter_parameters_are_trainable(lora_model) -> None:
    trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in lora_model.parameters())
    assert 0 < trainable < total


def test_adapter_ensemble_end_to_end(lora_model, input_ids: torch.Tensor) -> None:
    members = ensemble(lora_model, num_members=2, reset_params=False, predictor_type="logit_classifier")
    with torch.no_grad():
        sample = representer(members).represent(input_ids)
    assert quantify(sample)["total"].shape == (4,)
