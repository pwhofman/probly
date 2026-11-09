"""Tests for common/reflexive.py with a fake tokenizer and model (no download needed)."""

import math
from types import SimpleNamespace

import pytest
import torch

from common import reflexive


class FakeTokenizer:
    """Knows only the strings in ``vocab``; every other string is split into 2 tokens."""

    def __init__(self, vocab):
        self.vocab = vocab

    def encode(self, text, add_special_tokens=False):
        return [self.vocab[text]] if text in self.vocab else [98, 99]

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        return FakeEncoding(input_ids=torch.zeros(1, 3, dtype=torch.long))


class FakeEncoding(dict):
    def to(self, device):
        return self


class FakeModel:
    """Returns the same next-token logits for every prompt."""

    device = "cpu"

    def __init__(self, logits):
        self.logits = torch.tensor(logits).view(1, 1, -1)

    def __call__(self, **kwargs):
        return SimpleNamespace(logits=self.logits)


LOGITS = [2.0, 0.0, -1.0, -1.0]  # id 0 = "A", id 1 = "B"


def test_p_true_plain_ab():
    tok = FakeTokenizer({"A": 0, "B": 1})
    d = reflexive.p_true_details(FakeModel(LOGITS), tok, "q", "a", mode="plain_ab")
    assert d["p_true"] == pytest.approx(1 / (1 + math.exp(-2.0)))  # sigmoid(2 - 0)
    assert 0.0 < d["coverage"] <= 1.0


@pytest.mark.parametrize("vocab", [{"B": 1}, {"A": 0}, {}])
def test_empty_group_raises(vocab):
    tok = FakeTokenizer(vocab)
    with pytest.raises(ValueError, match="empty token group"):
        reflexive.p_true(FakeModel(LOGITS), tok, "q", "a", mode="plain_ab")