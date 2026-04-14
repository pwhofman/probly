"""NLI-based entailment model for semantic equivalence checking."""

from __future__ import annotations

from typing import Literal

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from core.paths import CACHE_DIR

NLIModel = Literal["microsoft/deberta-base-mnli", "microsoft/deberta-v2-xlarge-mnli"]
DEFAULT_NLI_MODEL: NLIModel = "microsoft/deberta-base-mnli"


class EntailmentModel:
    """NLI model for checking semantic entailment between text pairs."""

    def __init__(self, model_name: NLIModel = DEFAULT_NLI_MODEL, device: str | None = None) -> None:
        """Load the NLI model and tokenizer."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=CACHE_DIR).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def check_implication(self, premise: str, hypothesis: str) -> int:
        """Check if premise entails hypothesis.

        Returns:
            0 = contradiction, 1 = neutral, 2 = entailment
        """
        inputs = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512).to(
            self.device
        )
        logits = self.model(**inputs).logits
        return torch.argmax(logits, dim=-1).item()
