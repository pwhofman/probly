import numpy as np
import torch

from probly.representation.text_generation.torch import (
    TorchTextGeneration,
    TorchTextGenerationSample,
)
from probly.representer.semantic_clustering.huggingface import (
    HFGreedySemanticClusterer,
)
from probly.quantification import entropy

_clusterer = None


def get_clusterer():
    global _clusterer
    if _clusterer is None:
        _clusterer = HFGreedySemanticClusterer.from_model_name(
            "microsoft/deberta-base-mnli"
        )
    return _clusterer


def compute_se_and_dse(row):
    generation = TorchTextGeneration(
        text=np.asarray(row["samples"], dtype=object),
        log_likelihood=torch.tensor(row["samples_ll_mean"], dtype=torch.float32),
    )
    sample = TorchTextGenerationSample(tensor=generation, sample_dim=0)
    semantic = get_clusterer()(sample)

    se = float(entropy(semantic.to_dense()).item())
    dse = float(entropy(semantic.uniform_logits().to_dense()).item())
    return se, dse