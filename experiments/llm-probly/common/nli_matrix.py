"""Shared pairwise NLI matrix per question (A1).

One matrix per question holds the NLI probabilities for every ordered pair of
``[greedy_answer, sample_1, ..., sample_K]``. Semantic entropy, discrete semantic
entropy and the graph measures all read from it, so the NLI model runs once per
question instead of once per method.

The matrix itself comes from probly
(``HFSemanticClusterer.pairwise_nli``, last axis ordered contradiction, neutral,
entailment). This module only adds what the experiments need on top: which texts
go in, SE / discrete SE from a matrix, and the ``.nli.npz`` cache file format.

Cache file format (``data/<model>/<dataset>_<n>.nli.npz``, version 1)::

    nli                   float32 (N, K+1, K+1, 3)  index 0 = greedy answer, NaN = diagonal/padding
    n_texts               int16   (N,)              1 + number of samples per question
    ids                   int64   (N,)              question ids, same order as the data cache
    labels                str     (3,)              order of the last axis of ``nli`` in the file
    nli_model             str     ()                Hugging Face name of the NLI model
    use_question          bool    ()                True if the question was put in front of each text
    seconds_per_question  float   ()                NLI time when the file was built

The file stores the last axis as ``labels`` says (entail, neutral, contradict,
as in the week-2 files). :func:`load_nli_cache` always returns probly's order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from probly.quantification import entropy
from probly.representation.distribution.torch_sparse_log_categorical import TorchSparseLogCategoricalDistribution
from probly.representer.semantic_clustering.huggingface import NLI_LABELS, HFGreedySemanticClusterer

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from probly.representer.semantic_clustering.huggingface import HFSemanticClusterer

FORMAT_VERSION = 1
GREEDY_INDEX = 0
FILE_LABELS = ("entail", "neutral", "contradict")
"""Order of the last axis inside ``.nli.npz`` files (kept from week 2 so old readers keep working)."""


def question_texts(row: dict) -> list[str]:
    """Return the texts that go into the NLI matrix of one question.

    Args:
        row: One row of the data cache with ``greedy_answer`` and ``samples``.

    Returns:
        ``[greedy_answer, sample_1, ..., sample_K]``; index 0 is the greedy answer.
    """
    return [row["greedy_answer"], *row["samples"]]


def compute_matrix(
    clusterer: HFSemanticClusterer,
    row: dict,
    *,
    use_question: bool = False,
) -> torch.Tensor:
    """Compute the NLI matrix of one question with probly's clusterer.

    Args:
        clusterer: probly semantic clusterer that holds the NLI model.
        row: One row of the data cache.
        use_question: If True, the question is put in front of every text.

    Returns:
        Tensor of shape ``(K+1, K+1, 3)`` in probly's label order (``NLI_LABELS``).
    """
    context = row["question"] if use_question else None
    return clusterer.pairwise_nli(question_texts(row), context=context)


def sample_block(matrix: torch.Tensor) -> torch.Tensor:
    """Return the samples-vs-samples part of a question matrix (drops the greedy answer)."""
    return matrix[GREEDY_INDEX + 1 :, GREEDY_INDEX + 1 :]


def se_and_dse(matrix: torch.Tensor, samples_ll_mean: Sequence[float]) -> tuple[float, float]:
    """Semantic entropy and discrete semantic entropy from one question matrix.

    Clusters the K samples with probly's greedy rule on the shared matrix, then
    computes the entropy of the cluster distribution weighted by the sample
    likelihoods (SE) and with equal weights (discrete SE), exactly as the
    clusterer path did before.

    Args:
        matrix: Question matrix of shape ``(K+1, K+1, 3)`` in probly's label order.
        samples_ll_mean: Mean token log-likelihood of each of the K samples.

    Returns:
        ``(semantic_entropy, discrete_semantic_entropy)`` in nats.
    """
    samples = sample_block(matrix)
    if samples.shape[0] != len(samples_ll_mean):
        msg = f"matrix has {samples.shape[0]} samples but {len(samples_ll_mean)} likelihoods were given."
        raise ValueError(msg)
    group_ids = HFGreedySemanticClusterer.cluster_nli_matrix(samples)
    semantic = TorchSparseLogCategoricalDistribution(
        group_ids=group_ids,
        entry_logits=torch.tensor(list(samples_ll_mean), dtype=torch.float32),
    )
    se = float(entropy(semantic.to_dense()).item())
    dse = float(entropy(semantic.uniform_logits().to_dense()).item())
    return se, dse


def _file_to_probly_order(labels: Sequence[str]) -> list[int]:
    """Column indices that turn the file's label order into probly's order."""
    names = [str(label).lower() for label in labels]
    columns = []
    for wanted in NLI_LABELS:
        matches = [i for i, name in enumerate(names) if name.startswith(wanted[:6])]
        if len(matches) != 1:
            msg = f"cannot find label {wanted!r} in cache labels {names}."
            raise ValueError(msg)
        columns.append(matches[0])
    return columns


def save_nli_cache(
    path: str | Path,
    matrices: Sequence[torch.Tensor],
    ids: Sequence[int],
    *,
    nli_model: str,
    use_question: bool,
    seconds_per_question: float,
) -> None:
    """Save question matrices (probly order) as a ``.nli.npz`` file.

    Args:
        path: Output file.
        matrices: One ``(n_i, n_i, 3)`` matrix per question in probly's label order.
        ids: Question ids, same order as ``matrices``.
        nli_model: Name of the NLI model that produced the matrices.
        use_question: Whether the question was put in front of each text.
        seconds_per_question: Measured NLI time per question.
    """
    n_texts = np.array([m.shape[0] for m in matrices], dtype=np.int16)
    size = int(n_texts.max()) if len(matrices) else 0
    big = np.full((len(matrices), size, size, 3), np.nan, dtype=np.float32)
    to_file = [NLI_LABELS.index(name) for name in ("entailment", "neutral", "contradiction")]
    for i, matrix in enumerate(matrices):
        n = matrix.shape[0]
        big[i, :n, :n] = matrix.numpy()[..., to_file]
    np.savez_compressed(
        path,
        nli=big,
        n_texts=n_texts,
        ids=np.asarray(ids),
        labels=np.array(FILE_LABELS),
        nli_model=np.array(nli_model),
        use_question=np.array(use_question),
        seconds_per_question=np.array(seconds_per_question),
        format_version=np.array(FORMAT_VERSION),
    )


def load_nli_cache(path: str | Path) -> dict:
    """Load a ``.nli.npz`` file with the matrices in probly's label order.

    Args:
        path: The cache file (week-2 files without ``format_version`` load too).

    Returns:
        Dict with ``matrices`` (list of ``(n_i, n_i, 3)`` float32 tensors, probly order),
        ``ids``, ``nli_model``, ``use_question`` and ``seconds_per_question``.
    """
    with np.load(path, allow_pickle=False) as z:
        columns = _file_to_probly_order(z["labels"].tolist())
        big = z["nli"][..., columns]
        n_texts = z["n_texts"]
        return {
            "matrices": [torch.from_numpy(big[i, :n, :n].copy()) for i, n in enumerate(n_texts)],
            "ids": z["ids"].tolist(),
            "nli_model": str(z["nli_model"]),
            "use_question": bool(z["use_question"]),
            "seconds_per_question": float(z["seconds_per_question"]),
        }
