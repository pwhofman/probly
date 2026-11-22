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
    deduplicate           bool    ()                True if identical texts were scored once (A2); missing = False
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


QUESTIONS_PER_BATCH = 16
"""Questions whose NLI pairs share model batches (A3); a question has at most 110 pairs."""


def pick_device() -> str:
    """Return ``"cuda"``, ``"mps"`` or ``"cpu"``, whichever is available."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(name: str, device: str) -> torch.dtype:
    """Map ``"float32"`` / ``"float16"`` to a torch dtype; float16 only on a GPU.

    Args:
        name: ``"float32"`` or ``"float16"``.
        device: ``"cuda"``, ``"mps"`` or ``"cpu"``.

    Returns:
        The torch dtype for the NLI model.

    Raises:
        ValueError: For an unknown name, or float16 on the CPU (slow and not supported by every op).
    """
    dtypes = {"float32": torch.float32, "float16": torch.float16}
    if name not in dtypes:
        msg = f"dtype must be one of {sorted(dtypes)}, got {name!r}."
        raise ValueError(msg)
    if dtypes[name] == torch.float16 and device == "cpu":
        msg = "float16 NLI needs a GPU (cuda or mps); use float32 on the CPU."
        raise ValueError(msg)
    return dtypes[name]


def compute_matrices(
    clusterer: HFSemanticClusterer,
    rows: Sequence[dict],
    *,
    use_question: bool = False,
    deduplicate: bool = True,
) -> list[torch.Tensor]:
    """Compute the NLI matrices of several questions in shared model batches (A3).

    Args:
        clusterer: probly semantic clusterer that holds the NLI model.
        rows: Rows of the data cache.
        use_question: If True, the question is put in front of every text.
        deduplicate: If True, identical texts of a question are sent to the NLI model only once (A2).

    Returns:
        One ``(K+1, K+1, 3)`` tensor per row in probly's label order (``NLI_LABELS``).
    """
    contexts = [row["question"] if use_question else None for row in rows]
    groups = [question_texts(row) for row in rows]
    return clusterer.pairwise_nli_many(groups, contexts=contexts, deduplicate=deduplicate)


def compute_matrix(
    clusterer: HFSemanticClusterer,
    row: dict,
    *,
    use_question: bool = False,
    deduplicate: bool = True,
) -> torch.Tensor:
    """Compute the NLI matrix of one question; see :func:`compute_matrices`."""
    return compute_matrices(clusterer, [row], use_question=use_question, deduplicate=deduplicate)[0]


def nli_pair_count(row: dict, *, deduplicate: bool = True) -> int:
    """Number of ordered pairs the NLI model scores for one question.

    Args:
        row: One row of the data cache.
        deduplicate: Count distinct texts only, as :func:`compute_matrix` does with ``deduplicate=True``.

    Returns:
        ``m * (m - 1)`` with ``m`` the number of (distinct) texts of the question.
    """
    texts = question_texts(row)
    m = len(set(texts)) if deduplicate else len(texts)
    return m * (m - 1)


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
    deduplicate: bool = True,
) -> None:
    """Save question matrices (probly order) as a ``.nli.npz`` file.

    Args:
        path: Output file.
        matrices: One ``(n_i, n_i, 3)`` matrix per question in probly's label order.
        ids: Question ids, same order as ``matrices``.
        nli_model: Name of the NLI model that produced the matrices.
        use_question: Whether the question was put in front of each text.
        seconds_per_question: Measured NLI time per question.
        deduplicate: Whether identical texts were scored once.
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
        deduplicate=np.array(deduplicate),
        format_version=np.array(FORMAT_VERSION),
    )


def load_nli_cache(path: str | Path) -> dict:
    """Load a ``.nli.npz`` file with the matrices in probly's label order.

    Args:
        path: The cache file (week-2 files without ``format_version`` load too).

    Returns:
        Dict with ``matrices`` (list of ``(n_i, n_i, 3)`` float32 tensors, probly order),
        ``ids``, ``nli_model``, ``use_question``, ``seconds_per_question`` and ``deduplicate``.
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
            "deduplicate": bool(z["deduplicate"]) if "deduplicate" in z.files else False,
        }
