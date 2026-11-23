"""E: A1-A3 give the same semantic entropy as the old clustering path, with the same NLI model.

The week-3 before / after table (docs/methods.md) compares week 2 (deberta-base) with week 3
(deberta-large), so a change there can come from the NLI model as well as from the code. This
script keeps the NLI model fixed and changes only the code path, on real answers:

- ``old``:   probly's greedy clusterer as in week 2: one question at a time, every new answer is
             compared with the first answer of each cluster found so far (lazy, no shared matrix).
- ``A1``:    one full NLI matrix per question, every ordered pair, one question per model call.
- ``A1+A2``: the same, but identical texts are scored only once.
- ``A1-A3``: what the benchmark runs: identical texts once and the pairs of 16 questions in one
             shared, length-sorted model batch.

For every path it reports the NLI pairs, the time, and whether clusters / SE / DSE equal ``old``.
``A1`` vs ``A1-A3`` also gives the largest probability difference and label flips over all pairs
that both scored (batching and padding can move the logits a little).

Run from experiments/llm-probly (the NLI model must be downloadable or in the Hugging Face cache):
    uv run python -m scripts.check_nli_equivalence --model qwen0.5b --limit 20
    uv run python -m scripts.check_nli_equivalence --model qwen0.5b --limit 20 --device cpu

Writes ``results/<model>/triviaqa_200/analysis/nli_equivalence.md`` (``--out`` to change).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import TYPE_CHECKING

from common.nli_matrix import QUESTIONS_PER_BATCH, compute_matrices, nli_pair_count, pick_device, question_texts
import numpy as np
import torch

from probly.quantification import entropy
from probly.representation.distribution.torch_sparse_log_categorical import TorchSparseLogCategoricalDistribution
from probly.representer.semantic_clustering.huggingface import NLI_LABELS, HFGreedySemanticClusterer

if TYPE_CHECKING:
    from collections.abc import Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NLI_MODEL = "microsoft/deberta-base-mnli"
PATHS = ("old", "A1", "A1+A2", "A1-A3")
SE_ATOL = 1e-5
_ENTAILMENT = NLI_LABELS.index("entailment")


def se_and_dse_from_clusters(group_ids: torch.Tensor, samples_ll_mean: Sequence[float]) -> tuple[float, float]:
    """Semantic entropy (likelihood weights) and discrete semantic entropy (equal weights) of given clusters."""
    semantic = TorchSparseLogCategoricalDistribution(
        group_ids=group_ids,
        entry_logits=torch.tensor(list(samples_ll_mean), dtype=torch.float32),
    )
    return float(entropy(semantic.to_dense()).item()), float(entropy(semantic.uniform_logits().to_dense()).item())


class PairCounter:
    """Wraps a clusterer's model calls and counts the NLI pairs sent to the model."""

    def __init__(self, clusterer: HFGreedySemanticClusterer) -> None:
        """Patch ``_predict_pair_labels`` / ``_predict_pair_probabilities`` of ``clusterer`` to count pairs."""
        self.pairs = 0
        labels, probabilities = clusterer._predict_pair_labels, clusterer._predict_pair_probabilities  # noqa: SLF001

        def count_labels(statements: np.ndarray, pairs: torch.Tensor) -> torch.Tensor:
            self.pairs += int(pairs.shape[0])
            return labels(statements, pairs)

        def count_probabilities(statements: np.ndarray, pairs: torch.Tensor) -> torch.Tensor:
            self.pairs += int(pairs.shape[0])
            return probabilities(statements, pairs)

        clusterer._predict_pair_labels = count_labels  # noqa: SLF001
        clusterer._predict_pair_probabilities = count_probabilities  # noqa: SLF001

    def take(self) -> int:
        """Return the pairs counted since the last call and reset the counter."""
        pairs, self.pairs = self.pairs, 0
        return pairs


def run_paths(clusterer: HFGreedySemanticClusterer, rows: Sequence[dict]) -> dict[str, dict]:
    """Cluster ``rows`` along every path of :data:`PATHS` with the same NLI model.

    Args:
        clusterer: probly greedy clusterer holding the NLI model.
        rows: Rows of a data cache (``greedy_answer``, ``samples``, ``samples_ll_mean``).

    Returns:
        Per path: ``clusters`` and ``se_dse`` per question, ``pairs`` (sent to the model), ``seconds``,
        and for the matrix paths the ``matrices``.
    """
    counter = PairCounter(clusterer)
    out: dict[str, dict] = {}

    start = time.perf_counter()
    clusters = [clusterer._cluster_row(np.asarray(r["samples"], dtype=object)) for r in rows]  # noqa: SLF001
    out["old"] = {"clusters": clusters, "seconds": time.perf_counter() - start, "pairs": counter.take()}

    def matrix_path(name: str, matrices: list[torch.Tensor], seconds: float) -> None:
        clusters = [HFGreedySemanticClusterer.cluster_nli_matrix(m[1:, 1:]) for m in matrices]
        out[name] = {"clusters": clusters, "matrices": matrices, "seconds": seconds, "pairs": counter.take()}

    start = time.perf_counter()
    matrices = [compute_matrices(clusterer, [r], deduplicate=False)[0] for r in rows]
    matrix_path("A1", matrices, time.perf_counter() - start)

    start = time.perf_counter()
    matrices = [compute_matrices(clusterer, [r], deduplicate=True)[0] for r in rows]
    matrix_path("A1+A2", matrices, time.perf_counter() - start)

    start = time.perf_counter()
    matrices = []
    for first in range(0, len(rows), QUESTIONS_PER_BATCH):
        matrices += compute_matrices(clusterer, rows[first : first + QUESTIONS_PER_BATCH], deduplicate=True)
    matrix_path("A1-A3", matrices, time.perf_counter() - start)

    for result in out.values():
        result["se_dse"] = [
            se_and_dse_from_clusters(c, r["samples_ll_mean"]) for c, r in zip(result["clusters"], rows, strict=True)
        ]
    return out


def compare_matrices(
    reference: Sequence[torch.Tensor], other: Sequence[torch.Tensor], groups: Sequence[Sequence[str]]
) -> dict[str, float]:
    """Compare two sets of question matrices pair by pair.

    Pairs of two different texts are scored by the model in both sets: for them the largest
    probability difference and the label flips are counted. Pairs of identical texts are scored by
    the model only without deduplication (A2 sets them to entailment); for them it counts how often
    ``reference`` does *not* say entailment, i.e. how often A2's shortcut differs from the model.

    Returns:
        ``max_prob_diff``, ``label_flips``, ``pairs`` (different texts), ``identical_pairs`` and
        ``identical_not_entailment``.
    """
    result: dict[str, float] = dict.fromkeys(("label_flips", "pairs", "identical_pairs", "identical_not_entailment"), 0)
    result["max_prob_diff"] = 0.0
    for a, b, texts in zip(reference, other, groups, strict=True):
        n = len(texts)
        same = torch.tensor([[texts[i] == texts[j] for j in range(n)] for i in range(n)])
        scored = ~(torch.isnan(a).any(-1) | torch.isnan(b).any(-1))
        different = scored & ~same
        if different.any():
            result["max_prob_diff"] = max(result["max_prob_diff"], float((a[different] - b[different]).abs().max()))
            result["label_flips"] += int((a[different].argmax(-1) != b[different].argmax(-1)).sum())
            result["pairs"] += int(different.sum())
        identical = scored & same
        result["identical_pairs"] += int(identical.sum())
        result["identical_not_entailment"] += int((a[identical].argmax(-1) != _ENTAILMENT).sum())
    return result


def summarize(results: dict[str, dict], n_questions: int) -> list[dict]:
    """One table row per path: pairs and seconds per question, and agreement with ``old``."""
    old = results["old"]
    table = []
    for name in PATHS:
        result = results[name]
        same_clusters = sum(torch.equal(a, b) for a, b in zip(old["clusters"], result["clusters"], strict=True))
        se_diff = np.abs(np.array(result["se_dse"]) - np.array(old["se_dse"]))
        table.append(
            {
                "path": name,
                "pairs_per_question": result["pairs"] / n_questions,
                "seconds_per_question": result["seconds"] / n_questions,
                "same_clusters": same_clusters,
                "max_se_diff": float(se_diff[:, 0].max()),
                "max_dse_diff": float(se_diff[:, 1].max()),
            }
        )
    return table


def write_report(path: Path, args: argparse.Namespace, device: str, table: list[dict], extra: dict) -> None:
    """Write the markdown table."""
    n = extra["n_questions"]
    lines = [
        f"# NLI code paths with the same NLI model ({args.model}, TriviaQA-200, first {n} questions)",
        "",
        f"NLI model `{args.nli_model}`, device `{device}`, float32. Made by `scripts/check_nli_equivalence.py`.",
        "",
        "| path | NLI pairs / question | seconds / question | clusters = old | max abs SE diff | max abs DSE diff |",
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {r['path']} | {r['pairs_per_question']:.1f} | {r['seconds_per_question']:.3f} | "
        f"{r['same_clusters']}/{n} | {r['max_se_diff']:.1e} | {r['max_dse_diff']:.1e} |"
        for r in table
    ]
    lines += [
        "",
        (
            f"A1 vs A1-A3 over {extra['pairs']} pairs of different texts: largest probability difference "
            f"{extra['max_prob_diff']:.1e}, label flips {extra['label_flips']}. Pairs of identical texts: "
            f"{extra['identical_pairs']}, of which the model (A1) did not say entailment for "
            f"{extra['identical_not_entailment']} (A2 sets them to entailment without asking the model)."
        ),
        "",
        "- `old`: week-2 lazy greedy clustering (only the pairs it needs, but per method and per question).",
        "- `A1`: full matrix per question; costs more pairs than `old` alone, but SE, DSE and the graph",
        "  measures share it (one NLI pass instead of one per method).",
        "- `A1+A2`: identical answers scored once. `A1-A3`: plus 16 questions per model batch (the benchmark).",
        "- Times exclude loading the model; the model is warmed up before the first path.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Parse arguments, run all paths, print and write the table."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen0.5b", help="data/<model>/triviaqa_200.jsonl")
    ap.add_argument("--nli-model", default=DEFAULT_NLI_MODEL)
    ap.add_argument("--limit", type=int, default=20, help="first N questions (0 = all 200)")
    ap.add_argument("--device", default=None, help="cuda / mps / cpu (default: best available)")
    ap.add_argument("--out", default=None, help="markdown path (default: results/<model>/triviaqa_200/analysis/)")
    args = ap.parse_args()

    rows = [json.loads(line) for line in (ROOT / f"data/{args.model}/triviaqa_200.jsonl").open(encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]
    device = args.device or pick_device()
    clusterer = HFGreedySemanticClusterer.from_model_name(args.nli_model)
    clusterer.model.to(device)
    clusterer.model.eval()
    clusterer._cluster_row(np.asarray(["warm", "up"], dtype=object))  # noqa: SLF001  (first call is slow on mps)

    results = run_paths(clusterer, rows)
    table = summarize(results, len(rows))
    groups = [question_texts(r) for r in rows]
    extra = {
        "n_questions": len(rows),
        **compare_matrices(results["A1"]["matrices"], results["A1-A3"]["matrices"], groups),
    }
    expected_pairs = sum(nli_pair_count(r) for r in rows) / len(rows)

    for r in table:
        print(
            f"{r['path']:<7} pairs/q {r['pairs_per_question']:6.1f}  s/q {r['seconds_per_question']:.3f}  "
            f"clusters=old {r['same_clusters']}/{len(rows)}  max|dSE| {r['max_se_diff']:.1e}  "
            f"max|dDSE| {r['max_dse_diff']:.1e}"
        )
    print(
        f"A1 vs A1-A3: max prob diff {extra['max_prob_diff']:.1e}, label flips {extra['label_flips']} "
        f"of {extra['pairs']} pairs; identical texts not entailment {extra['identical_not_entailment']} "
        f"of {extra['identical_pairs']}"
    )
    print(f"(A1+A2 pairs per question expected from nli_pair_count: {expected_pairs:.1f})")

    out = Path(args.out) if args.out else ROOT / f"results/{args.model}/triviaqa_200/analysis/nli_equivalence.md"
    write_report(out, args, device, table, extra)
    print(f"wrote {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")


if __name__ == "__main__":
    main()
