# Gemma 4 Experiment

Semantic entropy, spectral uncertainty, and calibration experiments with Google Gemma 4 using probly.

## Setup

```bash
cd experiments/gemma
uv sync
```

Models are cached in `data/model_cache/` at the project root.
Results are written to `data/results/experiments/gemma/`.

## Download model

```bash
uv run python experiments/download.py
```

## Interactive chat

```bash
uv run python experiments/chat.py
uv run python experiments/chat.py --temperature 0.7 --top-k 40
```

## Semantic entropy

Measures how semantically diverse the model's responses are across multiple samples.

```bash
uv run python experiments/semantic_entropy.py --num-samples 10 --seed 42
uv run python experiments/semantic_entropy.py --nli-model microsoft/deberta-v2-xlarge-mnli
```

## Spectral uncertainty

Computes Von Neumann entropy from kernel matrices built on sentence embeddings
(Walha et al., 2025). Runs side-by-side with semantic entropy for comparison and
decomposes total uncertainty into aleatoric and epistemic components via
clarification-based two-stage sampling.

```bash
uv run python experiments/spectral_uncertainty.py --num-samples 10 --seed 42
uv run python experiments/spectral_uncertainty.py --gamma 2.0 --num-clarifications 8
```

## Calibration experiment

Full calibration pipeline with TriviaQA:

```bash
# Generate responses and compute entropy
uv run python experiments/calibration/generate.py --num-questions 200 --num-samples 10

# Run all temperature configurations
caffeinate -i uv run python experiments/calibration/run_all.py

# Analyze results (after manual correctness review)
uv run python experiments/calibration/analyze.py --results data/results/experiments/gemma/run.json
```
