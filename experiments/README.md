# Experiments

Practical experiments showcasing `probly` in ML applications. Run everything from the `experiments/` directory.

## Setup

```bash
cd experiments
uv sync
```

Models are cached in `data/model_cache/` at the project root.

## Gemma 4

### Download model

```bash
uv run python gemma/download.py
```

### Interactive chat

```bash
uv run python gemma/chat.py
uv run python gemma/chat.py --temperature 0.7 --top-k 40
```

### Semantic entropy

Measures how semantically diverse the model's responses are across multiple samples. High entropy = uncertain, low entropy = confident.

```bash
uv run python gemma/semantic_entropy/run.py --num-samples 10 --seed 42
uv run python gemma/semantic_entropy/run.py --nli-model microsoft/deberta-v2-xlarge-mnli
```
