"""Shared sanity test for reflexive uncertainty methods (R1, R2, R3).

Each question is asked twice: once with the correct answer and once with a
deliberately wrong one. A method that works should give clearly higher
confidence to the correct answers.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

SANITY = [
    ("What is the capital of France?", "Paris", "Berlin"),
    ("What is the chemical symbol for water?", "H2O", "CO2"),
    ("How many planets are in our solar system?", "8", "12"),
    ("Who wrote Romeo and Juliet?", "William Shakespeare", "Charles Dickens"),
    ("What is the largest ocean on Earth?", "Pacific Ocean", "Atlantic Ocean"),
    ("At what temperature in Celsius does water boil at sea level?", "100", "50"),
    ("Who was the first person to walk on the Moon?", "Neil Armstrong", "Yuri Gagarin"),
    ("What is the currency of Japan?", "Yen", "Won"),
    ("How many legs does a spider have?", "8", "6"),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci", "Pablo Picasso"),
]


def run_sanity(confidence_fn, samples=None):
    """Run the sanity test.

    Args:
        confidence_fn: function (question, answer, samples) -> confidence in [0, 1].
        samples: optional dict {question: list of sampled answers}, for brainstorm methods.

    Returns:
        dict with auroc (correct = positive), gap (mean confidence correct - wrong),
        n_distinct (number of distinct confidence values) and all confidences.
    """
    conf, label = [], []
    for question, right, wrong in SANITY:
        s = samples.get(question) if samples else None
        conf.append(confidence_fn(question, right, s))
        label.append(1)
        conf.append(confidence_fn(question, wrong, s))
        label.append(0)
    conf = np.array(conf, dtype=float)
    label = np.array(label)
    return {
        "auroc": float(roc_auc_score(label, conf)),
        "gap": float(conf[label == 1].mean() - conf[label == 0].mean()),
        "n_distinct": int(len(np.unique(np.round(conf, 3)))),
        "confidences": conf.tolist(),
    }


def save_sanity(result, method, model_name, folder="results/sanity"):
    """Save a sanity test result as results/sanity/<method>_<model>.json."""
    Path(folder).mkdir(parents=True, exist_ok=True)
    short = model_name.split("/")[-1]
    path = Path(folder) / f"{method}_{short}.json"
    with open(path, "w") as f:
        json.dump({"method": method, "model": model_name, **result}, f, indent=2)
    return path