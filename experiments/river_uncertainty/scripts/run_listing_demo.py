"""Verbatim runnable version of the ~5-line listing in paper §5.3."""

from river.datasets import synth
from river.forest import ARFClassifier

from probly.quantification import quantify
from probly.representer import representer

arf = ARFClassifier(n_models=10, seed=42)
decomp = None
for x, y in synth.STAGGER(seed=42).take(3000):
    arf.learn_one(x, y)
    decomp = quantify(representer(arf).represent(x))
    # decomp.total / .aleatoric / .epistemic — drift signal in .epistemic
assert decomp is not None
print(f"final epistemic: {decomp.epistemic:.4f}")
