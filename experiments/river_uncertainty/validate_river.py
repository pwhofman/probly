"""Quick validation script for the River ARF integration into probly."""

from river.forest import ARFClassifier
from river.datasets import synth
from probly.representer import representer
from probly.quantification import quantify

def main() -> None:
    """Run a quick validation of the River ARF integration."""
    arf = ARFClassifier(n_models=10, seed=42)
    for x, y in synth.Agrawal(seed=42).take(500):
        arf.learn_one(x, y)

    sample = representer(arf).represent(x)
    decomp = quantify(sample)
    print(f"total={decomp.total:.4f}  aleatoric={decomp.aleatoric:.4f}  epistemic={decomp.epistemic:.4f}")

if __name__ == '__main__':
    main()
