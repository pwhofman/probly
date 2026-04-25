"""Quick validation script for River ARF regression integration into probly."""

from river.datasets import synth
from river.forest import ARFRegressor

from probly.quantification import quantify
from probly.representer import representer


def main() -> None:
    """Run a quick validation of the River ARF regression integration."""
    arf = ARFRegressor(n_models=10, seed=42)
    for x, y in synth.Friedman(seed=42).take(500):
        arf.learn_one(x, y)

    sample = representer(arf).represent(x)
    print(f"sample type: {type(sample).__name__}")
    print(f"sample shape: {sample.shape}")

    decomp = quantify(sample)
    print(f"total={decomp.total}  aleatoric={decomp.aleatoric}  epistemic={decomp.epistemic}")


if __name__ == "__main__":
    main()
