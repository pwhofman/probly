"""Simple example for using the First-Order Data Generator.

This script demonstrates how to:
1. Create a simple dataset and model
2. Generate first-order distributions
3. Save and load them
4. Use them with FirstOrderDataset and DataLoader
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Import First-Order Generator components
from probly.data_generator.first_order_generator import FirstOrderDataGenerator, FirstOrderDataset, output_fo_dataloader

# ============================================================================
# 1. CREATE DUMMY DATASET AND MODEL
# ============================================================================


class SimpleDataset(Dataset):
    """A simple example dataset for demonstration purposes.

    Generates random input vectors and returns them with labels.
    """

    def __init__(self, n_samples: int = 100, input_dim: int = 10, n_classes: int = 3) -> None:
        """Initialize dataset.

        Args:
            n_samples: Number of samples in the dataset
            input_dim: Dimension of input vectors
            n_classes: Number of classes (for labels)
        """
        self.n_samples = n_samples
        self.input_dim = input_dim
        self.n_classes = n_classes

        # Generate random data
        torch.manual_seed(42)  # For reproducibility
        self.data = torch.randn(n_samples, input_dim)
        self.labels = torch.randint(0, n_classes, (n_samples,))

    def __len__(self) -> int:
        """Return dataset length."""
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Get item by index.

        Returns:
            tuple: (input tensor, label)
        """
        return self.data[idx], self.labels[idx]


class SimpleModel(nn.Module):
    """A simple neural network for classification."""

    def __init__(self, input_dim: int = 10, n_classes: int = 3) -> None:
        """Initialize model.

        Args:
            input_dim: Input dimension
            n_classes: Number of output classes
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Returns:
            torch.Tensor: Logits (not probabilities)
        """
        return self.network(x)


# ============================================================================
# 2. GENERATE DISTRIBUTIONS
# ============================================================================


def generate_first_order_distributions() -> tuple[FirstOrderDataGenerator, dict[int, list[float]], SimpleDataset, int]:
    """Generate first-order distributions from a model and dataset."""
    print("=" * 70)  # noqa: T201
    print("STEP 1: Generate first-order distributions")  # noqa: T201
    print("=" * 70)  # noqa: T201

    # Define parameters
    n_samples = 100
    input_dim = 10
    n_classes = 3

    # Create dataset
    print(f"\nCreating dataset with {n_samples} samples...")  # noqa: T201
    dataset = SimpleDataset(n_samples=n_samples, input_dim=input_dim, n_classes=n_classes)

    # Create and initialize model
    print("Creating and initializing model...")  # noqa: T201
    model = SimpleModel(input_dim=input_dim, n_classes=n_classes)
    model.eval()  # IMPORTANT: Set model to evaluation mode!

    # Select device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")  # noqa: T201

    # Initialize generator
    print("\nInitializing FirstOrderDataGenerator...")  # noqa: T201
    generator = FirstOrderDataGenerator(
        model=model,
        device=device,
        batch_size=32,
        output_mode="logits",  # Our model outputs logits
        model_name="simple_example_model",
    )

    # Generate distributions
    print("\nGenerating distributions (with progress bar)...")  # noqa: T201
    distributions = generator.generate_distributions(
        dataset,
        progress=True,  # Show progress
    )

    print(f"\nSuccessfully generated {len(distributions)} distributions!")  # noqa: T201

    # Display example distribution
    print("\nExample distribution for sample 0:")  # noqa: T201
    print(f"   Probabilities: {[f'{p:.4f}' for p in distributions[0]]}")  # noqa: T201
    print(f"   Sum: {sum(distributions[0]):.6f} (should be ≈ 1.0)")  # noqa: T201

    return generator, distributions, dataset, n_classes


# ============================================================================
# 3. SAVE AND LOAD DISTRIBUTIONS
# ============================================================================


def save_and_load_distributions(
    generator: FirstOrderDataGenerator, distributions: dict[int, list[float]], n_classes: int
) -> tuple[dict[int, list[float]], dict[str, Any]]:
    """Demonstrate saving and loading distributions."""
    print("\n" + "=" * 70)  # noqa: T201
    print("STEP 2: Save and load distributions")  # noqa: T201
    print("=" * 70)  # noqa: T201

    # Path for output file
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    save_path = output_dir / "example_first_order_dists.json"

    # Define metadata
    metadata = {
        "dataset": "SimpleDataset",
        "n_samples": len(distributions),
        "n_classes": n_classes,
        "note": "Generated for documentation example",
        "purpose": "demonstration",
    }

    # Save
    print(f"\nSaving distributions to: {save_path}")  # noqa: T201
    generator.save_distributions(
        path=save_path,
        distributions=distributions,
        meta=metadata,
    )
    print("Successfully saved!")  # noqa: T201

    # Load
    print(f"\nLoading distributions from: {save_path}")  # noqa: T201
    loaded_distributions, loaded_metadata = generator.load_distributions(save_path)

    print("Successfully loaded!")  # noqa: T201
    print("\nMetadata:")  # noqa: T201
    for key, value in loaded_metadata.items():
        print(f"   - {key}: {value}")  # noqa: T201

    # Verification
    print("\nVerification:")  # noqa: T201
    print(f"   - Number of distributions: {len(loaded_distributions)}")  # noqa: T201
    print(f"   - Original == Loaded: {distributions == loaded_distributions}")  # noqa: T201

    return loaded_distributions, loaded_metadata


# ============================================================================
# 4. USE FIRSTORDERDATASET
# ============================================================================


def use_first_order_dataset(dataset: SimpleDataset, distributions: dict[int, list[float]]) -> FirstOrderDataset:
    """Show how to use FirstOrderDataset."""
    print("\n" + "=" * 70)  # noqa: T201
    print("STEP 3: Use FirstOrderDataset")  # noqa: T201
    print("=" * 70)  # noqa: T201

    # Create FirstOrderDataset
    print("\nCreating FirstOrderDataset...")  # noqa: T201
    fo_dataset = FirstOrderDataset(base_dataset=dataset, distributions=distributions)

    print(f"Dataset created with {len(fo_dataset)} samples")  # noqa: T201

    # Retrieve a sample
    print("\nRetrieving sample 0:")  # noqa: T201
    sample = fo_dataset[0]

    # Sample can be (input, label, distribution) or (input, distribution)
    if len(sample) == 3:
        input_tensor, label, distribution = sample
        print(f"   - Input shape: {input_tensor.shape}")  # noqa: T201
        print(f"   - Label: {label}")  # noqa: T201
        print(f"   - Distribution shape: {distribution.shape}")  # noqa: T201
        print(f"   - Distribution: {[f'{p:.4f}' for p in distribution.tolist()]}")  # noqa: T201
    else:
        input_tensor, distribution = sample
        print(f"   - Input shape: {input_tensor.shape}")  # noqa: T201
        print(f"   - Distribution shape: {distribution.shape}")  # noqa: T201
        print(f"   - Distribution: {[f'{p:.4f}' for p in distribution.tolist()]}")  # noqa: T201

    return fo_dataset


# ============================================================================
# 5. DATALOADER WITH FIRST-ORDER DISTRIBUTIONS
# ============================================================================


def use_dataloader_with_distributions(dataset: SimpleDataset, distributions: dict[int, list[float]]) -> DataLoader:
    """Demonstrate using a DataLoader with first-order distributions."""
    print("\n" + "=" * 70)  # noqa: T201
    print("STEP 4: DataLoader with first-order distributions")  # noqa: T201
    print("=" * 70)  # noqa: T201

    # Create DataLoader
    print("\nCreating DataLoader with batch_size=16...")  # noqa: T201
    fo_loader = output_fo_dataloader(
        base_dataset=dataset,
        distributions=distributions,
        batch_size=16,
        shuffle=True,
        num_workers=0,  # For Windows compatibility
        pin_memory=False,
    )

    print("DataLoader created!")  # noqa: T201

    # Get first batch
    print("\nRetrieving first batch...")  # noqa: T201
    batch = next(iter(fo_loader))

    if len(batch) == 3:
        inputs, labels, distributions_batch = batch
        print(f"   - Inputs shape: {inputs.shape}")  # noqa: T201
        print(f"   - Labels shape: {labels.shape}")  # noqa: T201
        print(f"   - Distributions shape: {distributions_batch.shape}")  # noqa: T201
    else:
        inputs, distributions_batch = batch
        print(f"   - Inputs shape: {inputs.shape}")  # noqa: T201
        print(f"   - Distributions shape: {distributions_batch.shape}")  # noqa: T201

    # Show how to iterate over the loader
    print("\nIterating over all batches...")  # noqa: T201
    for batch_idx, batch in enumerate(fo_loader):
        if batch_idx == 0:
            print(f"   Batch {batch_idx}: {len(batch)} tensors")  # noqa: T201

    total_batches = len(fo_loader)
    print(f"   ... (total {total_batches} batches)")  # noqa: T201

    return fo_loader


# ============================================================================
# 6. TRAINING WITH SOFT TARGETS (BONUS)
# ============================================================================


def train_with_soft_targets(fo_loader: DataLoader, input_dim: int, n_classes: int, epochs: int = 3) -> nn.Module:
    """Show simple training with first-order distributions as soft targets."""
    print("\n" + "=" * 70)  # noqa: T201
    print("STEP 5: Training with soft targets (bonus)")  # noqa: T201
    print("=" * 70)  # noqa: T201

    # Create student model
    print("\nCreating student model...")  # noqa: T201
    student_model = SimpleModel(input_dim=input_dim, n_classes=n_classes)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    student_model = student_model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)

    print(f"Training for {epochs} epochs...")  # noqa: T201

    for epoch in range(epochs):
        student_model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in fo_loader:
            # Unpack batch
            if len(batch) == 3:
                inputs, _labels, target_distributions = batch
            else:
                inputs, target_distributions = batch

            # Move to device
            inputs = inputs.to(device)
            target_distributions = target_distributions.to(device)

            # Forward pass
            logits = student_model(inputs)

            # KL Divergence loss between model output and target distributions
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            loss = torch.nn.functional.kl_div(log_probs, target_distributions, reduction="batchmean")

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        print(f"   Epoch {epoch + 1}/{epochs} - Average loss: {avg_loss:.4f}")  # noqa: T201

    print("\nTraining completed!")  # noqa: T201

    return student_model


# ============================================================================
# MAIN FUNCTION
# ============================================================================


def main() -> None:
    """Main function that executes all steps."""
    print("\n" + "=" * 70)  # noqa: T201
    print("First-Order Data Generator - Simple Example")  # noqa: T201
    print("=" * 70)  # noqa: T201

    # Step 1: Generate distributions
    generator, distributions, dataset, n_classes = generate_first_order_distributions()

    # Step 2: Save and load
    loaded_distributions, _metadata = save_and_load_distributions(generator, distributions, n_classes)

    # Step 3: Use FirstOrderDataset
    _fo_dataset = use_first_order_dataset(dataset, loaded_distributions)

    # Step 4: Use DataLoader
    fo_loader = use_dataloader_with_distributions(dataset, loaded_distributions)

    # Step 5: Training (optional)
    print("\n" + "=" * 70)  # noqa: T201
    print("Would you like to demonstrate a short training session? (Optional)")  # noqa: T201
    print("=" * 70)  # noqa: T201
    print("Note: This demonstrates how to train with the generated distributions")  # noqa: T201
    print("      as 'soft targets'.")  # noqa: T201

    # For this example, we simply train
    _student_model = train_with_soft_targets(fo_loader, input_dim=10, n_classes=n_classes, epochs=2)

    print("\n" + "=" * 70)  # noqa: T201
    print("Example completed successfully!")  # noqa: T201
    print("=" * 70)  # noqa: T201
    print("\nSummary:")  # noqa: T201
    print(f"  {len(distributions)} distributions generated")  # noqa: T201
    print("  Distributions saved and loaded")  # noqa: T201
    print("  FirstOrderDataset created")  # noqa: T201
    print("  DataLoader used")  # noqa: T201
    print("  Student model trained")  # noqa: T201
    print("\nFor more information: See docs/data_generation_guide.md")  # noqa: T201
    print("=" * 70 + "\n")  # noqa: T201


if __name__ == "__main__":
    main()
