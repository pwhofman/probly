# ruff: noqa: INP001
"""Einfaches Beispiel zur Verwendung des First-Order Data Generators.

Dieses Skript zeigt, wie man:
1. Einen einfachen Datensatz und ein Modell erstellt
2. First-Order Verteilungen generiert
3. Diese speichert und lädt
4. Mit FirstOrderDataset und DataLoader verwendet
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Import der First-Order Generator Komponenten
from probly.data_generator.torch_first_order_generator import (
    FirstOrderDataGenerator,
    FirstOrderDataset,
    output_dataloader,
)

# ============================================================================
# 1. DUMMY DATASET UND MODELL ERSTELLEN
# ============================================================================


class SimpleDataset(Dataset):
    """Ein einfacher Beispiel-Datensatz für Demonstrationszwecke.

    Generiert zufällige Eingabevektoren und gibt diese mit Labels zurück.
    """

    def __init__(self, n_samples: int = 100, input_dim: int = 10, n_classes: int = 3) -> None:
        """Initialize dataset.

        Args:
            n_samples: Anzahl der Samples im Datensatz
            input_dim: Dimension der Eingabevektoren
            n_classes: Anzahl der Klassen (für Labels)
        """
        self.n_samples = n_samples
        self.input_dim = input_dim
        self.n_classes = n_classes

        # Generiere zufällige Daten
        torch.manual_seed(42)  # Für Reproduzierbarkeit
        self.data = torch.randn(n_samples, input_dim)
        self.labels = torch.randint(0, n_classes, (n_samples,))

    def __len__(self) -> int:
        """Return dataset length."""
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Get item by index."""
        # Gibt (input, label) zurück
        return self.data[idx], self.labels[idx]


class SimpleModel(nn.Module):
    """Ein einfaches neuronales Netzwerk für Klassifikation."""

    def __init__(self, input_dim: int = 10, n_classes: int = 3) -> None:
        """Initialize model.

        Args:
            input_dim: Dimension der Eingabe
            n_classes: Anzahl der Ausgabeklassen
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Gibt Logits zurück (nicht Wahrscheinlichkeiten!)
        return self.network(x)


# ============================================================================
# 2. VERTEILUNGEN GENERIEREN
# ============================================================================


def generate_first_order_distributions() -> tuple[
    FirstOrderDataGenerator,
    dict[int, list[float]],
    SimpleDataset,
    int,
]:
    """Generiert First-Order Verteilungen aus einem Modell und Datensatz."""
    print("=" * 70)  # noqa: T201
    print("SCHRITT 1: First-Order Verteilungen generieren")  # noqa: T201
    print("=" * 70)  # noqa: T201

    # Parameter definieren
    n_samples = 100
    input_dim = 10
    n_classes = 3

    # Dataset erstellen
    print(f"\nErstelle Datensatz mit {n_samples} Samples...")  # noqa: T201
    dataset = SimpleDataset(n_samples=n_samples, input_dim=input_dim, n_classes=n_classes)

    # Modell erstellen und initialisieren
    print("Erstelle und initialisiere Modell...")  # noqa: T201
    model = SimpleModel(input_dim=input_dim, n_classes=n_classes)
    model.eval()  # WICHTIG: Modell in Evaluationsmodus setzen!

    # Device auswählen
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Verwende Device: {device}")  # noqa: T201

    # Generator initialisieren
    print("\nInitialisiere FirstOrderDataGenerator...")  # noqa: T201
    generator = FirstOrderDataGenerator(
        model=model,
        device=device,
        batch_size=32,
        output_mode="logits",  # Unser Modell gibt Logits aus
        model_name="simple_example_model",
    )

    # Verteilungen generieren
    print("\nGeneriere Verteilungen (mit Fortschrittsanzeige)...")  # noqa: T201
    distributions = generator.generate_distributions(
        dataset,
        progress=True,  # Zeigt Fortschritt an
    )

    print(f"\nErfolgreich {len(distributions)} Verteilungen generiert!")  # noqa: T201

    # Beispiel-Verteilung anzeigen
    print("\nBeispiel-Verteilung für Sample 0:")  # noqa: T201
    print(f"   Wahrscheinlichkeiten: {[f'{p:.4f}' for p in distributions[0]]}")  # noqa: T201
    print(f"   Summe: {sum(distributions[0]):.6f} (sollte ≈ 1.0 sein)")  # noqa: T201

    return generator, distributions, dataset, n_classes


# ============================================================================
# 3. VERTEILUNGEN SPEICHERN UND LADEN
# ============================================================================


def save_and_load_distributions(
    generator: FirstOrderDataGenerator,
    distributions: dict[int, list[float]],
    n_classes: int,
) -> tuple[dict[int, list[float]], dict[str, Any]]:
    """Demonstriert das Speichern und Laden von Verteilungen."""
    print("\n" + "=" * 70)  # noqa: T201
    print("SCHRITT 2: Verteilungen speichern und laden")  # noqa: T201
    print("=" * 70)  # noqa: T201

    # Pfad für Output-Datei
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    save_path = output_dir / "example_first_order_dists.json"

    # Metadaten definieren
    metadata = {
        "dataset": "SimpleDataset",
        "n_samples": len(distributions),
        "n_classes": n_classes,
        "note": "Generated for documentation example",
        "purpose": "demonstration",
    }

    # Speichern
    print(f"\nSpeichere Verteilungen nach: {save_path}")  # noqa: T201
    generator.save_distributions(
        path=save_path,
        distributions=distributions,
        meta=metadata,
    )
    print("Erfolgreich gespeichert!")  # noqa: T201

    # Laden
    print(f"\nLade Verteilungen von: {save_path}")  # noqa: T201
    loaded_distributions, loaded_metadata = generator.load_distributions(save_path)

    print("Erfolgreich geladen!")  # noqa: T201
    print("\nMetadaten:")  # noqa: T201
    for key, value in loaded_metadata.items():
        print(f"   - {key}: {value}")  # noqa: T201

    # Verifizierung
    print("\nVerifizierung:")  # noqa: T201
    print(f"   - Anzahl Verteilungen: {len(loaded_distributions)}")  # noqa: T201
    print(f"   - Originale == Geladene: {distributions == loaded_distributions}")  # noqa: T201

    return loaded_distributions, loaded_metadata


# ============================================================================
# 4. FIRSTORDERDATASET VERWENDEN
# ============================================================================


def use_first_order_dataset(
    dataset: SimpleDataset,
    distributions: dict[int, list[float]],
) -> FirstOrderDataset:
    """Zeigt die Verwendung von FirstOrderDataset."""
    print("\n" + "=" * 70)  # noqa: T201
    print("SCHRITT 3: FirstOrderDataset verwenden")  # noqa: T201
    print("=" * 70)  # noqa: T201

    # FirstOrderDataset erstellen
    print("\nErstelle FirstOrderDataset...")  # noqa: T201
    fo_dataset = FirstOrderDataset(
        base_dataset=dataset,
        distributions=distributions,
    )

    print(f"Dataset erstellt mit {len(fo_dataset)} Samples")  # noqa: T201

    # Ein Sample abrufen
    print("\nSample 0 abrufen:")  # noqa: T201
    sample = fo_dataset[0]

    # Sample kann (input, label, distribution) oder (input, distribution) sein
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
# 5. DATALOADER MIT FIRST-ORDER VERTEILUNGEN
# ============================================================================


def use_dataloader_with_distributions(
    dataset: SimpleDataset,
    distributions: dict[int, list[float]],
) -> DataLoader:
    """Demonstriert die Verwendung eines DataLoaders mit First-Order Verteilungen."""
    print("\n" + "=" * 70)  # noqa: T201
    print("SCHRITT 4: DataLoader mit First-Order Verteilungen")  # noqa: T201
    print("=" * 70)  # noqa: T201

    # DataLoader erstellen
    print("\nErstelle DataLoader mit batch_size=16...")  # noqa: T201
    fo_loader = output_dataloader(
        base_dataset=dataset,
        distributions=distributions,
        batch_size=16,
        shuffle=True,
        num_workers=0,  # Für Windows Kompatibilität
        pin_memory=False,
    )

    print("DataLoader erstellt!")  # noqa: T201

    # Ersten Batch abrufen
    print("\nRufe ersten Batch ab...")  # noqa: T201
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

    # Zeige wie man über den Loader iteriert
    print("\nIteriere über alle Batches...")  # noqa: T201
    for batch_idx, batch in enumerate(fo_loader):
        if batch_idx == 0:
            print(f"   Batch {batch_idx}: {len(batch)} Tensoren")  # noqa: T201

    total_batches = len(fo_loader)
    print(f"   ... (insgesamt {total_batches} Batches)")  # noqa: T201

    return fo_loader


# ============================================================================
# 6. TRAINING MIT SOFT TARGETS (BONUS)
# ============================================================================


def train_with_soft_targets(
    fo_loader: DataLoader,
    input_dim: int,
    n_classes: int,
    epochs: int = 3,
) -> nn.Module:
    """Zeigt ein einfaches Training mit First-Order Verteilungen als Soft Targets."""
    print("\n" + "=" * 70)  # noqa: T201
    print("SCHRITT 5: Training mit Soft Targets (Bonus)")  # noqa: T201
    print("=" * 70)  # noqa: T201

    # Student-Modell erstellen
    print("\nErstelle Student-Modell...")  # noqa: T201
    student_model = SimpleModel(input_dim=input_dim, n_classes=n_classes)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    student_model = student_model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)

    print(f"Training für {epochs} Epochen...")  # noqa: T201

    for epoch in range(epochs):
        student_model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in fo_loader:
            # Batch entpacken
            if len(batch) == 3:
                inputs, _labels, target_distributions = batch
            else:
                inputs, target_distributions = batch

            # Zu Device verschieben
            inputs = inputs.to(device)
            target_distributions = target_distributions.to(device)

            # Forward pass
            logits = student_model(inputs)

            # KL Divergenz Loss zwischen Modell-Ausgabe und Ziel-Verteilungen
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            loss = torch.nn.functional.kl_div(
                log_probs,
                target_distributions,
                reduction="batchmean",
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        print(f"   Epoch {epoch + 1}/{epochs} - Durchschnittlicher Loss: {avg_loss:.4f}")  # noqa: T201

    print("\nTraining abgeschlossen!")  # noqa: T201

    return student_model


# ============================================================================
# MAIN FUNKTION
# ============================================================================


def main() -> None:
    """Hauptfunktion die alle Schritte ausführt."""
    print("\n" + "=" * 70)  # noqa: T201
    print("First-Order Data Generator - Einfaches Beispiel")  # noqa: T201
    print("=" * 70)  # noqa: T201

    # Schritt 1: Verteilungen generieren
    generator, distributions, dataset, n_classes = generate_first_order_distributions()

    # Schritt 2: Speichern und Laden
    loaded_distributions, _metadata = save_and_load_distributions(
        generator,
        distributions,
        n_classes,
    )

    # Schritt 3: FirstOrderDataset verwenden
    _fo_dataset = use_first_order_dataset(dataset, loaded_distributions)

    # Schritt 4: DataLoader verwenden
    fo_loader = use_dataloader_with_distributions(dataset, loaded_distributions)

    # Schritt 5: Training (optional)
    print("\n" + "=" * 70)  # noqa: T201
    print("Möchten Sie ein kurzes Training demonstrieren? (Optional)")  # noqa: T201
    print("=" * 70)  # noqa: T201
    print("Hinweis: Dies zeigt, wie man mit den generierten Verteilungen")  # noqa: T201
    print("         als 'Soft Targets' trainieren kann.")  # noqa: T201

    # Für das Beispiel trainieren wir einfach
    _student_model = train_with_soft_targets(
        fo_loader,
        input_dim=10,
        n_classes=n_classes,
        epochs=2,
    )

    print("\n" + "=" * 70)  # noqa: T201
    print("Beispiel erfolgreich abgeschlossen!")  # noqa: T201
    print("=" * 70)  # noqa: T201
    print("\nZusammenfassung:")  # noqa: T201
    print(f"  {len(distributions)} Verteilungen generiert")  # noqa: T201
    print("  Verteilungen gespeichert und geladen")  # noqa: T201
    print("  FirstOrderDataset erstellt")  # noqa: T201
    print("  DataLoader verwendet")  # noqa: T201
    print("  Student-Modell trainiert")  # noqa: T201
    print("\nWeitere Informationen: Siehe docs/data_generation_guide.md")  # noqa: T201
    print("=" * 70 + "\n")  # noqa: T201


if __name__ == "__main__":
    main()
