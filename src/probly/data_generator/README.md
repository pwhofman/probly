**Gliederung**

1. Einleitung
   - Motivation und Problemstellung
   - Zielsetzung des Projekts
   - Überblick über den Aufbau

2. Theoretischer Hintergrund
   - First-Order Logic im ML-Kontext
   - Wahrscheinlichkeitsverteilungen
   - Existierende Ansätze und Tools

3. Anforderungen und Design
   - Funktionale Anforderungen
   - Use Cases
   - Systemarchitektur
   - API-Design

4. Implementierung
   - Technologie-Stack
   - Modulübersicht
   - Kernkomponenten
     * Generator-Klasse
     * Distribution-Modul
     - Uncertainty-Modul
   - Code-Beispiele

5. Evaluation und Experimente
   - Testmethodik
   - Validierung der generierten Daten
   - Performance-Messungen
   - Beispiel-Anwendungen

6. Ergebnisse
   - Erreichte Funktionalität
   - Visualisierungen
   - Benchmark-Ergebnisse

7. Diskussion
   - Herausforderungen
   - Limitationen
   - Lessons Learned

8. Fazit und Ausblick
   - Zusammenfassung
   - Mögliche Erweiterungen
   - Zukünftige Arbeiten

# First-Order Data Generator

Ein Python-Tool zur Generierung von First-Order Data (Ground-Truth Conditional Distributions) für die Evaluation von Credal Sets und Unsicherheitsquantifizierung in Machine Learning.

##  Überblick

### Was ist First-Order Data?

In der Unsicherheitsquantifizierung mit **Credal Sets** benötigen wir Ground-Truth Conditional Distributions `p(Y|X)` um Coverage-Metriken zu berechnen. Diese Ground-Truth Distributions, auch **First-Order Data** genannt, sind normalerweise nicht direkt verfügbar.

Dieser Generator approximiert `p(Y|X)` durch:
1. Verwendung eines gut vortrainierten Modells (z.B. von Huggingface)
2. Transformation von Samples `x` zu Verteilungen `ĥ(x) ≈ p(·|x)`
3. Speicherung und Verwaltung dieser Verteilungen für Training und Evaluation

### Problem & Lösung

**Problem:** Coverage von Credal Sets messen
- Brauchen: Ground-truth `p(Y|X)`
- Haben: Nur Datenpunkte `(x, y)`

**Lösung:** First-Order Data Generator
- Input: Pretrained Model + Dataset
- Output: Approximierte Conditional Distributions
- Verwendung: Coverage Evaluation, Model Training

## Features

- **Model-Agnostic**: Funktioniert mit beliebigen ML-Modellen (PyTorch, Huggingface, etc.)
- **Distribution Generation**: Erstellt `ĥ(x) ≈ p(Y|x)` für alle Samples
- **Efficient Storage**: Speichert nur Distributions, nicht Outputs
- **Matching System**: Intelligentes Mapping zwischen Distributions und Originaldaten
- **DataLoader Integration**: Nahtlose Integration in PyTorch Workflows
- **Flexible API**: Einfache Verwendung mit Klassen oder Funktionen


##  Quick Start

### Beispiel 1: Classification Task
```python
from probly.first_order_generator import FirstOrderDataGenerator
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# 1. Pretrained Model laden (z.B. von Huggingface)
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Dataset vorbereiten
texts = ["I love this movie", "This is terrible", "Not sure about this"]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# 3. Generator erstellen
generator = FirstOrderDataGenerator(model)

# 4. First-Order Data generieren
first_order_data = generator.generate_distributions(inputs)

# first_order_data enthält jetzt ĥ(x) ≈ p(Y|x) für alle x

# 5. Speichern
generator.save_distributions(
    first_order_data,
    "ground_truth_distributions.json"
)
```

### Beispiel 2: Regression Task
```python
from probly.first_order_generator import FirstOrderDataGenerator
import torch
import torch.nn as nn

# Pretrained Regression Model
model = torch.load("pretrained_regression_model.pt")

# Dataset
dataset = YourDataset()  # Custom dataset
dataloader = DataLoader(dataset, batch_size=32)

# Generator
generator = FirstOrderDataGenerator(model)

# Batch-wise Generation
all_distributions = {}
for batch_idx, (x, y) in enumerate(dataloader):
    batch_distributions = generator.generate_distributions(x)
    all_distributions[f"batch_{batch_idx}"] = batch_distributions

# Speichern mit Matching-Information
generator.save_distributions(
    all_distributions,
    "regression_first_order.json",
    metadata={"dataset": "custom", "model": "regression_v1"}
)
```

### Beispiel 3: Distributions Laden und Matchen
```python
# Laden
loaded = generator.load_distributions("ground_truth_distributions.json")

# Matching mit Original-Daten
for sample_id, distribution in loaded.items():
    original_x = dataset[sample_id]
    # distribution ist ĥ(original_x)
    # Jetzt kann Coverage berechnet werden
```

## Dokumentation

- [User Guide](docs/data_generation_guide.md) - Ausführliche Anleitung
- [API Reference](docs/api_reference.md) - Vollständige API-Dokumentation
- [Examples](examples/) - Beispielskripte und Tutorials
- [Credal Sets Background](docs/credal_sets_background.md) - Theoretischer Hintergrund

## Übersicht

### Hauptklasse: FirstOrderDataGenerator
```python
class FirstOrderDataGenerator:
    def __init__(self, model, device='auto')
    def generate_distributions(self, inputs) -> dict
    def save_distributions(self, data, filepath, metadata=None)
    def load_distributions(self, filepath) -> tuple[dict, dict]
    def create_dataloader(self, distributions, batch_size=32)
```

### Hauptfunktion: generate_first_order_data
```python
def generate_first_order_data(
    model,
    dataset,
    batch_size=32,
    save_path=None
) -> dict
```

Siehe [API Reference](docs/api_reference.md) für Details.

##  Projektstruktur

examples/simple_classification.py
"""
Simple Classification Example: First-Order Data Generator
==========================================================

Dieses Beispiel zeigt wie man First-Order Data für einen
Classification Task generiert.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from probly import FirstOrderDataGenerator
import os


# ===== 1. Dummy Dataset =====
class SimpleDataset(Dataset):
    """Einfaches Classification Dataset."""
    def __init__(self, n_samples=1000, n_features=20, n_classes=5):
        self.n_samples = n_samples
        self.n_classes = n_classes
        # Random Features
        self.X = torch.randn(n_samples, n_features)
        # Random Labels
        self.y = torch.randint(0, n_classes, (n_samples,))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ===== 2. Pretrained Model (Simuliert) =====
class PretrainedClassifier(nn.Module):
    """Simuliert ein gut trainiertes Classification Model."""
    def __init__(self, n_features=20, n_classes=5):
        super().__init__()
        self.fc1 = nn.Linear(n_features, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits  # WICHTIG: Generator wendet Softmax später an


# ===== 3. Hauptfunktion =====
def main():
    print("=" * 70)
    print("First-Order Data Generator - Classification Example")
    print("Demonstriert: Credal Sets & Coverage Evaluation")
    print("=" * 70)

    # Schritt 1: Dataset erstellen
    print("\n[1/7] Dataset erstellen...")
    dataset = SimpleDataset(n_samples=1000, n_features=20, n_classes=5)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    print(f"✓ Dataset: {len(dataset)} samples, 5 classes")

    # Schritt 2: Pretrained Model laden (simuliert)
    print("\n[2/7] Pretrained Model laden...")
    model = PretrainedClassifier(n_features=20, n_classes=5)
    model.eval()  # Evaluation mode
    print("✓ Model geladen (simuliert als pretrained)")

    # Schritt 3: Generator initialisieren
    print("\n[3/7] FirstOrderDataGenerator initialisieren...")
    generator = FirstOrderDataGenerator(
        model=model,
        device='cpu'  # Oder 'cuda' falls GPU verfügbar
    )
    print("✓ Generator bereit")

    # Schritt 4: First-Order Data generieren
    print("\n[4/7] First-Order Data generieren...")
    print("      (Approximiert p(Y|X) mit ĥ(X))")

    all_distributions = {}
    sample_idx = 0

    for batch_x, batch_y in dataloader:
        # Generate distributions für Batch
        with torch.no_grad():
            logits = model(batch_x)
            # WICHTIG: Softmax anwenden!
            batch_distributions = torch.softmax(logits, dim=-1)

        # Speichern mit eindeutigen IDs
        for i in range(len(batch_x)):
            all_distributions[f"sample_{sample_idx:06d}"] = {
                "distribution": batch_distributions[i],
                "true_label": batch_y[i].item()  # Optional: für Validierung
            }
            sample_idx += 1

    print(f"✓ {len(all_distributions)} Distributions generiert")

    # Schritt 5: Distributions analysieren
    print("\n[5/7] Distributions analysieren...")
    example_dist = list(all_distributions.values())[0]["distribution"]
    print(f"      Beispiel Distribution shape: {example_dist.shape}")
    print(f"      Beispiel Distribution: {example_dist.numpy()}")
    print(f"      Sum to 1? {example_dist.sum():.6f}")

    # Schritt 6: Speichern
    print("\n[6/7] First-Order Data speichern...")
    os.makedirs("output", exist_ok=True)

    # Nur Distributions speichern (nicht true labels für echten Gebrauch)
    distributions_only = {
        k: v["distribution"] for k, v in all_distributions.items()
    }

    generator.save_distributions(
        data=distributions_only,
        filepath="output/first_order_classification.json",
        metadata={
            "task": "classification",
            "n_classes": 5,
            "n_samples": len(dataset),
            "model": "PretrainedClassifier"
        }
    )
    print("✓ Gespeichert: output/first_order_classification.json")

    # Schritt 7: Coverage Simulation
    print("\n[7/7] Coverage Evaluation simulieren...")
    print("      (Zeigt wie First-Order Data verwendet wird)")

    # Simuliere ein Credal Set Model
    def simulate_credal_set(x):
        """Simuliert ein Credal Set - Menge von möglichen Distributions."""
        with torch.no_grad():
            logits = model(x)
            base_dist = torch.softmax(logits, dim=-1)

            # Credal Set: Kleine Variationen um base_dist
            credal_set = []
            for _ in range(5):  # 5 Distributions im Set
                noise = torch.randn_like(base_dist) * 0.05
                perturbed = base_dist + noise
                perturbed = torch.clamp(perturbed, min=0)
                perturbed = perturbed / perturbed.sum(dim=-1, keepdim=True)
                credal_set.append(perturbed)

            return credal_set

    # Coverage berechnen
    def compute_coverage(credal_set, ground_truth):
        """Prüft ob ground_truth in credal_set enthalten ist."""
        for dist in credal_set:
            # L1-Distanz
            distance = (dist - ground_truth).abs().sum(dim=-1).item()
            if distance < 0.1:  # Toleranz
                return True
        return False

    # Evaluate Coverage über Dataset
    covered_count = 0
    for sample_id, data in all_distributions.items():
        # Get sample
        sample_idx_int = int(sample_id.split("_")[1])
        x, y = dataset[sample_idx_int]

        # Get ground truth (First-Order Data)
        ground_truth_dist = data["distribution"]

        # Get Credal Set
        credal_set = simulate_credal_set(x.unsqueeze(0))

        # Check Coverage
        is_covered = compute_coverage(credal_set, ground_truth_dist)
        covered_count += int(is_covered)

    coverage_rate = covered_count / len(all_distributions)
    print(f"\n✓ Coverage Rate: {coverage_rate:.2%}")
    print(f"  ({covered_count}/{len(all_distributions)} samples covered)")

    print("\n" + "=" * 70)
    print("✓ Beispiel abgeschlossen!")
    print("\nNext Steps:")
    print("  1. Verwende echtes pretrained Model (z.B. von Huggingface)")
    print("  2. Lade gespeicherte Distributions für Training")
    print("  3. Evaluiere dein Credal Set Model mit diesen Ground-Truths")
    print("=" * 70)


if __name__ == "__main__":
    main()
