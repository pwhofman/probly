# First-Order Data Generator - Benutzerhandbuch

## Überblick

Der **First-Order Data Generator** ermöglicht es, approximative Wahrscheinlichkeitsverteilungen (First-Order Data) aus einem vortrainierten Modell und einem Datensatz zu generieren. Diese Verteilungen können verwendet werden, um Unsicherheitsquantifizierung und Coverage-Metriken für Credal Sets zu evaluieren.

## Grundkonzept

In maschinellem Lernen arbeiten wir normalerweise mit der gemeinsamen Verteilung p(X,Y), wobei X die Eingabefeatures und Y die Zielvariablen darstellen. Die bedingte Verteilung p(Y|X) ist oft nicht direkt zugänglich. Der First-Order Data Generator approximiert diese Verteilung durch:

```
ĥ(x) ≈ p(·|x)
```

wobei ĥ ein vortrainiertes Modell ist (z.B. von Huggingface).

## Installation

Der Generator ist Teil des `probly` Pakets. Stelle sicher, dass PyTorch installiert ist:

```bash
pip install torch
```

## Hauptkomponenten

### 1. FirstOrderDataGenerator

Die Hauptklasse zur Generierung von First-Order Verteilungen.

**Parameter:**
- `model`: Ein aufrufbares Objekt (meist `torch.nn.Module`), das Eingaben auf Logits oder Wahrscheinlichkeiten abbildet
- `device`: Gerät für Inferenz (z.B. 'cpu' oder 'cuda'), Standard: 'cpu'
- `batch_size`: Batch-Größe beim Verarbeiten des Datensatzes, Standard: 64
- `output_mode`: Einer von {'auto', 'logits', 'probs'}, Standard: 'auto'
  - `'auto'`: Erkennt automatisch, ob Ausgaben Logits oder Wahrscheinlichkeiten sind
  - `'logits'`: Wendet Softmax auf die Modellausgaben an
  - `'probs'`: Verwendet die Ausgaben direkt als Wahrscheinlichkeiten
- `output_transform`: Optionale benutzerdefinierte Funktion zur Konvertierung von Modellausgaben (überschreibt `output_mode`)
- `input_getter`: Optionale Funktion zum Extrahieren der Modelleingabe aus Dataset-Elementen
- `model_name`: Optionaler String-Identifier (wird mit Metadaten gespeichert)

### 2. FirstOrderDataset

Ein PyTorch Dataset-Wrapper, der einen bestehenden Datensatz mit First-Order Verteilungen kombiniert.

### 3. output_fo_dataloader

Eine Hilfsfunktion zum Erstellen eines DataLoaders mit First-Order Verteilungen.

## Grundlegende Verwendung

### Schritt 1: Verteilungen generieren

```python
import torch
from torch.utils.data import Dataset
from probly.data_generator.first_order_generator import FirstOrderDataGenerator

# Ihr vortrainiertes Modell
model = torch.load('my_pretrained_model.pt')
model.eval()

# Ihr Datensatz
dataset = MyDataset()  # Ihre PyTorch Dataset-Implementierung

# Generator initialisieren
generator = FirstOrderDataGenerator(
    model=model,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    batch_size=32,
    output_mode='logits',  # Falls Ihr Modell Logits ausgibt
    model_name='my_model_v1'
)

# Verteilungen generieren
distributions = generator.generate_distributions(
    dataset,
    progress=True  # Zeigt Fortschritt an
)

# distributions ist ein dict: {index: [prob_class_0, prob_class_1, ...]}
```

### Schritt 2: Verteilungen speichern

```python
# Speichern mit Metadaten
generator.save_distributions(
    path='output/first_order_dists.json',
    distributions=distributions,
    meta={
        'dataset': 'CIFAR-10',
        'num_classes': 10,
        'note': 'Generated with ResNet-50'
    }
)
```

### Schritt 3: Verteilungen laden und verwenden

```python
# Verteilungen laden
loaded_dists, metadata = generator.load_distributions('output/first_order_dists.json')

print(f"Model: {metadata['model_name']}")
print(f"Dataset: {metadata['dataset']}")
print(f"Number of samples: {len(loaded_dists)}")

# Mit FirstOrderDataset verwenden
from probly.data_generator.first_order_generator import FirstOrderDataset

fo_dataset = FirstOrderDataset(
    base_dataset=dataset,
    distributions=loaded_dists
)

# Element abrufen
# Falls base_dataset (input, label) zurückgibt:
input_tensor, label, distribution = fo_dataset[0]

# Falls base_dataset nur input zurückgibt:
input_tensor, distribution = fo_dataset[0]
```

### Schritt 4: DataLoader mit First-Order Verteilungen erstellen

```python
from probly.data_generator.first_order_generator import output_fo_dataloader

# DataLoader erstellen
fo_loader = output_fo_dataloader(
    base_dataset=dataset,
    distributions=loaded_dists,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# Training mit weichen Zielen (Soft Targets)
for batch in fo_loader:
    if len(batch) == 3:  # Mit Labels
        inputs, labels, distributions = batch
    else:  # Ohne Labels
        inputs, distributions = batch
    
    # Ihr Trainingsschritt hier...
    logits = student_model(inputs)
    loss = kl_divergence(logits, distributions)
    loss.backward()
```

## Erweiterte Verwendung

### Benutzerdefinierte Input-Extraktion

Falls Ihr Dataset eine komplexere Struktur hat:

```python
def custom_input_getter(sample):
    # sample könnte ein Dict sein: {'image': tensor, 'metadata': dict, 'label': int}
    return sample['image']

generator = FirstOrderDataGenerator(
    model=model,
    input_getter=custom_input_getter,
    device='cpu'
)
```

### Benutzerdefinierte Output-Transformation

```python
def custom_transform(outputs):
    # Eigene Logik zur Konvertierung von Modellausgaben
    return torch.softmax(outputs, dim=-1) * 0.9 + 0.1 / outputs.shape[-1]

generator = FirstOrderDataGenerator(
    model=model,
    output_transform=custom_transform,
    device='cpu'
)
```

### Verwendung mit DataLoader statt Dataset

```python
from torch.utils.data import DataLoader

# Sie können auch direkt einen DataLoader übergeben
custom_loader = DataLoader(dataset, batch_size=16, shuffle=False)

distributions = generator.generate_distributions(custom_loader, progress=True)
```

## Best Practices

1. **Modell im Evaluationsmodus**: Stellen Sie sicher, dass Ihr Modell im Evaluationsmodus ist (`model.eval()`)

2. **Konsistente Indizierung**: Die generierten Verteilungen verwenden die Dataset-Indizes. Achten Sie darauf, dass Ihr Dataset konsistente Indizes hat.

3. **Gerätekompatibilität**: Verwenden Sie die gleiche Gerätekonfiguration für Generation und Training.

4. **Metadaten speichern**: Fügen Sie immer relevante Metadaten beim Speichern hinzu, um später nachvollziehen zu können, wie die Verteilungen generiert wurden.

5. **Speicherverbrauch**: Bei großen Datensätzen können die gespeicherten JSON-Dateien groß werden. Planen Sie entsprechend Speicherplatz ein.

## Fehlerbehebung

### Problem: "Model must return a torch.Tensor"
**Lösung**: Ihr Modell muss einen `torch.Tensor` zurückgeben. Falls es andere Strukturen zurückgibt, verwenden Sie `output_transform`.

### Problem: Warnung über unterschiedliche Längen
**Lösung**: Dies kann bei DataLoadern mit `drop_last=True` auftreten. Das ist normalerweise kein Problem, solange die Hauptdaten vollständig sind.

### Problem: Speicher-Fehler bei großen Datensätzen
**Lösung**: Reduzieren Sie die `batch_size` oder verwenden Sie gradient checkpointing.

### Problem: Verteilungen summieren sich nicht zu 1
**Lösung**: Stellen Sie sicher, dass `output_mode` korrekt konfiguriert ist. Bei Logits sollte 'logits' verwendet werden, bei bereits normalisierten Ausgaben 'probs'.

## Beispieldaten

Die generierten JSON-Dateien haben folgende Struktur:

```json
{
  "meta": {
    "model_name": "my_model_v1",
    "dataset": "CIFAR-10",
    "num_classes": 10
  },
  "distributions": {
    "0": [0.1, 0.2, 0.05, 0.15, 0.3, 0.05, 0.05, 0.05, 0.025, 0.025],
    "1": [0.05, 0.8, 0.03, 0.02, 0.05, 0.01, 0.01, 0.01, 0.01, 0.01],
    ...
  }
}
```

## API-Referenz

### FirstOrderDataGenerator.generate_distributions()

```python
def generate_distributions(
    self,
    dataset_or_loader: Dataset | DataLoader,
    *,
    progress: bool = True
) -> dict[int, list[float]]
```

Generiert Wahrscheinlichkeitsverteilungen für alle Samples im Datensatz.

**Returns:** Dictionary mit Dataset-Index als Schlüssel und Wahrscheinlichkeitsliste als Wert.

### FirstOrderDataGenerator.save_distributions()

```python
def save_distributions(
    self,
    path: str | Path,
    distributions: Mapping[int, Iterable[float]],
    *,
    meta: dict[str, Any] | None = None
) -> None
```

Speichert Verteilungen und Metadaten als JSON-Datei.

### FirstOrderDataGenerator.load_distributions()

```python
def load_distributions(
    self, 
    path: str | Path
) -> tuple[dict[int, list[float]], dict[str, Any]]
```

Lädt Verteilungen und Metadaten aus einer JSON-Datei.

**Returns:** Tuple aus (distributions, metadata).

## Weitere Ressourcen

- Siehe `examples/simple_usage.py` für ein vollständiges Beispiel
- Tests unter `tests/test_first_order_generator.py` zeigen weitere Anwendungsfälle
- Bei Fragen oder Problemen: Erstellen Sie ein Issue im Repository

## Lizenz und Kontakt

Teil des `probly` Projekts. Weitere Informationen finden Sie in der Haupt-README des Projekts.