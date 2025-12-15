# First-Order Data Generator - README

##  Überblick

Der **First-Order Data Generator** ist ein Werkzeug zur Generierung approximativer bedingter Wahrscheinlichkeitsverteilungen p(Y|X) aus vortrainierten Modellen. Diese Verteilungen werden als "First-Order Data" bezeichnet und sind essenziell für die Evaluierung von Unsicherheitsquantifizierungsmethoden, insbesondere Credal Sets.

##  Zweck

In Machine Learning haben wir normalerweise keinen direkten Zugang zur wahren bedingten Verteilung p(Y|X). Der First-Order Data Generator löst dieses Problem, indem er:

1. Ein vortrainiertes Modell ĥ als Approximation verwendet
2. Für jedes Sample x die Verteilung ĥ(x) ≈ p(·|x) berechnet
3. Diese Verteilungen persistent speichert
4. Einfache Integration in PyTorch-Workflows ermöglicht



### Einfachstes Beispiel (5 Zeilen)

```python
from probly.data_generator.first_order_generator import FirstOrderDataGenerator

# Generator mit Ihrem Modell und Dataset
generator = FirstOrderDataGenerator(model=your_model, device='cpu')
distributions = generator.generate_distributions(your_dataset)
generator.save_distributions('output.json', distributions)
```

### Vollständiges Beispiel

Siehe `examples/simple_usage.py` für ein ausführbares, kommentiertes Beispiel.

```bash
python examples/simple_usage.py
```

##  Hauptfunktionen

### 1. Verteilungen generieren

```python
from probly.data_generator.first_order_generator import FirstOrderDataGenerator

generator = FirstOrderDataGenerator(
    model=pretrained_model,        # Ihr PyTorch Modell
    device='cuda',                  # 'cuda' oder 'cpu'
    batch_size=32,                  # Batch-Größe für Inferenz
    output_mode='logits',           # 'logits', 'probs' oder 'auto'
    model_name='resnet50_v1'        # Optionaler Identifier
)

distributions = generator.generate_distributions(
    dataset,           # PyTorch Dataset oder DataLoader
    progress=True      # Fortschrittsanzeige aktivieren
)

# Ergebnis: dict[int, list[float]]
# {0: [0.1, 0.3, 0.6], 1: [0.2, 0.5, 0.3], ...}
```

### 2. Verteilungen speichern

```python
generator.save_distributions(
    path='first_order_data.json',
    distributions=distributions,
    meta={
        'dataset': 'MNIST',
        'model_architecture': 'ResNet-50',
        'timestamp': '2024-12-15',
        'accuracy': 0.95
    }
)
```

**Dateiformat:**
```json
{
  "meta": {
    "model_name": "resnet50_v1",
    "dataset": "MNIST",
    "model_architecture": "ResNet-50"
  },
  "distributions": {
    "0": [0.1, 0.3, 0.6],
    "1": [0.2, 0.5, 0.3]
  }
}
```

### 3. Verteilungen laden

```python
loaded_dists, metadata = generator.load_distributions('first_order_data.json')

print(f"Modell: {metadata['model_name']}")
print(f"Anzahl Samples: {len(loaded_dists)}")
```

### 4. Mit Dataset/DataLoader verwenden

```python
from probly.data_generator.first_order_generator import (
    FirstOrderDataset,
    output_fo_dataloader
)

# Option 1: FirstOrderDataset
fo_dataset = FirstOrderDataset(
    base_dataset=original_dataset,
    distributions=loaded_dists
)

# Gibt (input, label, distribution) oder (input, distribution) zurück
input, label, dist = fo_dataset[0]

# Option 2: Direkt DataLoader erstellen
fo_loader = output_fo_dataloader(
    base_dataset=original_dataset,
    distributions=loaded_dists,
    batch_size=32,
    shuffle=True
)

for batch in fo_loader:
    inputs, labels, distributions = batch
    # Ihr Training hier...
```

### 5. Training mit Soft Targets

```python
import torch.nn.functional as F

for inputs, labels, target_distributions in fo_loader:
    # Forward pass
    logits = student_model(inputs)
    
    # KL Divergenz zwischen Prediction und First-Order Distribution
    loss = F.kl_div(
        F.log_softmax(logits, dim=-1),
        target_distributions,
        reduction='batchmean'
    )
    
    # Backward pass
    loss.backward()
    optimizer.step()
```

##  Konfigurationsoptionen

### FirstOrderDataGenerator Parameter

| Parameter | Typ | Standard | Beschreibung |
|-----------|-----|----------|--------------|
| `model` | Callable | **Erforderlich** | PyTorch Modell oder aufrufbares Objekt |
| `device` | str | `'cpu'` | Device: `'cpu'` oder `'cuda'` |
| `batch_size` | int | `64` | Batch-Größe für Inferenz |
| `output_mode` | str | `'auto'` | `'auto'`, `'logits'` oder `'probs'` |
| `output_transform` | Callable\|None | `None` | Benutzerdefinierte Transformationsfunktion |
| `input_getter` | Callable\|None | `None` | Funktion zum Extrahieren von Inputs |
| `model_name` | str\|None | `None` | Identifier für Metadaten |

### output_mode Erklärung

- **`'auto'`**: Erkennt automatisch, ob Ausgaben Logits oder Probs sind
- **`'logits'`**: Wendet Softmax auf Modellausgaben an
- **`'probs'`**: Verwendet Ausgaben direkt (keine Transformation)

##  Erweiterte Verwendung

### Benutzerdefinierte Input-Extraktion

Für komplexe Dataset-Strukturen:

```python
def extract_image_from_dict(sample):
    # Sample ist z.B.: {'image': tensor, 'metadata': {...}, 'label': int}
    return sample['image']

generator = FirstOrderDataGenerator(
    model=model,
    input_getter=extract_image_from_dict
)
```

### Benutzerdefinierte Output-Transformation

```python
def custom_output_transform(logits):
    # Label Smoothing: (1-α) * softmax + α * uniform
    probs = torch.softmax(logits, dim=-1)
    alpha = 0.1
    n_classes = logits.shape[-1]
    return (1 - alpha) * probs + alpha / n_classes

generator = FirstOrderDataGenerator(
    model=model,
    output_transform=custom_output_transform
)
```

### Verwendung mit bestehenden DataLoadern

```python
# Sie können auch DataLoader direkt übergeben
your_loader = DataLoader(dataset, batch_size=16, shuffle=False)
distributions = generator.generate_distributions(your_loader)
```

##  Anwendungsfälle

### 1. Credal Set Evaluation

```python
# Generiere Teacher-Verteilungen
teacher_dists = generator.generate_distributions(test_set)

# Trainiere Student mit Credal Sets
student_credal_sets = train_credal_model(fo_loader)

# Berechne Coverage
coverage = compute_coverage(teacher_dists, student_credal_sets)
```

### 2. Knowledge Distillation

```python
# Lehrer-Verteilungen
teacher_gen = FirstOrderDataGenerator(model=large_teacher_model)
teacher_dists = teacher_gen.generate_distributions(dataset)

# Schüler trainieren
student_loader = output_fo_dataloader(dataset, teacher_dists, batch_size=64)
train_student_with_kd(student_model, student_loader)
```

### 3. Uncertainty Quantification

```python
# Verteilungen von mehreren Modellen
ensemble_dists = []
for model in ensemble_models:
    gen = FirstOrderDataGenerator(model=model)
    dists = gen.generate_distributions(dataset)
    ensemble_dists.append(dists)

# Analysiere Disagreement
uncertainty = compute_ensemble_uncertainty(ensemble_dists)
```

## ⚠️ Wichtige Hinweise

###  Best Practices

1. **Evaluationsmodus**: Setzen Sie Ihr Modell immer in `eval()` Modus
   ```python
   model.eval()
   ```

2. **Konsistente Indizierung**: Verwenden Sie `shuffle=False` beim Generieren
   ```python
   loader = DataLoader(dataset, batch_size=32, shuffle=False)
   ```

3. **Metadaten speichern**: Dokumentieren Sie Ihre Verteilungen
   ```python
   meta = {
       'model': 'ResNet-50',
       'dataset': 'CIFAR-10',
       'date': '2024-12-15',
       'accuracy': 0.95,
       'notes': 'Pre-trained on ImageNet'
   }
   ```

4. **Speicherplatz**: JSON-Dateien können groß werden (ca. 1 MB pro 10,000 Samples mit 10 Klassen)

### ⚡ Performance-Tipps

- Verwenden Sie GPU wenn verfügbar: `device='cuda'`
- Wählen Sie optimale Batch-Größe (größer = schneller, aber mehr Speicher)
- Verwenden Sie `num_workers > 0` in DataLoadern (außer auf Windows)
- Bei sehr großen Datasets: Generieren Sie in Chunks

### 🐛 Häufige Probleme

**Problem**: "Model must return a torch.Tensor"
```python
# Lösung: Verwenden Sie output_transform
def extract_tensor(output):
    return output['logits']  # Falls Model Dict zurückgibt

generator = FirstOrderDataGenerator(
    model=model,
    output_transform=extract_tensor
)
```

**Problem**: Warnung über unterschiedliche Längen
```python
# Das ist meist OK - passiert bei drop_last=True
# Falls es ein Problem ist, prüfen Sie Ihre DataLoader-Konfiguration
```

**Problem**: Verteilungen summieren nicht zu 1.0
```python
# Lösung: Stellen Sie output_mode korrekt ein
generator = FirstOrderDataGenerator(
    model=model,
    output_mode='logits'  # Falls Ihr Modell Logits ausgibt
)
```

##  Projektstruktur

```
probly/
├── data_generator/
│   ├── first_order_generator.py    # Hauptimplementierung
│   ├── base_generator.py           # Basis-Klasse (falls verwendet)
│   └── first_order_dists.json      # Beispiel-Output
├── examples/
│   └── simple_usage.py             # Vollständiges Beispiel
├── docs/
│   └── data_generation_guide.md    # Detaillierte Dokumentation
└── tests/
    └── test_first_order_generator.py  # Unit Tests
```

##  Tests ausführen

```bash
# Alle Tests
pytest tests/test_first_order_generator.py

# Spezifischer Test
pytest tests/test_first_order_generator.py::test_save_and_load_with_meta

# Mit Ausgabe
pytest tests/test_first_order_generator.py -v -s
```

##  Weitere Dokumentation

- **Detaillierte Anleitung**: `docs/data_generation_guide.md`
- **API-Referenz**: Docstrings in `first_order_generator.py`
- **Beispiele**: `examples/simple_usage.py`
- **Tests**: `tests/test_first_order_generator.py`

##  Kontakt und Support

Bei Fragen, Problemen oder Verbesserungsvorschlägen:

1. Erstellen Sie ein Issue im Repository
2. Konsultieren Sie die detaillierte Dokumentation
3. Schauen Sie sich die Tests für weitere Verwendungsbeispiele an

##  Lizenz

Teil des `probly` Projekts - siehe Haupt-Repository für Lizenzinformationen.

---

**Hinweis**: Diese README deckt die Grundlagen ab. Für tiefergehende Informationen, siehe `docs/data_generation_guide.md`.