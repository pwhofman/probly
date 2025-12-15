# First-Order Data Generator - API Referenz

## Inhaltsverzeichnis
- [FirstOrderDataGenerator](#firstorderdatagenerator)
- [FirstOrderDataset](#firstorderdataset)
- [output_fo_dataloader](#output_fo_dataloader)
- [Hilfsfunktionen](#hilfsfunktionen)

---

## FirstOrderDataGenerator

### Klassenbeschreibung

```python
@dataclass
class FirstOrderDataGenerator:
    """Generator für First-Order Wahrscheinlichkeitsverteilungen."""
```

Hauptklasse zur Generierung approximativer bedingter Wahrscheinlichkeitsverteilungen p(Y|X) aus vortrainierten Modellen.

### Konstruktor-Parameter

| Parameter | Typ | Standard | Beschreibung |
|-----------|-----|----------|--------------|
| `model` | `Callable[..., Any]` | **Erforderlich** | Aufrufbares Objekt (meist `torch.nn.Module`), das Eingaben auf Logits oder Wahrscheinlichkeiten abbildet |
| `device` | `str` | `"cpu"` | Gerät für Inferenz (`"cpu"` oder `"cuda"`) |
| `batch_size` | `int` | `64` | Batch-Größe für die Verarbeitung |
| `output_mode` | `str` | `"auto"` | Output-Modus: `"auto"`, `"logits"` oder `"probs"` |
| `output_transform` | `Callable[[torch.Tensor], torch.Tensor] \| None` | `None` | Benutzerdefinierte Funktion zur Konvertierung von Modellausgaben |
| `input_getter` | `Callable[[Any], Any] \| None` | `None` | Funktion zum Extrahieren der Modelleingabe aus Dataset-Elementen |
| `model_name` | `str \| None` | `None` | Optionaler String-Identifier (wird mit Metadaten gespeichert) |

### Methoden

#### `generate_distributions()`

```python
@torch.no_grad()
def generate_distributions(
    self,
    dataset_or_loader: object,
    *,
    progress: bool = True,
) -> dict[int, list[float]]
```

Generiert Wahrscheinlichkeitsverteilungen für alle Samples im Datensatz.

**Parameter:**
- `dataset_or_loader`: `torch.utils.data.Dataset` oder `torch.utils.data.DataLoader`
- `progress`: Bool, ob Fortschritt angezeigt werden soll

**Rückgabe:**
- `dict[int, list[float]]`: Mapping von Dataset-Index zu Wahrscheinlichkeitsliste

**Raises:**
- `TypeError`: Falls Modell keinen `torch.Tensor` zurückgibt

**Beispiel:**
```python
generator = FirstOrderDataGenerator(model=model, device='cuda')
distributions = generator.generate_distributions(dataset, progress=True)
```

#### `save_distributions()`

```python
def save_distributions(
    self,
    path: str | Path,
    distributions: Mapping[int, Iterable[float]],
    *,
    meta: dict[str, Any] | None = None,
) -> None
```

Speichert Verteilungen und Metadaten als JSON-Datei.

**Parameter:**
- `path`: Zielpfad für JSON-Datei
- `distributions`: Zu speichernde Verteilungen
- `meta`: Optionale Metadaten (als Dictionary)

**Raises:**
- `IOError`: Bei Schreibfehlern

**Beispiel:**
```python
generator.save_distributions(
    'output/dists.json',
    distributions,
    meta={'dataset': 'MNIST', 'accuracy': 0.95}
)
```

#### `load_distributions()`

```python
def load_distributions(
    self, 
    path: str | Path
) -> tuple[dict[int, list[float]], dict[str, Any]]
```

Lädt Verteilungen und Metadaten aus JSON-Datei.

**Parameter:**
- `path`: Quellpfad der JSON-Datei

**Rückgabe:**
- `tuple`: `(distributions, metadata)`
  - `distributions`: `dict[int, list[float]]` - Die Verteilungen
  - `metadata`: `dict[str, Any]` - Gespeicherte Metadaten

**Raises:**
- `FileNotFoundError`: Falls Datei nicht existiert
- `json.JSONDecodeError`: Bei ungültigem JSON

**Beispiel:**
```python
dists, meta = generator.load_distributions('output/dists.json')
print(f"Modell: {meta['model_name']}")
print(f"Samples: {len(dists)}")
```

#### `to_device()`

```python
def to_device(self, x: object) -> object
```

Verschiebt Tensor(en) auf das konfigurierte Gerät. Unterstützt verschachtelte Strukturen.

**Parameter:**
- `x`: Tensor oder verschachtelte Struktur von Tensoren

**Rückgabe:**
- `object`: Eingabe mit Tensoren auf Ziel-Gerät

#### `to_probs()`

```python
def to_probs(self, outputs: torch.Tensor) -> torch.Tensor
```

Konvertiert Modellausgaben zu Wahrscheinlichkeiten.

**Parameter:**
- `outputs`: Modellausgaben (Logits oder Probs)

**Rückgabe:**
- `torch.Tensor`: Wahrscheinlichkeitsverteilungen (summieren zu 1)

---

## FirstOrderDataset

### Klassenbeschreibung

```python
class FirstOrderDataset(Dataset):
    """PyTorch Dataset-Wrapper für First-Order Verteilungen."""
```

Kombiniert einen existierenden PyTorch Dataset mit First-Order Verteilungen.

### Konstruktor-Parameter

| Parameter | Typ | Beschreibung |
|-----------|-----|--------------|
| `base_dataset` | `Dataset` | Ursprünglicher PyTorch Dataset |
| `distributions` | `Mapping[int, Iterable[float]]` | Index-aligned Verteilungen |
| `input_getter` | `Callable[[object], object] \| None` | Optional: Funktion für Input-Extraktion |

### Methoden

#### `__len__()`

```python
def __len__(self) -> int
```

Gibt Anzahl der Samples im Dataset zurück.

**Rückgabe:**
- `int`: Anzahl der Samples

#### `__getitem__()`

```python
def __getitem__(self, idx: int) -> object
```

Gibt Input (+ optional Label) und Verteilung bei Index zurück.

**Parameter:**
- `idx`: Index des Samples

**Rückgabe:**
- `tuple`: `(input, distribution)` oder `(input, label, distribution)`

**Raises:**
- `KeyError`: Falls keine Verteilung für Index existiert

**Beispiel:**
```python
fo_dataset = FirstOrderDataset(base_dataset, distributions)

# Mit Labels
input, label, dist = fo_dataset[0]

# Oder ohne Labels
input, dist = fo_dataset[0]
```

---

## output_fo_dataloader

### Funktionsbeschreibung

```python
def output_fo_dataloader(
    base_dataset: Dataset,
    distributions: Mapping[int, Iterable[float]],
    *,
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    input_getter: Callable[[Any], Any] | None = None,
) -> DataLoader
```

Erstellt einen PyTorch DataLoader, der Inputs (+ Labels) mit First-Order Verteilungen paart.

### Parameter

| Parameter | Typ | Standard | Beschreibung |
|-----------|-----|----------|--------------|
| `base_dataset` | `Dataset` | **Erforderlich** | Ursprünglicher PyTorch Dataset |
| `distributions` | `Mapping[int, Iterable[float]]` | **Erforderlich** | Index-aligned Verteilungen |
| `batch_size` | `int` | `64` | Batch-Größe |
| `shuffle` | `bool` | `False` | Ob Daten gemischt werden sollen |
| `num_workers` | `int` | `0` | Anzahl Worker-Prozesse |
| `pin_memory` | `bool` | `False` | Ob Memory Pinning verwendet werden soll |
| `input_getter` | `Callable \| None` | `None` | Optional: Funktion für Input-Extraktion |

### Rückgabe

- `DataLoader`: PyTorch DataLoader mit First-Order Verteilungen

### Beispiel

```python
# Einfache Verwendung
fo_loader = output_fo_dataloader(
    base_dataset=dataset,
    distributions=distributions,
    batch_size=32
)

for batch in fo_loader:
    if len(batch) == 3:
        inputs, labels, dists = batch
    else:
        inputs, dists = batch
    
    # Training hier...
```

---

## Hilfsfunktionen

### `_is_probabilities()`

```python
def _is_probabilities(x: torch.Tensor, atol: float = 1e-4) -> bool
```

**Interne Funktion** - Prüft, ob ein Tensor Wahrscheinlichkeiten entlang der letzten Dimension darstellt.

**Parameter:**
- `x`: Input Tensor
- `atol`: Absolute Toleranz für Summierung

**Rückgabe:**
- `bool`: True falls Tensor Wahrscheinlichkeiten repräsentiert

---

## Datentypen

### Distribution Dict

```python
dict[int, list[float]]
```

Mapping von Dataset-Index (int) zu Wahrscheinlichkeitsliste (list[float]).

**Beispiel:**
```python
{
    0: [0.1, 0.3, 0.6],      # Sample 0: Klasse 0: 10%, Klasse 1: 30%, Klasse 2: 60%
    1: [0.2, 0.5, 0.3],      # Sample 1: Klasse 0: 20%, Klasse 1: 50%, Klasse 2: 30%
    ...
}
```

### Metadata Dict

```python
dict[str, Any]
```

Beliebige Metadaten für gespeicherte Verteilungen.

**Typische Felder:**
```python
{
    'model_name': str,           # Name des Modells
    'dataset': str,              # Name des Datensatzes
    'num_classes': int,          # Anzahl Klassen
    'accuracy': float,           # Modell-Genauigkeit
    'timestamp': str,            # Zeitstempel
    'note': str                  # Zusätzliche Notizen
}
```

---

## JSON-Dateiformat

### Struktur

```json
{
  "meta": {
    "model_name": "resnet50_v1",
    "dataset": "MNIST",
    "num_classes": 10,
    "accuracy": 0.95,
    "timestamp": "2024-12-15"
  },
  "distributions": {
    "0": [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.17, 0.13, 0.10, 0.07],
    "1": [0.03, 0.05, 0.08, 0.12, 0.18, 0.21, 0.15, 0.10, 0.06, 0.02],
    ...
  }
}
```

### Eigenschaften

- **Encoding**: UTF-8
- **Format**: JSON mit 2-space Einrückung
- **Keys in `distributions`**: Strings (automatisch von int konvertiert)
- **Values in `distributions`**: Listen von Floats (Wahrscheinlichkeiten)

---

## Typische Workflows

### Workflow 1: Generieren und Speichern

```python
# 1. Generator erstellen
generator = FirstOrderDataGenerator(
    model=pretrained_model,
    device='cuda',
    output_mode='logits'
)

# 2. Verteilungen generieren
distributions = generator.generate_distributions(dataset)

# 3. Speichern mit Metadaten
generator.save_distributions(
    'output/dists.json',
    distributions,
    meta={'dataset': 'CIFAR-10', 'accuracy': 0.92}
)
```

### Workflow 2: Laden und Verwenden

```python
# 1. Laden
generator = FirstOrderDataGenerator(model=dummy_model)
distributions, metadata = generator.load_distributions('output/dists.json')

# 2. DataLoader erstellen
fo_loader = output_fo_dataloader(
    dataset,
    distributions,
    batch_size=32,
    shuffle=True
)

# 3. Training
for inputs, labels, dists in fo_loader:
    # Training mit Soft Targets
    pass
```

### Workflow 3: Benutzerdefinierte Verarbeitung

```python
# Mit benutzerdefinierten Funktionen
def custom_input_getter(sample):
    return sample['image']

def custom_transform(logits):
    # Label smoothing
    probs = torch.softmax(logits, dim=-1)
    return 0.9 * probs + 0.1 / logits.shape[-1]

generator = FirstOrderDataGenerator(
    model=model,
    input_getter=custom_input_getter,
    output_transform=custom_transform
)
```

---

## Fehlerbehandlung

### Typische Exceptions

| Exception | Grund | Lösung |
|-----------|-------|--------|
| `TypeError: Model must return a torch.Tensor` | Modell gibt kein Tensor zurück | Verwenden Sie `output_transform` |
| `KeyError: No distribution for index X` | Fehlende Distribution für Index | Prüfen Sie Index-Alignment |
| `FileNotFoundError` | JSON-Datei nicht gefunden | Prüfen Sie Pfad |
| `json.JSONDecodeError` | Ungültiges JSON | Prüfen Sie Dateiformat |

### Warnungen

```python
warnings.warn(
    "[FirstOrderDataset] distributions count does not match dataset length."
)
```

Tritt auf wenn Anzahl Verteilungen ≠ Dataset-Länge. Meist harmlos, aber prüfen Sie Index-Alignment.

---

## Kompatibilität

- **PyTorch**: >= 1.8.0
- **Python**: >= 3.8
- **Geräte**: CPU, CUDA
- **Frameworks**: PyTorch (primär), kompatibel mit TensorFlow/JAX via Konvertierung

---

## Performance-Hinweise

1. **Batch-Größe**: Größere Batches = schneller, aber mehr GPU-Speicher
2. **num_workers**: > 0 für CPU-Dataset-Loading (außer Windows)
3. **pin_memory**: True für CUDA-Training (reduziert Transfer-Zeit)
4. **progress**: False für Batch-Jobs (reduziert I/O)

**Beispiel für maximale Performance:**
```python
generator = FirstOrderDataGenerator(
    model=model,
    device='cuda',
    batch_size=128  # Größer für schnellere Generierung
)

fo_loader = output_fo_dataloader(
    dataset,
    distributions,
    batch_size=64,
    num_workers=4,      # Paralleles Laden
    pin_memory=True     # Schnellerer GPU-Transfer
)
```

---

## Weitere Ressourcen

- **Hauptdokumentation**: `docs/data_generation_guide.md`
- **Beispiele**: `examples/simple_usage.py`
- **Tests**: `tests/test_first_order_generator.py`
- **Quellcode**: `probly/data_generator/first_order_generator.py`