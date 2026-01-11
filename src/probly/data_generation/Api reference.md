# First-Order Data Generator - API Referenz

## Inhaltsverzeichnis
- [Factory Pattern](#factory-pattern)
- [Framework-Spezifische Implementierungen](#framework-spezifische-implementierungen)
 - [PyTorch](#pytorch-firstorderdatagenerator)
 - [JAX](#jax-firstorderdatagenerator)
 - [Framework-Agnostic](#framework-agnostic-firstorderdatagenerator)
- [Base Generator](#base-generator)
- [Hilfsfunktionen](#hilfsfunktionen)
- [Datentypen & Workflows](#datentypen--workflows)

---

## Factory Pattern

### create_data_generator()

```python
def create_data_generator(
 framework: str,
 model: object,
 dataset: object,
 batch_size: int = 32,
 device: str | None = None,
) -> BaseDataGenerator
```

Erstellt einen framework-spezifischen Data Generator.

**Parameter:**
- `framework`: String - `"pytorch"`, `"tensorflow"` oder `"jax"`
- `model`: Modell-Objekt (framework-spezifisch)
- `dataset`: Dataset-Objekt (framework-spezifisch)
- `batch_size`: Batch-Größe für Verarbeitung
- `device`: Optional - Device-String für Inferenz

**Rückgabe:**
- `BaseDataGenerator`: Framework-spezifische Generator-Instanz

**Raises:**
- `ValueError`: Falls framework unbekannt ist

**Beispiel:**
```python
from probly.data_generator.factory import create_data_generator

# Automatische Framework-Selektion
generator = create_data_generator(
 framework='pytorch',
 model=my_torch_model,
 dataset=my_dataset,
 batch_size=32,
 device='cuda'
)
```

---

## Framework-Spezifische Implementierungen

## PyTorch FirstOrderDataGenerator

### Klassenbeschreibung

```python
@dataclass
class FirstOrderDataGenerator:
 """PyTorch-spezifischer First-Order Data Generator."""
```

Hauptklasse für PyTorch-basierte Generierung von First-Order Verteilungen.

### Konstruktor-Parameter

| Parameter | Typ | Standard | Beschreibung |
|-----------|-----|----------|--------------|
| `model` | `torch.nn.Module \| Callable` | **Erforderlich** | PyTorch Modell oder Callable |
| `device` | `str` | `"cpu"` | Device: `"cpu"` oder `"cuda"` |
| `batch_size` | `int` | `64` | Batch-Größe für Inferenz |
| `output_mode` | `str` | `"auto"` | `"auto"`, `"logits"` oder `"probs"` |
| `output_transform` | `Callable[[torch.Tensor], torch.Tensor] \| None` | `None` | Custom Transform |
| `input_getter` | `Callable[[Any], Any] \| None` | `None` | Custom Input-Extraktion |
| `model_name` | `str \| None` | `None` | Identifier für Metadaten |

### Methoden

#### `generate_distributions()`

```python
@torch.no_grad()
def generate_distributions(
 self,
 dataset_or_loader: Dataset | DataLoader,
 *,
 progress: bool = True,
) -> dict[int, list[float]]
```

Generiert Wahrscheinlichkeitsverteilungen für alle Samples.

**Parameter:**
- `dataset_or_loader`: PyTorch `Dataset` oder `DataLoader`
- `progress`: Bool - Fortschrittsanzeige aktivieren

**Rückgabe:**
- `dict[int, list[float]]`: Index → Wahrscheinlichkeitsliste

**Raises:**
- `TypeError`: Falls Modell keinen `torch.Tensor` zurückgibt
- `warnings.warn`: Falls Anzahl Verteilungen ≠ Dataset-Länge

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

Speichert Verteilungen als JSON.

**Parameter:**
- `path`: Zielpfad für JSON-Datei
- `distributions`: Zu speichernde Verteilungen
- `meta`: Optionale Metadaten

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

Lädt Verteilungen aus JSON.

**Rückgabe:**
- `tuple`: `(distributions, metadata)`

**Beispiel:**
```python
dists, meta = generator.load_distributions('output/dists.json')
print(f"Model: {meta['model_name']}, Samples: {len(dists)}")
```

#### `to_device()`

```python
def to_device(self, x: object) -> object
```

Verschiebt Tensor(en) auf konfiguriertes Device. Unterstützt verschachtelte Strukturen (lists, tuples, dicts).

#### `to_probs()`

```python
def to_probs(self, outputs: torch.Tensor) -> torch.Tensor
```

Konvertiert Modellausgaben zu Wahrscheinlichkeiten.
- Wendet `output_transform` an falls vorhanden
- Sonst: basierend auf `output_mode`
 - `"auto"`: Detektiert automatisch
 - `"logits"`: Wendet Softmax an
 - `"probs"`: Direkt verwenden

#### `prepares_batch_inp()` / `extract_input()`

```python
def prepares_batch_inp(self, sample: object) -> object
```

Extrahiert Modell-Input aus Dataset-Sample.
- Verwendet `input_getter` falls vorhanden
- Sonst: Entpackt Tuple/List `(input, label, ...)`

**Note:** `extract_input()` ist deprecated, verwenden Sie `prepares_batch_inp()`

#### `get_posterior_distributions()`

```python
def get_posterior_distributions(self) -> dict[str, dict[str, torch.Tensor]]
```

Extrahiert μ und ρ von BayesLinear Layern (für Bayesian Neural Networks).

**Rückgabe:**
- `dict`: Parameter mit `"mu"` und `"rho"` Tensoren

---

### FirstOrderDataset (PyTorch)

```python
class FirstOrderDataset(Dataset):
 """PyTorch Dataset-Wrapper für First-Order Verteilungen."""
```

**Parameter:**
- `base_dataset`: Original PyTorch Dataset
- `distributions`: Index-aligned Verteilungen
- `input_getter`: Optional - Custom Input-Extraktion

**Methoden:**
- `__len__()`: Gibt Anzahl Samples zurück
- `__getitem__(idx)`: Gibt `(input, label, distribution)` oder `(input, distribution)` zurück

**Beispiel:**
```python
fo_dataset = FirstOrderDataset(base_dataset, distributions)
input, label, dist = fo_dataset[0] # Mit Labels
# oder
input, dist = fo_dataset[0] # Ohne Labels
```

### output_dataloader() (PyTorch)

```python
def output_dataloader(
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

Erstellt PyTorch DataLoader mit First-Order Verteilungen.

**Beispiel:**
```python
fo_loader = output_dataloader(
 base_dataset=dataset,
 distributions=distributions,
 batch_size=32,
 shuffle=True,
 num_workers=4,
 pin_memory=True
)

for inputs, labels, dists in fo_loader:
 # Training...
 pass
```

---

## JAX FirstOrderDataGenerator

### Klassenbeschreibung

```python
@dataclass
class FirstOrderDataGenerator:
 """JAX-native First-Order Data Generator."""
```

JAX-spezifische Implementierung mit jnp.ndarray Support.

### Konstruktor-Parameter

| Parameter | Typ | Standard | Beschreibung |
|-----------|-----|----------|--------------|
| `model` | `Callable[..., Any]` | **Erforderlich** | JAX-transformierte Funktion |
| `device` | `str` | `"cpu"` | Device: `"cpu"`, `"gpu"`, `"tpu"` |
| `batch_size` | `int` | `64` | Batch-Größe |
| `output_mode` | `str` | `"auto"` | Output-Modus |
| `output_transform` | `Callable[[jnp.ndarray], jnp.ndarray] \| None` | `None` | Custom Transform |
| `input_getter` | `Callable \| None` | `None` | Input-Extraktion |
| `model_name` | `str \| None` | `None` | Identifier |

### JAX-spezifische Methoden

#### `to_device()`

```python
def to_device(self, x: object) -> object
```

Verschiebt Arrays auf JAX Device mittels `jax.device_put()`.

**Unterstützte Devices:**
- Plattform-Namen: `"cpu"`, `"gpu"`, `"tpu"`
- Spezifische IDs: `"gpu:0"`, `"tpu:1"`

#### `to_probs()`

```python
def to_probs(self, outputs: jnp.ndarray) -> jnp.ndarray
```

Konvertiert zu Wahrscheinlichkeiten mittels `jax.nn.softmax()`.

**JAX-spezifische Features:**
- Verwendet `jnp.ndarray` statt `torch.Tensor`
- Automatische Device-Platzierung
- Kompatibel mit jax.jit und jax.vmap

#### `_batchify_inputs()`

```python
def _batchify_inputs(self, batch: Sequence[object]) -> jnp.ndarray
```

Konvertiert Python-Liste zu jnp.ndarray für Batch-Verarbeitung.

### JaxDataLoader

```python
class JaxDataLoader:
 """Minimaler JAX-freundlicher DataLoader."""
```

**Parameter:**
- `dataset`: DatasetLike mit `__len__` und `__getitem__`
- `batch_size`: Batch-Größe
- `shuffle`: Bool - Zufällige Reihenfolge

**Methoden:**
- `__len__()`: Anzahl Batches
- `__iter__()`: Iterator über Batches (als Listen)

**Beispiel:**
```python
from probly.data_generator.jax_first_order_generator import JaxDataLoader

loader = JaxDataLoader(dataset, batch_size=32, shuffle=True)
for batch in loader:
 # batch ist Liste von Samples
 pass
```

### output_dataloader() (JAX)

```python
def output_dataloader(
 base_dataset: DatasetLike,
 distributions: Mapping[int, Iterable[float]],
 *,
 batch_size: int = 64,
 shuffle: bool = False,
 num_workers: int = 0, # Ignored - nur für API-Kompatibilität
 pin_memory: bool = False, # Ignored
 input_getter: Callable[[Any], Any] | None = None,
) -> JaxDataLoader
```

**Note:** `num_workers` und `pin_memory` werden ignoriert (warnt bei non-default Werten).

---

## Framework-Agnostic FirstOrderDataGenerator

### Klassenbeschreibung

```python
@dataclass
class FirstOrderDataGenerator:
 """Pure Python First-Order Generator (kein Framework nötig)."""
```

Framework-unabhängige Implementierung für maximale Kompatibilität.

### Konstruktor-Parameter

| Parameter | Typ | Standard | Beschreibung |
|-----------|-----|----------|--------------|
| `model` | `Callable[..., Any]` | **Erforderlich** | Callable Modell |
| `device` | `str` | `"cpu"` | Nur für API-Symmetrie (unused) |
| `batch_size` | `int` | `64` | Batch-Größe |
| `output_mode` | `str` | `"auto"` | Output-Modus |
| `output_transform` | `Callable \| None` | `None` | Custom Transform |
| `input_getter` | `Callable \| None` | `None` | Input-Extraktion |
| `model_name` | `str \| None` | `None` | Identifier |

### Pure Python Features

#### `to_probs()`

```python
def to_probs(self, outputs: object) -> list[list[float]]
```

Konvertiert beliebige Ausgaben zu Python Listen von Wahrscheinlichkeiten.

**Unterstützt:**
- Numpy arrays
- Python Listen/Tuples
- Skalare Werte
- Framework-spezifische Arrays (via Konvertierung)

**Output:** `list[list[float]]` - 2D Liste von Wahrscheinlichkeiten

#### `_to_batch_outputs()`

```python
def _to_batch_outputs(outputs: object) -> list[list[float]]
```

Normalisiert verschiedene Output-Shapes zu `list[list[float]]`.

#### `_softmax_row()`

```python
def _softmax_row(row: Sequence[float]) -> list[float]
```

Pure Python Softmax-Implementierung (keine Dependencies).

### SimpleDataLoader

```python
class SimpleDataLoader:
 """Minimaler Python DataLoader (keine Framework-Dependencies)."""
```

**Parameter:**
- `dataset`: DatasetLike mit `__len__` und `__getitem__`
- `batch_size`: Batch-Größe
- `shuffle`: Bool - Zufällige Reihenfolge

**Beispiel:**
```python
from probly.data_generator.first_order_datagenerator import SimpleDataLoader

loader = SimpleDataLoader(dataset, batch_size=32, shuffle=False)
for batch in loader:
 # batch ist Liste von Samples
 pass
```

---

## Base Generator

### BaseDataGenerator

```python
class BaseDataGenerator[M, D, Dev](ABC):
 """Abstract base class für alle Data Generators."""
```

**Generics:**
- `M`: Model-Typ
- `D`: Dataset-Typ
- `Dev`: Device-Typ

**Abstract Methods:**
```python
@abstractmethod
def generate(self) -> dict[str, Any]:
 """Run model und collect stats."""

@abstractmethod
def save(self, path: str) -> None:
 """Save results to file."""

@abstractmethod
def load(self, path: str) -> dict[str, Any]:
 """Load results from file."""
```

**Implementiert in:**
- `PyTorchDataGenerator`: Für PyTorch metrics
- `TensorFlowDataGenerator`: Für TensorFlow metrics
- `JAXDataGenerator`: Für JAX metrics

---

## Hilfsfunktionen

### _is_probabilities() (PyTorch)

```python
def _is_probabilities(x: torch.Tensor, atol: float = 1e-4) -> bool
```

Prüft ob Tensor Wahrscheinlichkeiten darstellt:
- Alle Werte in [0, 1]
- Zeilen summieren zu ~1.0 (mit Toleranz)

### _is_probabilities() (JAX)

```python
def _is_probabilities(x: jnp.ndarray, atol: float = 1e-4) -> bool
```

JAX-Version mit gleicher Logik.

### _is_prob_vector() (Framework-Agnostic)

```python
def _is_prob_vector(v: Sequence[float], atol: float = 1e-4) -> bool
```

Pure Python Version für einzelne Vektoren.

### _ensure_2d() (JAX)

```python
def _ensure_2d(x: jnp.ndarray) -> jnp.ndarray
```

Stellt sicher dass Array 2D ist (fügt Batch-Dimension hinzu falls nötig).

### _get_device() (JAX)

```python
def _get_device(device: str | None) -> jax.Device | None
```

Konvertiert Device-String zu JAX Device-Objekt.

---

## Datentypen

### Distribution Dict

```python
dict[int, list[float]]
```

Mapping von Dataset-Index zu Wahrscheinlichkeitsliste.

**Beispiel:**
```python
{
 0: [0.1, 0.3, 0.6], # Sample 0
 1: [0.2, 0.5, 0.3], # Sample 1
 ...
}
```

**Invarianten:**
- Keys: Non-negative Integers
- Values: Listen von Floats in [0, 1]
- Sum(values) ≈ 1.0 (innerhalb Toleranz)

### Metadata Dict

```python
dict[str, Any]
```

Beliebige Metadaten für gespeicherte Verteilungen.

**Standard-Felder:**
```python
{
 'model_name': str, # Name des Modells
 'dataset': str, # Name des Datensatzes
 'num_classes': int, # Anzahl Klassen
 'framework': str, # 'pytorch', 'jax', 'tensorflow'
 'accuracy': float, # Modell-Genauigkeit
 'timestamp': str, # ISO-Format Zeitstempel
}
```

### DatasetLike Protocol

```python
class DatasetLike(Protocol):
 def __len__(self) -> int: ...
 def __getitem__(self, idx: int) -> object: ...
```

Minimales Dataset-Interface für Framework-Kompatibilität.

---

## JSON-Dateiformat

### Struktur

```json
{
 "meta": {
 "model_name": "resnet50_v1",
 "dataset": "MNIST",
 "num_classes": 10,
 "framework": "pytorch",
 "accuracy": 0.95,
 "timestamp": "2025-01-11T10:30:00Z"
 },
 "distributions": {
 "0": [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.17, 0.13, 0.10, 0.07],
 "1": [0.03, 0.05, 0.08, 0.12, 0.18, 0.21, 0.15, 0.10, 0.06, 0.02],
 "...": "..."
 }
}
```

### Eigenschaften

- **Encoding**: UTF-8
- **Format**: JSON mit `ensure_ascii=False`
- **Keys in distributions**: Strings (von int konvertiert)
- **Values**: Listen von Floats
- **Cross-Framework**: Alle Frameworks können gleiche JSON-Dateien lesen

---

## Typische Workflows

### Workflow 1: PyTorch mit GPU

```python
from probly.data_generator.torch_first_order_generator import (
 FirstOrderDataGenerator,
 output_dataloader
)

# 1. Generator erstellen
generator = FirstOrderDataGenerator(
 model=pretrained_model,
 device='cuda',
 output_mode='logits',
 batch_size=128
)

# 2. Verteilungen generieren
distributions = generator.generate_distributions(dataset, progress=True)

# 3. Speichern
generator.save_distributions(
 'output/dists.json',
 distributions,
 meta={'dataset': 'CIFAR-10', 'accuracy': 0.92}
)

# 4. Laden und Training
dists, meta = generator.load_distributions('output/dists.json')
fo_loader = output_dataloader(dataset, dists, batch_size=64, shuffle=True)

for inputs, labels, target_dists in fo_loader:
 # Training...
 pass
```

### Workflow 2: JAX auf TPU

```python
from probly.data_generator.jax_first_order_generator import FirstOrderDataGenerator
import jax

# 1. Generator mit TPU
generator = FirstOrderDataGenerator(
 model=jax_model_fn,
 device='tpu',
 output_mode='logits',
 batch_size=256
)

# 2. Generieren
distributions = generator.generate_distributions(jax_dataset)

# 3. Speichern (kompatibel mit anderen Frameworks)
generator.save_distributions('output/jax_dists.json', distributions)
```

### Workflow 3: Framework-Wechsel

```python
# Generieren mit PyTorch
torch_gen = FirstOrderDataGenerator(model=torch_model, device='cuda')
dists = torch_gen.generate_distributions(torch_dataset)
torch_gen.save_distributions('dists.json', dists)

# Laden mit JAX
from probly.data_generator.jax_first_order_generator import FirstOrderDataGenerator
jax_gen = FirstOrderDataGenerator(model=jax_model)
loaded_dists, meta = jax_gen.load_distributions('dists.json')
# Verteilungen sind Framework-unabhängig!
```

### Workflow 4: Custom Processing

```python
# Custom Input Getter
def get_image_and_metadata(sample):
 return sample['image'], sample['metadata']

# Custom Output Transform (Label Smoothing)
def label_smoothing(logits):
 probs = torch.softmax(logits, dim=-1)
 alpha = 0.1
 return (1 - alpha) * probs + alpha / logits.shape[-1]

generator = FirstOrderDataGenerator(
 model=model,
 input_getter=get_image_and_metadata,
 output_transform=label_smoothing,
 device='cuda'
)
```

---

## Fehlerbehandlung

### Typische Exceptions

| Exception | Framework | Grund | Lösung |
|-----------|-----------|-------|--------|
| `TypeError: Model must return a torch.Tensor` | PyTorch | Falscher Return-Typ | Verwenden Sie `output_transform` |
| `TypeError: Model must return a jnp.ndarray` | JAX | Falscher Return-Typ | Stellen Sie sicher Modell gibt jnp.ndarray zurück |
| `KeyError: No distribution for index X` | Alle | Fehlende Distribution | Prüfen Sie Index-Alignment |
| `ValueError: Invalid output_mode '...'` | Alle | Ungültiger Mode | Verwenden Sie 'auto', 'logits' oder 'probs' |
| `FileNotFoundError` | Alle | JSON nicht gefunden | Prüfen Sie Pfad |
| `json.JSONDecodeError` | Alle | Ungültiges JSON | Prüfen Sie Dateiformat |

### Warnings

```python
warnings.warn(
 "[FirstOrderDataGenerator] generated X distributions, but dataset length is Y."
)
```

Tritt auf wenn Anzahl Verteilungen ≠ Dataset-Länge. Meist harmlos (z.B. bei `drop_last=True`), aber prüfen Sie Index-Alignment.

```python
warnings.warn(
 "[JAX output_dataloader] 'num_workers'=4 is ignored on JAX."
)
```

JAX-spezifisch: `num_workers` und `pin_memory` haben keine Wirkung.

---

## Kompatibilität

### Framework Compatibility Matrix

| Feature | PyTorch | TensorFlow | JAX | Agnostic |
|---------|---------|------------|-----|----------|
| Distribution Generation | | | | |
| JSON Save/Load | | | | |
| Custom Transforms | | | | |
| DataLoader Integration | | | | |
| GPU Support | | | | N/A |
| TPU Support | | | | N/A |
| Auto-detection | | | | |

### Python & Dependencies

- **Python**: >= 3.8 (Type hints erforderlich)
- **PyTorch**: >= 1.8.0 (für torch module)
- **TensorFlow**: >= 2.4.0 (für tensorflow module)
- **JAX**: >= 0.3.0 (für jax module)
- **Optional**: numpy (für agnostic module)

---

## Performance-Hinweise

### Batch-Größe

```python
# Klein (32-64): Weniger GPU-Speicher, langsamer
generator = FirstOrderDataGenerator(model=model, batch_size=32)

# Mittel (64-128): Gute Balance
generator = FirstOrderDataGenerator(model=model, batch_size=128)

# Groß (256+): Schneller, braucht viel GPU-Speicher
generator = FirstOrderDataGenerator(model=model, batch_size=256)
```

### DataLoader Optimierung

```python
# PyTorch
fo_loader = output_dataloader(
 dataset, distributions,
 batch_size=64,
 num_workers=4, # CPU-Parallelisierung (nicht auf Windows)
 pin_memory=True, # Schnellerer GPU-Transfer
 shuffle=True
)

# JAX (num_workers ignoriert)
fo_loader = output_dataloader(
 dataset, distributions,
 batch_size=64,
 shuffle=True # In-memory shuffle
)
```

### Memory Management

```python
# Bei großen Datasets: Batch-weise Generierung
distributions = {}
for batch_idx, batch in enumerate(dataloader):
 batch_dists = generator.generate_distributions(batch)
 distributions.update(batch_dists)

 if batch_idx % 10 == 0:
 # Periodisches Speichern
 generator.save_distributions(f'checkpoint_{batch_idx}.json', distributions)
```

---

## Weitere Ressourcen

- **Hauptdokumentation**: `docs/data_generation_guide.md`
- **Multi-Framework Guide**: `docs/multi_framework_guide.md`
- **Beispiele**: `examples/simple_usage.py`
- **Tests**: `tests/test_*_first_order.py`
- **Quellcode**: `probly/data_generator/`

---

**Last Updated**: Januar 2025 
**Maintainer**: ProblyPros
