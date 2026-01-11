# First-Order Data Generator - Benutzerhandbuch

## Überblick

Der **First-Order Data Generator** ist ein framework-agnostisches Werkzeug zur Generierung approximativer Wahrscheinlichkeitsverteilungen (First-Order Data) aus vortrainierten Modellen. Diese Verteilungen sind essentiell für Unsicherheitsquantifizierung und Coverage-Evaluierung von Credal Sets.

## Grundkonzept

### Problem

In Machine Learning arbeiten wir normalerweise mit der gemeinsamen Verteilung p(X,Y), wobei X die Eingabefeatures und Y die Zielvariablen darstellen. Die bedingte Verteilung **p(Y|X)** - die Wahrscheinlichkeit von Y gegeben X - ist oft nicht direkt zugänglich.

### Lösung

Der First-Order Data Generator approximiert diese Verteilung durch:

```
ĥ(x) ≈ p(·|x)
```

wobei ĥ ein vortrainiertes Modell ist (z.B. von Huggingface, ein ResNet, oder ein anderes trainiertes Netz).

**Workflow:**
1. **Input**: Vortrainiertes Modell + Dataset
2. **Prozess**: Für jedes Sample x → generiere ĥ(x)
3. **Output**: Approximierte bedingte Verteilungen p(Y|X)
4. **Verwendung**: Coverage-Evaluation, Knowledge Distillation, Uncertainty Quantification

## Installation

```bash
# Basis-Installation
pip install probly

# Mit PyTorch
pip install probly[torch]

# Mit JAX
pip install probly[jax]

# Mit TensorFlow
pip install probly[tensorflow]

# Alle Frameworks
pip install probly[all]
```

## Hauptkomponenten

### 1. FirstOrderDataGenerator

Die Hauptklasse zur Generierung von First-Order Verteilungen. Verfügbar für:
- **PyTorch**: `torch_first_order_generator.py`
- **JAX**: `jax_first_order_generator.py`
- **Framework-Agnostic**: `first_order_datagenerator.py`

**Konstruktor-Parameter:**

| Parameter | Typ | Standard | Beschreibung |
|-----------|-----|----------|--------------|
| `model` | Callable | **Erforderlich** | Modell-Funktion oder nn.Module |
| `device` | str | `"cpu"` | `"cpu"`, `"cuda"`, `"gpu"`, `"tpu"` |
| `batch_size` | int | `64` | Batch-Größe für Inferenz |
| `output_mode` | str | `"auto"` | `"auto"`, `"logits"`, `"probs"` |
| `output_transform` | Callable | None | Custom Transform-Funktion |
| `input_getter` | Callable | None | Custom Input-Extraktion |
| `model_name` | str | None | Identifier für Metadaten |

**output_mode Erklärung:**
- **`"auto"`** (Empfohlen): Erkennt automatisch ob Ausgaben Logits oder Probabilitäten sind
 - Prüft: Werte in [0,1] und Summe ≈ 1.0
 - Wendet Softmax an falls nötig
- **`"logits"`**: Wendet immer Softmax an (für rohe Logits)
- **`"probs"`**: Verwendet Ausgaben direkt (für bereits normalisierte Probabilitäten)

### 2. FirstOrderDataset

PyTorch Dataset-Wrapper der einen bestehenden Dataset mit First-Order Verteilungen kombiniert.

```python
FirstOrderDataset(
 base_dataset: Dataset, # Original Dataset
 distributions: dict[int, list], # Index-aligned Verteilungen
 input_getter: Callable = None # Optional: Custom Input-Extraktion
)
```

**Rückgabe:**
- Mit Labels: `(input, label, distribution)`
- Ohne Labels: `(input, distribution)`

### 3. output_dataloader / output_fo_dataloader

Hilfsfunktion zum Erstellen eines DataLoaders mit First-Order Verteilungen.

```python
output_dataloader(
 base_dataset: Dataset,
 distributions: dict[int, list],
 batch_size: int = 64,
 shuffle: bool = False,
 num_workers: int = 0,
 pin_memory: bool = False,
 input_getter: Callable = None
) -> DataLoader
```

---

## Grundlegende Verwendung

### Schritt 1: Verteilungen generieren (PyTorch)

```python
import torch
from torch.utils.data import Dataset
from probly.data_generator.torch_first_order_generator import FirstOrderDataGenerator

# 1. Vortrainiertes Modell laden
model = torch.load('my_pretrained_model.pt')
model.eval() # WICHTIG: Evaluationsmodus aktivieren!

# 2. Ihr Dataset
dataset = MyDataset() # Ihre PyTorch Dataset-Implementierung

# 3. Generator initialisieren
generator = FirstOrderDataGenerator(
 model=model,
 device='cuda' if torch.cuda.is_available() else 'cpu',
 batch_size=64,
 output_mode='logits', # Falls Ihr Modell Logits ausgibt
 model_name='my_model_v1'
)

# 4. Verteilungen generieren
distributions = generator.generate_distributions(
 dataset,
 progress=True # Zeigt Fortschritt in Console
)

# distributions ist ein dict: {index: [prob_class_0, prob_class_1, ...]}
# Beispiel: {0: [0.1, 0.3, 0.6], 1: [0.2, 0.5, 0.3], ...}
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
 'accuracy': 0.92,
 'timestamp': '2025-01-11',
 'note': 'Generated with ResNet-50 pre-trained on ImageNet'
 }
)
```

**JSON-Format:**
```json
{
 "meta": {
 "model_name": "my_model_v1",
 "dataset": "CIFAR-10",
 "num_classes": 10,
 "accuracy": 0.92,
 "timestamp": "2025-01-11"
 },
 "distributions": {
 "0": [0.1, 0.2, 0.05, 0.15, 0.3, 0.05, 0.05, 0.05, 0.025, 0.025],
 "1": [0.05, 0.8, 0.03, 0.02, 0.05, 0.01, 0.01, 0.01, 0.01, 0.01],
 "...": "..."
 }
}
```

### Schritt 3: Verteilungen laden und verwenden

```python
# Verteilungen laden
loaded_dists, metadata = generator.load_distributions('output/first_order_dists.json')

print(f"Model: {metadata['model_name']}")
print(f"Dataset: {metadata['dataset']}")
print(f"Number of samples: {len(loaded_dists)}")

# Mit FirstOrderDataset verwenden
from probly.data_generator.torch_first_order_generator import FirstOrderDataset

fo_dataset = FirstOrderDataset(
 base_dataset=dataset,
 distributions=loaded_dists
)

# Element abrufen
# Falls base_dataset (input, label) zurückgibt:
input_tensor, label, distribution = fo_dataset[0]

# Falls base_dataset nur input zurückgibt:
input_tensor, distribution = fo_dataset[0]

print(f"Distribution: {distribution}") # torch.Tensor mit Probabilitäten
print(f"Sum: {distribution.sum()}") # Sollte ≈ 1.0 sein
```

### Schritt 4: DataLoader mit First-Order Verteilungen erstellen

```python
from probly.data_generator.torch_first_order_generator import output_dataloader

# DataLoader erstellen
fo_loader = output_dataloader(
 base_dataset=dataset,
 distributions=loaded_dists,
 batch_size=32,
 shuffle=True,
 num_workers=4, # Paralleles Laden (nicht auf Windows)
 pin_memory=True # Schnellerer GPU-Transfer
)

# Training mit Soft Targets (Knowledge Distillation)
import torch.nn.functional as F

for batch in fo_loader:
 if len(batch) == 3: # Mit Labels
 inputs, labels, target_distributions = batch
 else: # Ohne Labels
 inputs, target_distributions = batch

 # Zu GPU verschieben
 inputs = inputs.to(device)
 target_distributions = target_distributions.to(device)

 # Forward pass
 logits = student_model(inputs)

 # KL Divergence Loss
 loss = F.kl_div(
 F.log_softmax(logits, dim=-1),
 target_distributions,
 reduction='batchmean'
 )

 # Backward pass
 loss.backward()
 optimizer.step()
```

---

## Erweiterte Verwendung

### 1. Benutzerdefinierte Input-Extraktion

Falls Ihr Dataset eine komplexere Struktur hat:

```python
def custom_input_getter(sample):
 """
 Extrahiert nur das Image aus einem komplexen Sample.

 Sample könnte sein:
 {
 'image': torch.Tensor,
 'metadata': dict,
 'label': int,
 'id': str
 }
 """
 return sample['image']

generator = FirstOrderDataGenerator(
 model=model,
 input_getter=custom_input_getter,
 device='cuda'
)

distributions = generator.generate_distributions(complex_dataset)
```

**Weitere Beispiele:**

```python
# Beispiel 1: Mehrere Inputs konkatenieren
def multi_input_getter(sample):
 image = sample['image']
 features = sample['features']
 return torch.cat([image.flatten(), features], dim=0)

# Beispiel 2: Preprocessing anwenden
def preprocess_getter(sample):
 image = sample['image']
 # Normalisierung
 mean = torch.tensor([0.485, 0.456, 0.406])
 std = torch.tensor([0.229, 0.224, 0.225])
 return (image - mean[:, None, None]) / std[:, None, None]

# Beispiel 3: Tuple unpacking
def tuple_getter(sample):
 # Sample ist (image, label, metadata)
 return sample[0] # Nur Image
```

### 2. Benutzerdefinierte Output-Transformation

```python
def custom_transform(outputs):
 """
 Benutzerdefinierte Transformation von Modellausgaben.

 Beispiel: Label Smoothing
 """
 # Softmax anwenden
 probs = torch.softmax(outputs, dim=-1)

 # Label Smoothing: (1-α) * probs + α * uniform
 alpha = 0.1
 n_classes = outputs.shape[-1]
 uniform = torch.ones_like(probs) / n_classes

 return (1 - alpha) * probs + alpha * uniform

generator = FirstOrderDataGenerator(
 model=model,
 output_transform=custom_transform,
 device='cuda'
)
```

**Weitere Beispiele:**

```python
# Beispiel 1: Temperature Scaling
def temperature_scaling(outputs, temperature=2.0):
 return torch.softmax(outputs / temperature, dim=-1)

# Beispiel 2: Top-K Truncation
def topk_transform(outputs, k=5):
 probs = torch.softmax(outputs, dim=-1)
 topk_probs, topk_indices = torch.topk(probs, k, dim=-1)
 # Renormalize
 topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
 # Create full probability vector
 result = torch.zeros_like(probs)
 result.scatter_(1, topk_indices, topk_probs)
 return result

# Beispiel 3: Ensemble Averaging
def ensemble_transform(outputs_list):
 # outputs_list ist Liste von Modell-Ausgaben
 probs_list = [torch.softmax(out, dim=-1) for out in outputs_list]
 return torch.stack(probs_list).mean(dim=0)
```

### 3. Verwendung mit DataLoader statt Dataset

```python
from torch.utils.data import DataLoader

# Sie können auch direkt einen DataLoader übergeben
custom_loader = DataLoader(
 dataset,
 batch_size=16,
 shuffle=False, # WICHTIG: shuffle=False für korrekte Indizierung!
 num_workers=4
)

distributions = generator.generate_distributions(custom_loader, progress=True)
```

### 4. Framework-Agnostic (Pure Python)

```python
from probly.data_generator.first_order_datagenerator import FirstOrderDataGenerator

# Funktioniert mit beliebigen Callable Models (keine PyTorch-Dependency)
def my_model(inputs):
 # Ihre Modell-Implementierung
 # Kann NumPy, custom Framework, etc. verwenden
 predictions = model_forward(inputs)
 return predictions

generator = FirstOrderDataGenerator(
 model=my_model,
 batch_size=64,
 output_mode='logits'
)

# SimpleDataLoader (keine PyTorch-Dependency)
from probly.data_generator.first_order_datagenerator import SimpleDataLoader

loader = SimpleDataLoader(dataset, batch_size=32, shuffle=False)
distributions = generator.generate_distributions(loader)
```

### 5. JAX Implementation

```python
import jax
import jax.numpy as jnp
from probly.data_generator.jax_first_order_generator import FirstOrderDataGenerator

# JAX Model Function
def jax_model_fn(x):
 # Ihr JAX-Modell
 return jnp.dot(x, params['W']) + params['b']

# JAX Generator
generator = FirstOrderDataGenerator(
 model=jax_model_fn,
 device='gpu', # oder 'cpu', 'tpu'
 batch_size=64,
 output_mode='logits'
)

# Generieren mit JAX
distributions = generator.generate_distributions(jax_dataset, progress=True)

# Speichern (kompatibel mit PyTorch!)
generator.save_distributions('jax_dists.json', distributions)
```

**JAX-spezifische Features:**

```python
# JIT-compiled Model
@jax.jit
def optimized_model(x):
 return forward_pass(x, params)

generator = FirstOrderDataGenerator(model=optimized_model, device='gpu')

# Device Placement
data_on_gpu = generator.to_device(data)

# Unterstützt verschachtelte Strukturen
nested_data = {
 'images': jnp.array(images),
 'features': {
 'categorical': jnp.array(cat_features),
 'numerical': jnp.array(num_features)
 }
}
nested_on_gpu = generator.to_device(nested_data)
```

---

## Anwendungsfälle

### 1. Credal Set Evaluation

```python
# Schritt 1: Generiere Teacher-Verteilungen (Ground Truth)
teacher_gen = FirstOrderDataGenerator(model=pretrained_teacher_model, device='cuda')
ground_truth_dists = teacher_gen.generate_distributions(test_set)

# Schritt 2: Trainiere Student-Modell mit Credal Sets
student_model = train_credal_model(train_set)

# Schritt 3: Evaluiere Coverage
student_credal_sets = student_model.predict_credal_sets(test_set)

def compute_coverage(credal_sets, ground_truth_dists):
 """
 Prüft ob Ground-Truth in Credal Set enthalten ist.

 Credal Set: Menge von Wahrscheinlichkeitsverteilungen
 Ground Truth: Einzelne Wahrscheinlichkeitsverteilung

 Returns: Coverage rate (Anteil der Samples wo GT im Credal Set liegt)
 """
 covered = 0
 total = len(ground_truth_dists)

 for idx, gt_dist in ground_truth_dists.items():
 credal_set = credal_sets[idx]

 # Prüfe ob GT in Credal Set liegt
 # (Implementation abhängig von Credal Set Repräsentation)
 if is_in_credal_set(gt_dist, credal_set):
 covered += 1

 return covered / total

coverage = compute_coverage(student_credal_sets, ground_truth_dists)
print(f"Coverage: {coverage:.2%}")
```

### 2. Knowledge Distillation

```python
# Schritt 1: Lehrer-Verteilungen generieren
teacher_gen = FirstOrderDataGenerator(
 model=large_teacher_model,
 device='cuda',
 output_mode='logits'
)
teacher_dists = teacher_gen.generate_distributions(train_set)

# Schritt 2: Schüler mit Soft Targets trainieren
student_loader = output_dataloader(
 train_set,
 teacher_dists,
 batch_size=64,
 shuffle=True,
 num_workers=4,
 pin_memory=True
)

# Schritt 3: Training Loop
student_model = SmallStudentModel()
optimizer = torch.optim.Adam(student_model.parameters())

for epoch in range(num_epochs):
 for inputs, labels, teacher_probs in student_loader:
 inputs = inputs.to(device)
 teacher_probs = teacher_probs.to(device)

 # Forward pass
 student_logits = student_model(inputs)

 # Distillation Loss (KL Divergence)
 loss = F.kl_div(
 F.log_softmax(student_logits / temperature, dim=-1),
 teacher_probs,
 reduction='batchmean'
 )

 # Backward pass
 loss.backward()
 optimizer.step()
```

### 3. Uncertainty Quantification

```python
# Schritt 1: Verteilungen von mehreren Modellen sammeln
ensemble_models = [model1, model2, model3, model4, model5]
ensemble_dists = []

for model in ensemble_models:
 gen = FirstOrderDataGenerator(model=model, device='cuda')
 dists = gen.generate_distributions(dataset)
 ensemble_dists.append(dists)

# Schritt 2: Uncertainty Metriken berechnen
def compute_prediction_entropy(ensemble_dists):
 """
 Berechnet Entropy der gemittelten Vorhersage (Epistemic Uncertainty).

 Hohe Entropy = Modell ist unsicher
 Niedrige Entropy = Modell ist sicher
 """
 n_models = len(ensemble_dists)
 n_samples = len(ensemble_dists[0])

 uncertainties = {}

 for idx in range(n_samples):
 # Sammle alle Verteilungen für dieses Sample
 sample_dists = [dists[idx] for dists in ensemble_dists]

 # Durchschnittliche Verteilung
 avg_dist = torch.stack([torch.tensor(d) for d in sample_dists]).mean(dim=0)

 # Entropy berechnen
 entropy = -(avg_dist * torch.log(avg_dist + 1e-10)).sum()
 uncertainties[idx] = entropy.item()

 return uncertainties

uncertainties = compute_prediction_entropy(ensemble_dists)

# Schritt 3: Samples mit höchster Uncertainty finden
sorted_samples = sorted(uncertainties.items(), key=lambda x: x[1], reverse=True)
most_uncertain = sorted_samples[:10] # Top 10 uncertain samples

print("Most uncertain samples:")
for idx, uncertainty in most_uncertain:
 print(f" Sample {idx}: Uncertainty = {uncertainty:.4f}")
```

### 4. Active Learning

```python
# Schritt 1: Initial Training
model = train_initial_model(labeled_pool)

# Schritt 2: Generiere Verteilungen für unlabeled pool
generator = FirstOrderDataGenerator(model=model, device='cuda')
unlabeled_dists = generator.generate_distributions(unlabeled_pool)

# Schritt 3: Wähle Samples mit höchster Uncertainty
def select_uncertain_samples(distributions, n=100):
 """Wählt n Samples mit höchster Entropy."""
 entropies = {}
 for idx, dist in distributions.items():
 dist_tensor = torch.tensor(dist)
 entropy = -(dist_tensor * torch.log(dist_tensor + 1e-10)).sum()
 entropies[idx] = entropy.item()

 # Sortiere nach Entropy (höchste zuerst)
 sorted_indices = sorted(entropies.items(), key=lambda x: x[1], reverse=True)
 return [idx for idx, _ in sorted_indices[:n]]

uncertain_indices = select_uncertain_samples(unlabeled_dists, n=100)

# Schritt 4: Labele die ausgewählten Samples
newly_labeled = label_samples(unlabeled_pool, uncertain_indices)

# Schritt 5: Retrain mit erweiterten Daten
labeled_pool.extend(newly_labeled)
model = train_model(labeled_pool)
```

---

## Best Practices

### Empfohlene Praktiken

#### 1. Modell im Evaluationsmodus

```python
# PyTorch
model.eval()

# TensorFlow
model.trainable = False

# Oder explizit:
generator = FirstOrderDataGenerator(model=model, device='cuda')
with torch.no_grad():
 distributions = generator.generate_distributions(dataset)
```

**Warum?** Deaktiviert Dropout, Batch Normalization wird nicht aktualisiert, keine Gradienten werden berechnet.

#### 2. Konsistente Indizierung

```python
# WICHTIG: shuffle=False beim Generieren!
loader = DataLoader(dataset, batch_size=32, shuffle=False)
distributions = generator.generate_distributions(loader)

# Dann kann shuffle=True beim Training
training_loader = output_dataloader(
 dataset,
 distributions,
 batch_size=32,
 shuffle=True # Jetzt OK!
)
```

**Warum?** Distributions werden mit Dataset-Indices gespeichert. Bei shuffle=True würden die Indices nicht mehr matchen!

#### 3. Metadaten dokumentieren

```python
from datetime import datetime

meta = {
 'model_architecture': 'ResNet-50',
 'model_name': 'resnet50_imagenet',
 'dataset': 'CIFAR-10',
 'dataset_split': 'train',
 'num_samples': len(dataset),
 'num_classes': 10,
 'training_accuracy': 0.95,
 'validation_accuracy': 0.92,
 'timestamp': datetime.now().isoformat(),
 'device': 'cuda',
 'batch_size': 64,
 'output_mode': 'logits',
 'notes': 'Pre-trained on ImageNet, fine-tuned on CIFAR-10',
 'framework': 'pytorch',
 'torch_version': torch.__version__
}

generator.save_distributions('output/dists.json', distributions, meta=meta)
```

**Warum?** Reproduzierbarkeit! Sie können später nachvollziehen wie die Verteilungen generiert wurden.

#### 4. Speicherplatz planen

```python
# Abschätzung:
# Größe ≈ num_samples * num_classes * 8 bytes (float) + Overhead
#
# Beispiel: 50,000 samples, 10 classes
# → ~50,000 * 10 * 8 = 4 MB (+ JSON Overhead ≈ 5-6 MB)
#
# Beispiel: 50,000 samples, 1000 classes
# → ~50,000 * 1000 * 8 = 400 MB (+ JSON Overhead ≈ 500 MB)

import os

def estimate_json_size(num_samples, num_classes):
 # Konservative Schätzung
 bytes_per_value = 10 # JSON Float encoding + overhead
 return num_samples * num_classes * bytes_per_value

estimated_size = estimate_json_size(len(dataset), num_classes)
print(f"Geschätzte Dateigröße: {estimated_size / 1e6:.1f} MB")

# Prüfe verfügbaren Speicher
import shutil
free_space = shutil.disk_usage('.').free
print(f"Verfügbarer Speicher: {free_space / 1e9:.1f} GB")
```

#### 5. Batch-Größe optimieren

```python
# Zu klein: Langsam
generator = FirstOrderDataGenerator(model=model, batch_size=8)

# Optimal: Balance zwischen Speed und Memory
generator = FirstOrderDataGenerator(model=model, batch_size=64)

# Zu groß: CUDA out of memory
try:
 generator = FirstOrderDataGenerator(model=model, batch_size=512)
 distributions = generator.generate_distributions(dataset)
except RuntimeError as e:
 if "out of memory" in str(e):
 print("CUDA OOM! Reduziere batch_size")
 # Fallback auf kleinere batch_size
 generator = FirstOrderDataGenerator(model=model, batch_size=64)
 distributions = generator.generate_distributions(dataset)
```

**Auto-Tuning:**

```python
def find_optimal_batch_size(model, dataset, device):
 """Findet optimale batch_size durch binäre Suche."""
 min_batch = 1
 max_batch = 512
 optimal = min_batch

 while min_batch <= max_batch:
 mid_batch = (min_batch + max_batch) // 2

 try:
 # Test mit mid_batch
 generator = FirstOrderDataGenerator(
 model=model,
 device=device,
 batch_size=mid_batch
 )
 # Test auf kleinem Subset
 test_subset = torch.utils.data.Subset(dataset, range(mid_batch * 2))
 _ = generator.generate_distributions(test_subset, progress=False)

 # Erfolgreich → Versuche größere batch_size
 optimal = mid_batch
 min_batch = mid_batch + 1

 # Clean up
 torch.cuda.empty_cache()

 except RuntimeError as e:
 if "out of memory" in str(e):
 # OOM → Versuche kleinere batch_size
 max_batch = mid_batch - 1
 torch.cuda.empty_cache()
 else:
 raise e

 return optimal

optimal_batch_size = find_optimal_batch_size(model, dataset, 'cuda')
print(f"Optimale batch_size: {optimal_batch_size}")
```

---

## Fehlerbehebung

### Problem 1: "Model must return a torch.Tensor"

**Ursache:** Ihr Modell gibt etwas anderes als `torch.Tensor` zurück (z.B. Dictionary, Tuple, Liste).

**Lösung:**

```python
# Vorher: Model gibt Dictionary zurück
class MyModel(nn.Module):
 def forward(self, x):
 logits = self.network(x)
 return {'logits': logits, 'features': features}

# Lösung 1: output_transform verwenden
def extract_logits(output):
 return output['logits']

generator = FirstOrderDataGenerator(
 model=model,
 output_transform=extract_logits,
 device='cuda'
)

# Lösung 2: Model-Wrapper
class ModelWrapper(nn.Module):
 def __init__(self, model):
 super().__init__()
 self.model = model

 def forward(self, x):
 output = self.model(x)
 return output['logits']

wrapped_model = ModelWrapper(model)
generator = FirstOrderDataGenerator(model=wrapped_model, device='cuda')
```

### Problem 2: Warnung über unterschiedliche Längen

**Warnung:**
```
[FirstOrderDataGenerator] generated 960 distributions, but dataset length is 1000.
```

**Ursache:** DataLoader mit `drop_last=True` oder ungleichmäßige Batch-Größe.

**Lösung:**

```python
# Option 1: drop_last=False (empfohlen)
loader = DataLoader(dataset, batch_size=64, drop_last=False)

# Option 2: Ignorieren (meist harmlos)
# Die ersten 960 Samples haben Verteilungen, die letzten 40 nicht
```

### Problem 3: Speicher-Fehler bei großen Datasets

**Fehler:**
```
RuntimeError: CUDA out of memory
```

**Lösungen:**

```python
# Lösung 1: Reduziere batch_size
generator = FirstOrderDataGenerator(model=model, batch_size=16)

# Lösung 2: Batch-weise Generierung
distributions = {}
for batch_idx, batch in enumerate(dataloader):
 batch_dists = generator.generate_distributions(batch, progress=False)
 distributions.update(batch_dists)

 # Periodisch speichern
 if (batch_idx + 1) % 100 == 0:
 generator.save_distributions(
 f'checkpoint_{batch_idx}.json',
 distributions
 )

 # Clear cache
 torch.cuda.empty_cache()

# Lösung 3: CPU Fallback
try:
 generator = FirstOrderDataGenerator(model=model, device='cuda')
 distributions = generator.generate_distributions(dataset)
except RuntimeError as e:
 if "out of memory" in str(e):
 print("CUDA OOM, fallback to CPU")
 model = model.to('cpu')
 generator = FirstOrderDataGenerator(model=model, device='cpu')
 distributions = generator.generate_distributions(dataset)
```

### Problem 4: Verteilungen summieren sich nicht zu 1

**Symptom:**
```python
dist = distributions[0]
print(sum(dist)) # 0.23 oder 145.67 statt ~1.0
```

**Ursache:** Falscher `output_mode`.

**Lösung:**

```python
# Falls Ihr Modell Logits ausgibt:
generator = FirstOrderDataGenerator(
 model=model,
 output_mode='logits' # Wendet Softmax an
)

# Falls Ihr Modell bereits normalisierte Probs ausgibt:
generator = FirstOrderDataGenerator(
 model=model,
 output_mode='probs' # Keine Transformation
)

# Unsicher? Verwenden Sie 'auto' (empfohlen)
generator = FirstOrderDataGenerator(
 model=model,
 output_mode='auto' # Erkennt automatisch
)

# Verifizieren
distributions = generator.generate_distributions(dataset)
first_dist = distributions[0]
print(f"Sum: {sum(first_dist):.6f}") # Sollte ≈ 1.0 sein
```

### Problem 5: Index-Mismatch

**Symptom:** Labels und Distributions matchen nicht.

**Ursache:** Dataset wurde zwischen Generierung und Verwendung verändert.

**Lösung:**

```python
# WICHTIG: Gleicher Dataset-Zustand!

# Schritt 1: Generierung
dataset = MyDataset(split='train', shuffle=False) # shuffle=False!
generator = FirstOrderDataGenerator(model=model)
distributions = generator.generate_distributions(dataset)
generator.save_distributions('dists.json', distributions)

# Schritt 2: Verwendung (später)
dataset = MyDataset(split='train', shuffle=False) # Exakt gleich!
dists, _ = generator.load_distributions('dists.json')
fo_dataset = FirstOrderDataset(dataset, dists)

# Verifizierung
for i in range(min(5, len(fo_dataset))):
 original_input, original_label = dataset[i]
 fo_input, fo_label, fo_dist = fo_dataset[i]

 # Sollten identisch sein!
 assert torch.equal(original_input, fo_input)
 assert original_label == fo_label
```

---

## API-Kurzreferenz

### FirstOrderDataGenerator

```python
generator = FirstOrderDataGenerator(
 model: Callable, # Modell
 device: str = 'cpu', # Device
 batch_size: int = 64, # Batch-Größe
 output_mode: str = 'auto', # 'auto', 'logits', 'probs'
 output_transform: Callable = None,
 input_getter: Callable = None,
 model_name: str = None
)

# Methoden
distributions = generator.generate_distributions(
 dataset_or_loader,
 progress=True
)

generator.save_distributions(
 path,
 distributions,
 meta=None
)

distributions, metadata = generator.load_distributions(path)
```

### FirstOrderDataset

```python
fo_dataset = FirstOrderDataset(
 base_dataset: Dataset,
 distributions: dict[int, list],
 input_getter: Callable = None
)

input, label, dist = fo_dataset[idx] # Mit Labels
input, dist = fo_dataset[idx] # Ohne Labels
```

### output_dataloader

```python
fo_loader = output_dataloader(
 base_dataset: Dataset,
 distributions: dict[int, list],
 batch_size: int = 64,
 shuffle: bool = False,
 num_workers: int = 0,
 pin_memory: bool = False,
 input_getter: Callable = None
)
```

---

## Weitere Ressourcen

- **API Reference**: `api_reference.md` - Vollständige API-Dokumentation
- **Multi-Framework Guide**: `multi_framework_guide.md` - JAX, TensorFlow Support
- **Examples**: `simple_usage.py` - Ausführbares Beispiel
- **Tutorial Notebook**: `first_order_tutorial.ipynb` - Interaktives Tutorial
- **Tests**: `test_first_order_generator.py` - Test-Beispiele

---

## Support & Kontakt

- **Issues**: Erstellen Sie ein Issue im Repository
- **Dokumentation**: Konsultieren Sie die detaillierte Dokumentation
- **Tests**: Schauen Sie sich die Tests für weitere Beispiele an

---

## Lizenz

Teil des `probly` Projekts. Weitere Informationen in der Haupt-README.

---

**Last Updated**: Januar 2025
**Maintainer**: ProblyPros
