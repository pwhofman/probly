# Multi-Framework Guide - First-Order Data Generator

## Überblick

Der First-Order Data Generator unterstützt drei Haupt-Frameworks: **PyTorch**, **TensorFlow/Keras** und **JAX**, plus eine framework-agnostische Pure Python Implementation. Dieser Guide erklärt die Unterschiede, Best Practices und Migrations-Strategien.

## Inhaltsverzeil chnis

- [Framework-Selektion](#framework-selektion)
- [PyTorch Implementation](#pytorch-implementation)
- [JAX Implementation](#jax-implementation)
- [TensorFlow Implementation](#tensorflow-implementation)
- [Framework-Agnostic Implementation](#framework-agnostic-implementation)
- [Cross-Framework Compatibility](#cross-framework-compatibility)
- [Migration zwischen Frameworks](#migration-zwischen-frameworks)

---

## Framework-Selektion

### Factory Pattern

Die einfachste Methode ist die Verwendung des Factory Patterns:

```python
from probly.data_generator.factory import create_data_generator

# Automatische Framework-Auswahl
generator = create_data_generator(
 framework='pytorch', # oder 'tensorflow', 'jax'
 model=model,
 dataset=dataset,
 batch_size=32,
 device='cuda'
)
```

### Direkte Imports

Alternativ können Sie framework-spezifische Module direkt importieren:

```python
# PyTorch
from probly.data_generator.torch_first_order_generator import FirstOrderDataGenerator

# JAX
from probly.data_generator.jax_first_order_generator import FirstOrderDataGenerator

# TensorFlow
from probly.data_generator.tensorflow_generator import TensorFlowDataGenerator

# Framework-Agnostic
from probly.data_generator.first_order_datagenerator import FirstOrderDataGenerator
```

---

## PyTorch Implementation

### Überblick

**Module**: `torch_first_order_generator.py`

**Features:**
- Vollständige `torch.nn.Module` Integration
- Native `torch.utils.data.DataLoader` Support
- GPU/CUDA Support
- Automatic Mixed Precision kompatibel

### Quick Start

```python
import torch
from torch.utils.data import Dataset, DataLoader
from probly.data_generator.torch_first_order_generator import (
 FirstOrderDataGenerator,
 FirstOrderDataset,
 output_dataloader
)

# 1. Model Setup
model = torch.load('pretrained_model.pt')
model.eval()

# 2. Generator erstellen
generator = FirstOrderDataGenerator(
 model=model,
 device='cuda' if torch.cuda.is_available() else 'cpu',
 batch_size=64,
 output_mode='logits',
 model_name='resnet50'
)

# 3. Verteilungen generieren
distributions = generator.generate_distributions(dataset, progress=True)

# 4. Speichern
generator.save_distributions(
 'pytorch_dists.json',
 distributions,
 meta={'framework': 'pytorch', 'device': 'cuda'}
)
```

### PyTorch-Spezifische Features

#### 1. Device Management

```python
# Automatische Device-Erkennung
generator = FirstOrderDataGenerator(
 model=model,
 device='cuda' # Automatisch cuda:0
)

# Spezifisches GPU Device
generator = FirstOrderDataGenerator(
 model=model,
 device='cuda:1' # GPU 1
)

# CPU
generator = FirstOrderDataGenerator(
 model=model,
 device='cpu'
)
```

#### 2. DataLoader Integration

```python
# Erstelle DataLoader mit Verteilungen
fo_loader = output_dataloader(
 base_dataset=dataset,
 distributions=distributions,
 batch_size=32,
 shuffle=True,
 num_workers=4, # Multi-process loading
 pin_memory=True, # Faster GPU transfer
)

# Training Loop
for inputs, labels, target_dists in fo_loader:
 inputs = inputs.to(device)
 target_dists = target_dists.to(device)

 logits = model(inputs)
 loss = torch.nn.functional.kl_div(
 torch.nn.functional.log_softmax(logits, dim=-1),
 target_dists,
 reduction='batchmean'
 )
 loss.backward()
 optimizer.step()
```

#### 3. Bayesian Neural Networks Support

```python
# Für BayesLinear Layers
posterior_dists = generator.get_posterior_distributions()

# posterior_dists enthält:
# {
# 'layer_name': {
# 'mu': torch.Tensor,
# 'rho': torch.Tensor
# },
# ...
# }

# Speichern für spätere Verwendung
torch.save(posterior_dists, 'posterior_params.pt')
```

#### 4. Tensor Operations

```python
# Automatisches Device-Handling für verschachtelte Strukturen
data = {
 'images': torch.randn(10, 3, 224, 224),
 'metadata': {
 'features': torch.randn(10, 128)
 }
}

# Alle Tensors werden automatisch zum richtigen Device verschoben
data_on_device = generator.to_device(data)
```

### PyTorch Best Practices

```python
# 1. Model in eval() Modus
model.eval()

# 2. Keine Gradienten berechnen
with torch.no_grad():
 distributions = generator.generate_distributions(dataset)

# 3. Batch-Größe an GPU-Speicher anpassen
generator = FirstOrderDataGenerator(
 model=model,
 device='cuda',
 batch_size=256 if torch.cuda.get_device_properties(0).total_memory > 16e9 else 64
)

# 4. Memory-efficient für große Datasets
torch.cuda.empty_cache()
distributions = generator.generate_distributions(dataset, progress=True)
torch.cuda.empty_cache()
```

---

## JAX Implementation

### Überblick

**Module**: `jax_first_order_generator.py`

**Features:**
- Native `jnp.ndarray` Support
- TPU Support
- JIT-compilation kompatibel
- Functional programming paradigm

### Quick Start

```python
import jax
import jax.numpy as jnp
from probly.data_generator.jax_first_order_generator import (
 FirstOrderDataGenerator,
 FirstOrderDataset,
 output_dataloader
)

# 1. JAX Model Function
def model_fn(x):
 # Ihr JAX-Modell
 return jnp.dot(x, params['W']) + params['b']

# 2. Generator erstellen
generator = FirstOrderDataGenerator(
 model=model_fn,
 device='gpu', # oder 'cpu', 'tpu'
 batch_size=64,
 output_mode='logits'
)

# 3. Generieren
distributions = generator.generate_distributions(jax_dataset, progress=True)

# 4. Speichern
generator.save_distributions('jax_dists.json', distributions)
```

### JAX-Spezifische Features

#### 1. Device Management

```python
# Platform-based
generator = FirstOrderDataGenerator(
 model=model_fn,
 device='gpu' # Findet erste GPU
)

# Specific device ID
generator = FirstOrderDataGenerator(
 model=model_fn,
 device='gpu:1' # GPU 1
)

# TPU
generator = FirstOrderDataGenerator(
 model=model_fn,
 device='tpu' # Findet erste TPU
)

# Liste verfügbare Devices
devices = jax.devices()
print(f"Available: {[str(d) for d in devices]}")
```

#### 2. JIT-Compiled Models

```python
import jax

# JIT-compiled Model
@jax.jit
def model_fn(x):
 return forward_pass(x, params)

# Funktioniert direkt mit Generator
generator = FirstOrderDataGenerator(model=model_fn, device='gpu')
distributions = generator.generate_distributions(dataset)
```

#### 3. Vmap für Batch Processing

```python
# Vmap-basiertes Model
def single_prediction(x):
 return model_fn(x)

# Automatisches Batching mit vmap
batched_model = jax.vmap(single_prediction)

generator = FirstOrderDataGenerator(model=batched_model, device='gpu')
```

#### 4. Device Placement

```python
# Automatisches Device Placement
data = {
 'images': jnp.array(images),
 'labels': jnp.array(labels)
}

# Verschiebt alle Arrays zum konfigurierten Device
data_on_device = generator.to_device(data)

# Funktioniert mit verschachtelten Strukturen
nested_data = {
 'inputs': {
 'images': jnp.array(images),
 'features': jnp.array(features)
 },
 'metadata': jnp.array(metadata)
}
data_on_device = generator.to_device(nested_data)
```

### JAX Best Practices

```python
# 1. Pre-allocate Arrays
images = jnp.zeros((batch_size, 224, 224, 3), dtype=jnp.float32)

# 2. Use functional style
def pure_model(params, x):
 return forward(params, x)

# 3. Leverage JIT
@jax.jit
def generate_batch_dists(model, x):
 logits = model(x)
 return jax.nn.softmax(logits, axis=-1)

# 4. Handle static vs traced arrays
generator = FirstOrderDataGenerator(
 model=lambda x: model_fn(x, static_params=True),
 device='gpu'
)
```

### JAX vs PyTorch Unterschiede

| Feature | PyTorch | JAX |
|---------|---------|-----|
| Arrays | `torch.Tensor` | `jnp.ndarray` |
| Device API | `.to(device)` | `jax.device_put` |
| Gradients | `.backward()` | `jax.grad` |
| JIT | `torch.jit.script` | `@jax.jit` |
| Parallel | `DataLoader(num_workers)` | `jax.vmap` |
| TPU | Limited | Native |

---

## TensorFlow Implementation

### Überblick

**Module**: `tensorflow_generator.py`

**Features:**
- `tf.keras.Model` Integration
- `tf.data.Dataset` Support
- TPU/GPU Support
- Eager execution & Graph mode

### Quick Start

```python
import tensorflow as tf
from probly.data_generator.tensorflow_generator import TensorFlowDataGenerator

# 1. Keras Model
model = tf.keras.models.load_model('pretrained_model.h5')

# 2. tf.data.Dataset
dataset = tf.data.Dataset.from_tensor_slices((x, y))
dataset = dataset.batch(32)

# 3. Generator
generator = TensorFlowDataGenerator(
 model=model,
 dataset=dataset,
 batch_size=32,
 device='GPU:0' # oder 'CPU:0'
)

# 4. Generate (metrics-based)
results = generator.generate() # Returns accuracy, confidence, etc.

# 5. Save
generator.save('tf_results.json')
```

### TensorFlow-Spezifische Features

#### 1. tf.data.Dataset Integration

```python
# Preprocessing pipeline
dataset = tf.data.Dataset.from_tensor_slices((images, labels))
dataset = dataset.map(preprocess_fn)
dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(tf.data.AUTOTUNE)

# Direct use with generator
generator = TensorFlowDataGenerator(
 model=model,
 dataset=dataset,
 batch_size=32
)
```

#### 2. Strategy API

```python
# Multi-GPU mit MirroredStrategy
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
 model = create_model()
 generator = TensorFlowDataGenerator(
 model=model,
 dataset=dataset,
 batch_size=32 * strategy.num_replicas_in_sync
 )
```

#### 3. TPU Support

```python
# TPU Configuration
resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)

strategy = tf.distribute.TPUStrategy(resolver)

with strategy.scope():
 model = create_model()
 generator = TensorFlowDataGenerator(model=model, dataset=dataset)
```

### TensorFlow Best Practices

```python
# 1. Use tf.function für Performance
@tf.function
def model_predict(model, x):
 return model(x, training=False)

# 2. Prefetch Data
dataset = dataset.prefetch(tf.data.AUTOTUNE)

# 3. Mixed Precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# 4. Disable training mode
model.trainable = False
```

---

## Framework-Agnostic Implementation

### Überblick

**Module**: `first_order_datagenerator.py`

**Features:**
- Keine Framework-Dependencies
- Pure Python Implementation
- Maximale Kompatibilität
- Fallback für unbekannte Frameworks

### Quick Start

```python
from probly.data_generator.first_order_datagenerator import (
 FirstOrderDataGenerator,
 SimpleDataLoader
)

# 1. Beliebiges Callable Model
def my_model(inputs):
 # Ihre Implementierung (numpy, custom, etc.)
 return predictions

# 2. Generator
generator = FirstOrderDataGenerator(
 model=my_model,
 batch_size=64,
 output_mode='logits'
)

# 3. Simple Dataset
class MyDataset:
 def __len__(self):
 return len(self.data)

 def __getitem__(self, idx):
 return self.data[idx], self.labels[idx]

# 4. Generieren
distributions = generator.generate_distributions(MyDataset())

# 5. SimpleDataLoader
loader = SimpleDataLoader(dataset, batch_size=32, shuffle=True)
for batch in loader:
 # Process batch
 pass
```

### Wann Framework-Agnostic verwenden?

**Geeignet für:**
- Custom ML Frameworks
- Scikit-learn Models
- NumPy-basierte Models
- Legacy Code
- Prototyping

**Nicht geeignet für:**
- Production PyTorch/TensorFlow/JAX Code
- Performance-kritische Anwendungen
- GPU-beschleunigtes Training

---

## Cross-Framework Compatibility

### JSON-Format ist Framework-unabhängig

```python
# Generieren mit PyTorch
from probly.data_generator.torch_first_order_generator import FirstOrderDataGenerator
torch_gen = FirstOrderDataGenerator(model=torch_model)
torch_gen.save_distributions('dists.json', distributions)

# Laden mit JAX
from probly.data_generator.jax_first_order_generator import FirstOrderDataGenerator
jax_gen = FirstOrderDataGenerator(model=jax_model)
dists, meta = jax_gen.load_distributions('dists.json')
# Funktioniert! Verteilungen sind Framework-unabhängig
```

### Metadata Convention

Fügen Sie Framework-Info in Metadaten hinzu:

```python
generator.save_distributions(
 'dists.json',
 distributions,
 meta={
 'framework': 'pytorch', # oder 'jax', 'tensorflow'
 'device': 'cuda',
 'model_name': 'resnet50',
 # ... weitere Metadaten
 }
)
```

### Conversion Utilities

```python
# PyTorch → NumPy → JAX
torch_tensor = torch.randn(10, 10)
numpy_array = torch_tensor.cpu().numpy()
jax_array = jnp.array(numpy_array)

# JAX → NumPy → PyTorch
jax_array = jnp.ones((10, 10))
numpy_array = np.array(jax_array)
torch_tensor = torch.from_numpy(numpy_array)
```

---

## Migration zwischen Frameworks

### PyTorch → JAX

```python
# Vorher (PyTorch)
import torch
from probly.data_generator.torch_first_order_generator import FirstOrderDataGenerator

torch_model = torch.load('model.pt')
torch_gen = FirstOrderDataGenerator(model=torch_model, device='cuda')
dists = torch_gen.generate_distributions(torch_dataset)

# Nachher (JAX)
import jax
from probly.data_generator.jax_first_order_generator import FirstOrderDataGenerator

def jax_model_fn(x):
 # Port PyTorch model zu JAX
 return jax_forward_pass(x, converted_params)

jax_gen = FirstOrderDataGenerator(model=jax_model_fn, device='gpu')
dists = jax_gen.generate_distributions(jax_dataset)
```

### Wichtige Unterschiede beachten:

| Aspekt | PyTorch | JAX |
|--------|---------|-----|
| Model Type | `nn.Module` | Function |
| Arrays | `torch.Tensor` | `jnp.ndarray` |
| Device API | `.to('cuda')` | `jax.device_put()` |
| In-place Ops | Allowed | Not allowed (functional) |
| Gradients | `loss.backward()` | `jax.grad(loss_fn)` |

### Dataset Conversion

```python
# PyTorch Dataset → JAX
class PyTorchDataset(torch.utils.data.Dataset):
 def __getitem__(self, idx):
 return self.data[idx], self.labels[idx]

# JAX-kompatible Version
class JAXDataset:
 def __getitem__(self, idx):
 # Convert to JAX arrays
 return jnp.array(self.data[idx]), jnp.array(self.labels[idx])
```

---

## Performance Comparison

### Benchmark Setup

```python
# Same task: Generate distributions for 10,000 samples, 10 classes
dataset_size = 10000
num_classes = 10
batch_size = 64
device = 'GPU'
```

### Results (Approximate)

| Framework | Time (s) | Memory (GB) | Notes |
|-----------|----------|-------------|-------|
| PyTorch | 12.3 | 2.1 | CUDA, fp32 |
| JAX | 10.8 | 1.9 | GPU, JIT |
| TensorFlow | 13.5 | 2.3 | GPU, eager |
| Agnostic | 45.2 | 0.8 | CPU only |

**Factors:**
- PyTorch: Mature, extensive ecosystem
- JAX: Fastest with JIT, functional style
- TensorFlow: Graph optimization, TPU support
- Agnostic: Slowest but universal

### Optimization Tips

```python
# PyTorch
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

# JAX
@jax.jit
def optimized_model(x):
 return model(x)

# TensorFlow
tf.config.optimizer.set_jit(True)
```

---

## Framework Selection Guide

### Wählen Sie PyTorch wenn:
 Sie bereits PyTorch verwenden
 Große Community & Ressourcen wichtig sind
 Flexibilität bei Model-Definition wichtig ist
 Debugging-Freundlichkeit Priorität hat

### Wählen Sie JAX wenn:
 Performance kritisch ist
 Sie funktionale Programmierung bevorzugen
 TPU-Support benötigt wird
 Automatische Differentiation zentral ist

### Wählen Sie TensorFlow wenn:
 Production Deployment wichtig ist
 TensorFlow Serving verwendet wird
 TPU-Training benötigt wird
 Mobile Deployment (TFLite) geplant ist

### Wählen Sie Framework-Agnostic wenn:
 Keine Dependencies erlaubt sind
 Custom Framework verwendet wird
 Maximale Portabilität benötigt wird
 Nur CPU-Inferenz benötigt wird

---

## Troubleshooting

### PyTorch Issues

**Problem**: CUDA out of memory
```python
# Lösung: Reduzieren Sie batch_size
generator = FirstOrderDataGenerator(model=model, batch_size=32) # statt 128

# Oder: Clear cache periodisch
torch.cuda.empty_cache()
```

**Problem**: Model not on same device as input
```python
# Lösung: Explizites Device-Management
model = model.to(device)
generator = FirstOrderDataGenerator(model=model, device=device)
```

### JAX Issues

**Problem**: "Array has been deleted"
```python
# Lösung: Kopieren Sie Arrays
x_copy = jnp.array(x) # Erstellt Kopie
```

**Problem**: JIT compilation Fehler
```python
# Lösung: Deaktivieren Sie JIT temporär
with jax.disable_jit():
 distributions = generator.generate_distributions(dataset)
```

### Cross-Framework Issues

**Problem**: Unterschiedliche Precision
```python
# PyTorch: float32 (default)
# JAX: float32 (default)
# Lösung: Explizit casten
jax_array = jnp.array(torch_tensor.numpy(), dtype=jnp.float32)
```

---

## Best Practices Summary

### Allgemein (alle Frameworks)

```python
# 1. Model in Eval-Modus
model.eval() # PyTorch
# model.trainable = False # TensorFlow

# 2. Konsistente Batch-Größe
batch_size = 64 # Standard, gut für die meisten GPUs

# 3. Progress Tracking
distributions = generator.generate_distributions(dataset, progress=True)

# 4. Metadaten dokumentieren
meta = {
 'framework': 'pytorch',
 'device': device,
 'timestamp': datetime.now().isoformat(),
 'model_name': 'resnet50',
 'accuracy': 0.95
}
```

### Framework-Spezifisch

```python
# PyTorch
with torch.no_grad():
 distributions = generator.generate_distributions(dataset)

# JAX
@jax.jit
def optimized_forward(x):
 return model(x)

# TensorFlow
@tf.function
def predict(x):
 return model(x, training=False)
```

---

**Last Updated**: Januar 2025
**Maintainer**: ProblyPros
