=====================
First-Order Data Generator
=====================

Ein framework-agnostisches Python-Tool zur Generierung von First-Order Data für die Evaluation von Credal Sets und Unsicherheitsquantifizierung in Machine Learning.

Überblick
=========

Was ist First-Order Data?
--------------------------

In der Unsicherheitsquantifizierung mit Credal Sets benötigen wir Ground-Truth Conditional Distributions ``p(Y|X)``, um Coverage-Metriken zu berechnen. Diese Ground-Truth Distributions, auch First-Order Data genannt, sind normalerweise nicht direkt verfügbar.

Dieser Generator approximiert ``p(Y|X)`` durch:

1. Verwendung eines gut vortrainierten Modells (z.B. von Huggingface)
2. Transformation von Samples ``x`` zu Verteilungen ``ĥ(x) ≈ p(·|x)``
3. Speicherung und Verwaltung dieser Verteilungen für Training und Evaluation

Problem und Lösung
-------------------

**Problem:** Coverage von Credal Sets messen

- Benötigt: Ground-truth ``p(Y|X)``
- Verfügbar: Nur Datenpunkte ``(x, y)``

**Lösung:** First-Order Data Generator

- Input: Pretrained Model + Dataset
- Output: Approximierte Conditional Distributions
- Verwendung: Coverage Evaluation, Model Training

Features
========

Multi-Framework Support
-----------------------

- **PyTorch**: Vollständige Integration mit torch.nn.Module und DataLoader
- **TensorFlow/Keras**: Native tf.data.Dataset und tf.keras.Model Support
- **JAX**: JAX-native Implementierung mit jnp.ndarray
- **Framework-Agnostic**: Pure Python Fallback für andere Frameworks

Kernfunktionen
--------------

- **Distribution Generation**: Erstellt ``ĥ(x) ≈ p(Y|x)`` für alle Samples
- **Flexible Output Handling**: Auto-detection von Logits vs. Probabilities
- **Efficient Storage**: JSON-basierte Persistierung mit Metadaten
- **DataLoader Integration**: Nahtlose Integration in Training Workflows
- **Customizable Processing**: Benutzerdefinierte Input-Getter und Output-Transforms

Installation
============

.. code-block:: bash

   # Basis-Installation (framework-agnostic)
   pip install probly

   # Mit PyTorch Support
   pip install probly[torch]

   # Mit TensorFlow Support
   pip install probly[tensorflow]

   # Mit JAX Support
   pip install probly[jax]

Quick Start
===========

Einfachstes Beispiel (PyTorch)
-------------------------------

.. code-block:: python

   from probly.data_generator.torch_first_order_generator import FirstOrderDataGenerator
   import torch

   # Vortrainiertes Modell
   model = torch.load('my_pretrained_model.pt')
   model.eval()

   # Dataset
   dataset = MyDataset()

   # Generator erstellen und Verteilungen generieren
   generator = FirstOrderDataGenerator(
       model=model,
       device='cuda',
       output_mode='logits'
   )

   distributions = generator.generate_distributions(dataset, progress=True)

   # Speichern
   generator.save_distributions(
       'output/distributions.json',
       distributions,
       meta={'dataset': 'MNIST', 'accuracy': 0.95}
   )

Multi-Framework Beispiel
-------------------------

.. code-block:: python

   from probly.data_generator.factory import create_data_generator

   # Automatische Framework-Erkennung
   generator = create_data_generator(
       framework='pytorch',  # oder 'tensorflow', 'jax'
       model=model,
       dataset=dataset,
       batch_size=32,
       device='cuda'
   )

   distributions = generator.generate_distributions(dataset)

Dokumentation
=============

Hauptdokumentation
------------------

- **Benutzerhandbuch** - ``docs/data_generation_guide.rst`` - Ausführliche Anleitung mit Beispielen
- **API Referenz** - ``docs/api_reference.rst`` - Vollständige API-Dokumentation
- **Multi-Framework Guide** - ``docs/multi_framework_guide.rst`` - Framework-spezifische Details

Beispiele
---------

- **Simple Usage** - ``examples/simple_usage.py`` - Grundlegendes PyTorch Beispiel
- **Tutorial Notebook** - ``examples/first_order_tutorial.ipynb`` - Interaktives Tutorial
- **Advanced Examples** - ``examples/`` - Weitere Anwendungsfälle

Architektur
===========

Projektstruktur
---------------

.. code-block:: text

   probly/data_generator/
       base_generator.py         # Abstract base class
       factory.py                # Framework factory
       torch_first_order_generator.py    # PyTorch implementation
       first_order_datagenerator.py      # Framework-agnostic implementation
       jax_first_order_generator.py      # JAX implementation
       pytorch_generator.py      # PyTorch metrics generator
       tensorflow_generator.py   # TensorFlow metrics generator
       jax_generator.py          # JAX metrics generator

Design Patterns
---------------

.. code-block:: python

   # Abstract Base Class
   class BaseDataGenerator[M, D, Dev](ABC):
       """Framework-agnostic interface"""
       @abstractmethod
       def generate(self) -> dict[str, Any]: ...
       @abstractmethod
       def save(self, path: str) -> None: ...
       @abstractmethod
       def load(self, path: str) -> dict[str, Any]: ...

   # Framework-specific implementations
   class PyTorchDataGenerator(BaseDataGenerator[torch.nn.Module, Dataset, str]):
       """PyTorch-specific implementation"""

   class TensorFlowDataGenerator(BaseDataGenerator[tf.keras.Model, tf.data.Dataset, str]):
       """TensorFlow-specific implementation"""

   class JAXDataGenerator(BaseDataGenerator[object, tuple, str]):
       """JAX-specific implementation"""

Verwendungsbeispiele
====================

1. PyTorch Classification
--------------------------

.. code-block:: python

   from probly.data_generator.torch_first_order_generator import FirstOrderDataGenerator
   from torch.utils.data import DataLoader

   # Generator initialisieren
   generator = FirstOrderDataGenerator(
       model=pretrained_model,
       device='cuda',
       batch_size=64,
       output_mode='logits',
       model_name='resnet50'
   )

   # Verteilungen generieren
   distributions = generator.generate_distributions(dataset, progress=True)

   # Speichern mit Metadaten
   generator.save_distributions(
       'output/dists.json',
       distributions,
       meta={
           'dataset': 'CIFAR-10',
           'num_classes': 10,
           'accuracy': 0.92
       }
   )

2. DataLoader und Training
---------------------------

.. code-block:: python

   from probly.data_generator.torch_first_order_generator import (
       FirstOrderDataset,
       output_dataloader
   )

   # Laden
   dists, meta = generator.load_distributions('output/dists.json')

   # DataLoader mit First-Order Verteilungen erstellen
   fo_loader = output_dataloader(
       base_dataset=dataset,
       distributions=dists,
       batch_size=32,
       shuffle=True,
       num_workers=4,
       pin_memory=True
   )

   # Training mit Soft Targets (Knowledge Distillation)
   for inputs, labels, target_dists in fo_loader:
       logits = student_model(inputs)
       loss = F.kl_div(
           F.log_softmax(logits, dim=-1),
           target_dists,
           reduction='batchmean'
       )
       loss.backward()
       optimizer.step()

3. JAX Implementation
----------------------

.. code-block:: python

   from probly.data_generator.jax_first_order_generator import FirstOrderDataGenerator
   import jax.numpy as jnp

   # JAX-native Generator
   generator = FirstOrderDataGenerator(
       model=jax_model_fn,
       device='gpu',
       batch_size=64,
       output_mode='logits'
   )

   # Generiere Verteilungen
   distributions = generator.generate_distributions(jax_dataset, progress=True)

   # Speichern (kompatibel mit anderen Frameworks)
   generator.save_distributions('output/jax_dists.json', distributions)

4. Framework-Agnostic (Pure Python)
------------------------------------

.. code-block:: python

   from probly.data_generator.first_order_datagenerator import FirstOrderDataGenerator

   # Funktioniert mit beliebigen Callable Models
   def my_model(inputs):
       # Ihre Modell-Implementierung
       return predictions

   generator = FirstOrderDataGenerator(
       model=my_model,
       batch_size=64,
       output_mode='logits'
   )

   distributions = generator.generate_distributions(dataset)

5. Benutzerdefinierte Verarbeitung
-----------------------------------

.. code-block:: python

   # Custom Input Getter für komplexe Datenstrukturen
   def custom_input_getter(sample):
       return sample['image']

   # Custom Output Transform (z.B. Label Smoothing)
   def custom_transform(logits):
       probs = torch.softmax(logits, dim=-1)
       alpha = 0.1
       n_classes = logits.shape[-1]
       return (1 - alpha) * probs + alpha / n_classes

   generator = FirstOrderDataGenerator(
       model=model,
       input_getter=custom_input_getter,
       output_transform=custom_transform,
       device='cuda'
   )

JSON Dateiformat
================

Struktur
--------

.. code-block:: json

   {
       "meta": {
           "model_name": "resnet50_v1",
           "dataset": "MNIST",
           "num_classes": 10,
           "accuracy": 0.95,
           "framework": "pytorch",
           "timestamp": "2025-01-11"
       },
       "distributions": {
           "0": [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.17, 0.13, 0.10, 0.07],
           "1": [0.03, 0.05, 0.08, 0.12, 0.18, 0.21, 0.15, 0.10, 0.06, 0.02],
           "...": "..."
       }
   }

Eigenschaften
-------------

- **Encoding**: UTF-8
- **Format**: JSON mit ``ensure_ascii=False``
- **Keys in distributions**: Strings (von int konvertiert)
- **Values**: Listen von Floats
- **Cross-Framework**: Alle Frameworks können gleiche JSON-Dateien lesen

Konfigurationsoptionen
======================

FirstOrderDataGenerator Parameter
----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Typ
     - Standard
     - Beschreibung
   * - ``model``
     - Callable
     - **Erforderlich**
     - Modell-Funktion oder nn.Module
   * - ``device``
     - str
     - ``"cpu"``
     - Device: ``"cpu"``, ``"cuda"``, ``"gpu"``, ``"tpu"``
   * - ``batch_size``
     - int
     - ``64``
     - Batch-Größe für Inferenz
   * - ``output_mode``
     - str
     - ``"auto"``
     - ``"auto"``, ``"logits"`` oder ``"probs"``
   * - ``output_transform``
     - Callable|None
     - ``None``
     - Custom Transform-Funktion
   * - ``input_getter``
     - Callable|None
     - ``None``
     - Custom Input-Extraktion
   * - ``model_name``
     - str|None
     - ``None``
     - Identifier für Metadaten

output_mode Erklärung
---------------------

- **auto**: Erkennt automatisch Logits vs. Probabilities

  - Prüft ob Werte in [0,1] und summieren zu 1
  - Wendet Softmax an falls nötig

- **logits**: Wendet immer Softmax an
- **probs**: Verwendet Ausgaben direkt ohne Transformation

Anwendungsfälle
===============

1. Credal Set Evaluation
------------------------

.. code-block:: python

   # Generiere Teacher-Verteilungen
   teacher_gen = FirstOrderDataGenerator(model=teacher_model)
   teacher_dists = teacher_gen.generate_distributions(test_set)

   # Evaluiere Student Credal Sets
   coverage = evaluate_credal_coverage(
       student_credal_sets=student_model.predict(test_set),
       ground_truth_dists=teacher_dists
   )
   print(f"Coverage: {coverage:.2%}")

2. Knowledge Distillation
--------------------------

.. code-block:: python

   # Teacher-Verteilungen generieren
   teacher_dists = teacher_gen.generate_distributions(train_set)

   # Student mit Soft Targets trainieren
   student_loader = output_dataloader(train_set, teacher_dists, batch_size=64)
   train_with_soft_targets(student_model, student_loader, epochs=10)

3. Uncertainty Quantification
------------------------------

.. code-block:: python

   # Ensemble von Modellen
   ensemble_dists = []
   for model in ensemble_models:
       gen = FirstOrderDataGenerator(model=model)
       dists = gen.generate_distributions(dataset)
       ensemble_dists.append(dists)

   # Analysiere Vorhersage-Varianz
   uncertainty_scores = compute_prediction_entropy(ensemble_dists)

Best Practices
==============

Empfohlene Praktiken
--------------------

1. **Evaluationsmodus aktivieren**

   .. code-block:: python

      model.eval()  # PyTorch
      # oder model.trainable = False  # TensorFlow

2. **Konsistente Indizierung**

   .. code-block:: python

      loader = DataLoader(dataset, batch_size=32, shuffle=False)
      # shuffle=False wichtig für korrekte Index-Zuordnung

3. **Metadaten dokumentieren**

   .. code-block:: python

      meta = {
          'model_architecture': 'ResNet-50',
          'dataset': 'CIFAR-10',
          'training_accuracy': 0.95,
          'timestamp': datetime.now().isoformat(),
          'notes': 'Pre-trained on ImageNet'
      }

4. **Speicherplatz planen**

   - Ca. 1 MB pro 10,000 Samples mit 10 Klassen
   - Ca. 10 MB pro 10,000 Samples mit 100 Klassen

Performance-Optimierung
-----------------------

.. code-block:: python

   # Für maximale Performance
   generator = FirstOrderDataGenerator(
       model=model,
       device='cuda',
       batch_size=128  # Größer = schneller (wenn GPU-Speicher ausreicht)
   )

   fo_loader = output_dataloader(
       dataset,
       distributions,
       batch_size=64,
       num_workers=4,  # Paralleles Laden (nicht auf Windows)
       pin_memory=True,  # Schnellerer GPU-Transfer
       shuffle=True
   )

Häufige Probleme und Lösungen
------------------------------

**Problem**: ``"Model must return a torch.Tensor"``

.. code-block:: python

   # Lösung: Verwenden Sie output_transform
   def extract_logits(output):
       return output['logits'] if isinstance(output, dict) else output

   generator = FirstOrderDataGenerator(
       model=model,
       output_transform=extract_logits
   )

**Problem**: Verteilungen summieren nicht zu 1.0

.. code-block:: python

   # Lösung: Stellen Sie output_mode korrekt ein
   generator = FirstOrderDataGenerator(
       model=model,
       output_mode='logits'  # Falls Ihr Modell Logits ausgibt
   )

**Problem**: Warnung über unterschiedliche Längen

.. code-block:: python

   # Meist harmlos - passiert bei drop_last=True
   # Falls problematisch: Prüfen Sie DataLoader-Konfiguration
   loader = DataLoader(dataset, batch_size=32, drop_last=False)

Tests
=====

.. code-block:: bash

   # Alle Tests ausführen
   pytest tests/test_first_order_generator.py -v

   # PyTorch Tests
   pytest tests/test_torch_first_order.py -v

   # JAX Tests
   pytest tests/test_jax_first_order.py -v

   # TensorFlow Tests
   pytest tests/test_tf_first_order.py -v

   # Mit Coverage
   pytest tests/ --cov=probly.data_generator --cov-report=html

Weitere Ressourcen
==================

Dokumentation
-------------

- **Data Generation Guide** - ``docs/data_generation_guide.rst`` - Ausführliches Tutorial
- **API Reference** - ``docs/api_reference.rst`` - Vollständige API-Dokumentation
- **Multi-Framework Guide** - ``docs/multi_framework_guide.rst`` - Framework-spezifische Details
- **Architecture Overview** - ``docs/architecture.rst`` - System-Design

Beispiele und Tutorials
------------------------

- **Simple Usage** - ``examples/simple_usage.py`` - Grundlegendes Beispiel
- **Advanced Tutorial** - ``examples/advanced_usage.py`` - Erweiterte Funktionen
- **Jupyter Notebook** - ``examples/first_order_tutorial.ipynb`` - Interaktives Tutorial

Code
----

- **GitHub Repository** - https://github.com/your-org/probly - Quellcode
- **Issue Tracker** - https://github.com/your-org/probly/issues - Bugs und Features

Kompatibilität
==============

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 40

   * - Framework
     - Version
     - Status
     - Notes
   * - PyTorch
     - >= 1.8.0
     - Full Support
     - Empfohlen
   * - TensorFlow
     - >= 2.4.0
     - Full Support
     - tf.data.Dataset
   * - JAX
     - >= 0.3.0
     - Full Support
     - jnp.ndarray
   * - Python
     - >= 3.8
     - Required
     - Type hints

**Devices:**

- CPU (alle Frameworks)
- CUDA/GPU (PyTorch, JAX, TensorFlow)
- TPU (JAX, TensorFlow)

Lizenz
======

Teil des ``probly`` Projekts - siehe Haupt-Repository für Lizenzinformationen.
