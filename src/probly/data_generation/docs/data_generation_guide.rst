First-Order Data Generator - User Guide
=============================================



Overview
---------


The **First-Order Data Generator** is a framework-agnostic tool for generating approximate probability distributions (First-Order Data) from pre-trained models. These distributions are essential for uncertainty quantification and coverage evaluation of Credal Sets.


Basic Concept
------------



Problem
~~~~~~~


In Machine Learning we normally work with the joint distribution p(X,Y), where X represents input features and Y represents target variables. The conditional distribution **p(Y|X)** - the probability of Y given X - is often not directly accessible.


Solution
~~~~~~


The First-Order Data Generator approximates this distribution through:

````
ĥ(x) ≈ p(·|x)
`````

where ĥ is a pre-trained model (e.g. from Huggingface, a ResNet, or another trained network).

**Workflow:**
1. **Input**: Pre-trained model + dataset
2. **Process**: For each sample x → generate ĥ(x)
3. **Output**: Approximated conditional distributions p(Y|X)
4. **Use**: Coverage evaluation, Knowledge Distillation, Uncertainty Quantification

Main Components
----------------



1. FirstOrderDataGenerator
~~~~~~~~~~~~~~~~~~~~~~~~~~


The main class for generating First-Order distributions. Available for:
- **PyTorch**: ``torch_first_order_generator.py``
- **JAX**: ``jax_first_order_generator.py``
- **Framework-Agnostic**: ``first_order_datagenerator.py``

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|-----|----------|--------------|
| ``model`` | Callable | **Required** | Model function or nn.Module |
| ``device`` | str | ``"cpu"`` | ``"cpu"``, ``"cuda"``, ``"gpu"``, ``"tpu"`` |
| ``batch_size`` | int | ``64`` | Batch size for inference |
| ``output_mode`` | str | ``"auto"`` | ``"auto"``, ``"logits"``, ``"probs"`` |
| ``output_transform`` | Callable | None | Custom transform function |
| ``input_getter`` | Callable | None | Custom input extraction |
| ``model_name`` | str | None | Identifier for metadata |

**output_mode Explanation:**
- **``"auto"``** (Recommended): Automatically detects if outputs are logits or probabilities
 - Checks: Values in [0,1] and sum ≈ 1.0
 - Applies Softmax if needed
- **``"logits"``**: Always applies Softmax (for raw logits)
- **``"probs"``**: Uses outputs directly (for already normalized probabilities)


2. FirstOrderDataset
~~~~~~~~~~~~~~~~~~~~


PyTorch Dataset wrapper that combines an existing dataset with First-Order distributions.

`````python
FirstOrderDataset(
 base_dataset: Dataset, # Original Dataset
 distributions: dict[int, list], # Index-aligned distributions
 input_getter: Callable = None # Optional: Custom input extraction
)
`````

**Returns:**
- With labels: ``(input, label, distribution)``
- Without labels: ``(input, distribution)``


3. output_dataloader / output_fo_dataloader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Helper function to create a DataLoader with First-Order distributions.

`````python
output_dataloader(
 base_dataset: Dataset,
 distributions: dict[int, list],
 batch_size: int = 64,
 shuffle: bool = False,
 num_workers: int = 0,
 pin_memory: bool = False,
 input_getter: Callable = None
) -> DataLoader
`````

---


Basic Usage
-----------------------



Step 1: Generate distributions (PyTorch)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


`````python
import torch
from torch.utils.data import Dataset
from probly.data_generator.torch_first_order_generator import FirstOrderDataGenerator

1. Load pre-trained model
==============================

model = torch.load('my_pretrained_model.pt')
model.eval() # IMPORTANT: Activate evaluation mode!

2. Your dataset
==============

dataset = MyDataset() # Your PyTorch Dataset implementation

3. Initialize generator
===========================

generator = FirstOrderDataGenerator(
 model=model,
 device='cuda' if torch.cuda.is_available() else 'cpu',
 batch_size=64,
 output_mode='logits', # If your model outputs logits
 model_name='my_model_v1'
)

4. Generate distributions
==========================

distributions = generator.generate_distributions(
 dataset,
 progress=True # Shows progress in console
)

distributions is a dict: {index: [prob_class_0, prob_class_1, ...]}
======================================================================

Example: {0: [0.1, 0.3, 0.6], 1: [0.2, 0.5, 0.3], ...}
=======================================================

`````


Step 2: Save distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


`````python
Save with metadata
=======================

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
`````

**JSON Format:**
`````json
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
`````


Step 3: Load and use distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


`````python
Load distributions
==================

loaded_dists, metadata = generator.load_distributions('output/first_order_dists.json')

print(f"Model: {metadata['model_name']}")
print(f"Dataset: {metadata['dataset']}")
print(f"Number of samples: {len(loaded_dists)}")

Use with FirstOrderDataset
===============================

from probly.data_generator.torch_first_order_generator import FirstOrderDataset

fo_dataset = FirstOrderDataset(
 base_dataset=dataset,
 distributions=loaded_dists
)

Retrieve element
===============

If base_dataset returns (input, label):
=============================================

input_tensor, label, distribution = fo_dataset[0]

If base_dataset only returns input:
========================================

input_tensor, distribution = fo_dataset[0]

print(f"Distribution: {distribution}") # torch.Tensor with probabilities
print(f"Sum: {distribution.sum()}") # Should be ≈ 1.0
`````


Step 4: Create DataLoader with First-Order distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


`````python
from probly.data_generator.torch_first_order_generator import output_dataloader

Create DataLoader
====================

fo_loader = output_dataloader(
 base_dataset=dataset,
 distributions=loaded_dists,
 batch_size=32,
 shuffle=True,
 num_workers=4, # Parallel loading (not on Windows)
 pin_memory=True # Faster GPU transfer
)

Training with Soft Targets (Knowledge Distillation)
==================================================

import torch.nn.functional as F

for batch in fo_loader:
 if len(batch) == 3: # With labels
 inputs, labels, target_distributions = batch
 else: # Without labels
 inputs, target_distributions = batch

 # Move to GPU
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
`````

---


Advanced Usage
---------------------



1. Custom Input Extraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


If your dataset has a more complex structure:

`````python
def custom_input_getter(sample):
 """
 Extracts only the image from a complex sample.

 Sample could be:
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
`````

**Further Examples:**

`````python
Example 1: Concatenate multiple inputs
========================================

def multi_input_getter(sample):
 image = sample['image']
 features = sample['features']
 return torch.cat([image.flatten(), features], dim=0)

Example 2: Apply preprocessing
==================================

def preprocess_getter(sample):
 image = sample['image']
 # Normalization
 mean = torch.tensor([0.485, 0.456, 0.406])
 std = torch.tensor([0.229, 0.224, 0.225])
 return (image - mean[:, None, None]) / std[:, None, None]

Example 3: Tuple unpacking
===========================

def tuple_getter(sample):
 # Sample is (image, label, metadata)
 return sample[0] # Only image
`````


2. Custom Output Transformation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


`````python
def custom_transform(outputs):
 """
 Custom transformation of model outputs.

 Example: Label Smoothing
 """
 # Apply Softmax
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
`````

**Further Examples:**

`````python
Example 1: Temperature Scaling
===============================

def temperature_scaling(outputs, temperature=2.0):
 return torch.softmax(outputs / temperature, dim=-1)

Example 2: Top-K Truncation
============================

def topk_transform(outputs, k=5):
 probs = torch.softmax(outputs, dim=-1)
 topk_probs, topk_indices = torch.topk(probs, k, dim=-1)
 # Renormalize
 topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
 # Create full probability vector
 result = torch.zeros_like(probs)
 result.scatter_(1, topk_indices, topk_probs)
 return result

Example 3: Ensemble Averaging
==============================

def ensemble_transform(outputs_list):
 # outputs_list is list of model outputs
 probs_list = [torch.softmax(out, dim=-1) for out in outputs_list]
 return torch.stack(probs_list).mean(dim=0)
`````


3. Using DataLoader instead of Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


`````python
from torch.utils.data import DataLoader

You can also pass a DataLoader directly
=================================================

custom_loader = DataLoader(
 dataset,
 batch_size=16,
 shuffle=False, # IMPORTANT: shuffle=False for correct indexing!
 num_workers=4
)

distributions = generator.generate_distributions(custom_loader, progress=True)
`````


4. Framework-Agnostic (Pure Python)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


`````python
from probly.data_generator.first_order_datagenerator import FirstOrderDataGenerator

Works with any Callable models (no PyTorch dependency)
======================================================================

def my_model(inputs):
 # Your model implementation
 # Can use NumPy, custom framework, etc.
 predictions = model_forward(inputs)
 return predictions

generator = FirstOrderDataGenerator(
 model=my_model,
 batch_size=64,
 output_mode='logits'
)

SimpleDataLoader (no PyTorch dependency)
===========================================

from probly.data_generator.first_order_datagenerator import SimpleDataLoader

loader = SimpleDataLoader(dataset, batch_size=32, shuffle=False)
distributions = generator.generate_distributions(loader)
`````


5. JAX Implementation
~~~~~~~~~~~~~~~~~~~~~


`````python
import jax
import jax.numpy as jnp
from probly.data_generator.jax_first_order_generator import FirstOrderDataGenerator

JAX Model Function
==================

def jax_model_fn(x):
 # Your JAX model
 return jnp.dot(x, params['W']) + params['b']

JAX Generator
=============

generator = FirstOrderDataGenerator(
 model=jax_model_fn,
 device='gpu', # or 'cpu', 'tpu'
 batch_size=64,
 output_mode='logits'
)

Generate with JAX
==================

distributions = generator.generate_distributions(jax_dataset, progress=True)

Save (compatible with PyTorch!)
===================================

generator.save_distributions('jax_dists.json', distributions)
`````

**JAX-specific Features:**

`````python
JIT-compiled Model
==================

@jax.jit
def optimized_model(x):
 return forward_pass(x, params)

generator = FirstOrderDataGenerator(model=optimized_model, device='gpu')

Device Placement
================

data_on_gpu = generator.to_device(data)

Supports nested structures
=====================================

nested_data = {
 'images': jnp.array(images),
 'features': {
 'categorical': jnp.array(cat_features),
 'numerical': jnp.array(num_features)
 }
}
nested_on_gpu = generator.to_device(nested_data)
`````

---


Use Cases
---------------



1. Credal Set Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~


`````python
Step 1: Generate teacher distributions (Ground Truth)
========================================================

teacher_gen = FirstOrderDataGenerator(model=pretrained_teacher_model, device='cuda')
ground_truth_dists = teacher_gen.generate_distributions(test_set)

Step 2: Train student model with Credal Sets
===================================================

student_model = train_credal_model(train_set)

Step 3: Evaluate coverage
=============================

student_credal_sets = student_model.predict_credal_sets(test_set)

def compute_coverage(credal_sets, ground_truth_dists):
 """
 Checks if ground truth is contained in credal set.

 Credal Set: Set of probability distributions
 Ground Truth: Single probability distribution

 Returns: Coverage rate (fraction of samples where GT is in credal set)
 """
 covered = 0
 total = len(ground_truth_dists)

 for idx, gt_dist in ground_truth_dists.items():
 credal_set = credal_sets[idx]

 # Check if GT is in credal set
 # (Implementation depends on credal set representation)
 if is_in_credal_set(gt_dist, credal_set):
 covered += 1

 return covered / total

coverage = compute_coverage(student_credal_sets, ground_truth_dists)
print(f"Coverage: {coverage:.2%}")
`````


2. Knowledge Distillation
~~~~~~~~~~~~~~~~~~~~~~~~~


`````python
Step 1: Generate teacher distributions
=========================================

teacher_gen = FirstOrderDataGenerator(
 model=large_teacher_model,
 device='cuda',
 output_mode='logits'
)
teacher_dists = teacher_gen.generate_distributions(train_set)

Step 2: Train student with soft targets
==============================================

student_loader = output_dataloader(
 train_set,
 teacher_dists,
 batch_size=64,
 shuffle=True,
 num_workers=4,
 pin_memory=True
)

Step 3: Training loop
========================

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
`````


3. Uncertainty Quantification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


`````python
Step 1: Collect distributions from multiple models
=====================================================

ensemble_models = [model1, model2, model3, model4, model5]
ensemble_dists = []

for model in ensemble_models:
 gen = FirstOrderDataGenerator(model=model, device='cuda')
 dists = gen.generate_distributions(dataset)
 ensemble_dists.append(dists)

Step 2: Calculate uncertainty metrics
=========================================

def compute_prediction_entropy(ensemble_dists):
 """
 Calculates entropy of averaged prediction (Epistemic Uncertainty).

 High entropy = Model is uncertain
 Low entropy = Model is certain
 """
 n_models = len(ensemble_dists)
 n_samples = len(ensemble_dists[0])

 uncertainties = {}

 for idx in range(n_samples):
 # Collect all distributions for this sample
 sample_dists = [dists[idx] for dists in ensemble_dists]

 # Average distribution
 avg_dist = torch.stack([torch.tensor(d) for d in sample_dists]).mean(dim=0)

 # Calculate entropy
 entropy = -(avg_dist * torch.log(avg_dist + 1e-10)).sum()
 uncertainties[idx] = entropy.item()

 return uncertainties

uncertainties = compute_prediction_entropy(ensemble_dists)

Step 3: Find samples with highest uncertainty
==================================================

sorted_samples = sorted(uncertainties.items(), key=lambda x: x[1], reverse=True)
most_uncertain = sorted_samples[:10] # Top 10 uncertain samples

print("Most uncertain samples:")
for idx, uncertainty in most_uncertain:
 print(f" Sample {idx}: Uncertainty = {uncertainty:.4f}")
`````


4. Active Learning
~~~~~~~~~~~~~~~~~~


`````python
Step 1: Initial training
===========================

model = train_initial_model(labeled_pool)

Step 2: Generate distributions for unlabeled pool
====================================================

generator = FirstOrderDataGenerator(model=model, device='cuda')
unlabeled_dists = generator.generate_distributions(unlabeled_pool)

Step 3: Select samples with highest uncertainty
=================================================

def select_uncertain_samples(distributions, n=100):
 """Selects n samples with highest entropy."""
 entropies = {}
 for idx, dist in distributions.items():
 dist_tensor = torch.tensor(dist)
 entropy = -(dist_tensor * torch.log(dist_tensor + 1e-10)).sum()
 entropies[idx] = entropy.item()

 # Sort by entropy (highest first)
 sorted_indices = sorted(entropies.items(), key=lambda x: x[1], reverse=True)
 return [idx for idx, _ in sorted_indices[:n]]

uncertain_indices = select_uncertain_samples(unlabeled_dists, n=100)

Step 4: Label selected samples
==========================================

newly_labeled = label_samples(unlabeled_pool, uncertain_indices)

Step 5: Retrain with extended data
========================================

labeled_pool.extend(newly_labeled)
model = train_model(labeled_pool)
`````

---


Best Practices
--------------



Recommended Practices
~~~~~~~~~~~~~~~~~~~~



1. Model in evaluation mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


`````python
PyTorch
=======

model.eval()

TensorFlow
==========

model.trainable = False

Or explicitly:
==============

generator = FirstOrderDataGenerator(model=model, device='cuda')
with torch.no_grad():
 distributions = generator.generate_distributions(dataset)
`````

**Why?** Deactivates Dropout, Batch Normalization is not updated, no gradients are calculated.


2. Consistent indexing
~~~~~~~~~~~~~~~~~~~~~~~~~~


`````python
IMPORTANT: shuffle=False when generating!
=======================================

loader = DataLoader(dataset, batch_size=32, shuffle=False)
distributions = generator.generate_distributions(loader)

Then shuffle=True during training is OK
====================================

training_loader = output_dataloader(
 dataset,
 distributions,
 batch_size=32,
 shuffle=True # Now OK!
)
`````

**Why?** Distributions are saved with dataset indices. With shuffle=True the indices wouldn't match anymore!


3. Document metadata
~~~~~~~~~~~~~~~~~~~~~~~~~~


`````python
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
`````

**Why?** Reproducibility! You can later trace how the distributions were generated.


4. Plan storage space
~~~~~~~~~~~~~~~~~~~~~~~


`````python
Estimation:
============

Size ≈ num_samples * num_classes * 8 bytes (float) + Overhead
==============================================================

#
Example: 50,000 samples, 10 classes
====================================

→ ~50,000 * 10 * 8 = 4 MB (+ JSON Overhead ≈ 5-6 MB)
====================================================

#
Example: 50,000 samples, 1000 classes
======================================

→ ~50,000 * 1000 * 8 = 400 MB (+ JSON Overhead ≈ 500 MB)
========================================================


import os

def estimate_json_size(num_samples, num_classes):
 # Conservative estimate
 bytes_per_value = 10 # JSON Float encoding + overhead
 return num_samples * num_classes * bytes_per_value

estimated_size = estimate_json_size(len(dataset), num_classes)
print(f"Estimated file size: {estimated_size / 1e6:.1f} MB")

Check available storage
==========================

import shutil
free_space = shutil.disk_usage('.').free
print(f"Available space: {free_space / 1e9:.1f} GB")
`````


5. Optimize batch size
~~~~~~~~~~~~~~~~~~~~~~~~~


`````python
Too small: Slow
=================

generator = FirstOrderDataGenerator(model=model, batch_size=8)

Optimal: Balance between speed and memory
==========================================

generator = FirstOrderDataGenerator(model=model, batch_size=64)

Too large: CUDA out of memory
===========================

try:
 generator = FirstOrderDataGenerator(model=model, batch_size=512)
 distributions = generator.generate_distributions(dataset)
except RuntimeError as e:
 if "out of memory" in str(e):
 print("CUDA OOM! Reduce batch_size")
 # Fallback to smaller batch_size
 generator = FirstOrderDataGenerator(model=model, batch_size=64)
 distributions = generator.generate_distributions(dataset)
`````

**Auto-Tuning:**

`````python
def find_optimal_batch_size(model, dataset, device):
 """Finds optimal batch_size through binary search."""
 min_batch = 1
 max_batch = 512
 optimal = min_batch

 while min_batch <= max_batch:
 mid_batch = (min_batch + max_batch) // 2

 try:
 # Test with mid_batch
 generator = FirstOrderDataGenerator(
 model=model,
 device=device,
 batch_size=mid_batch
 )
 # Test on small subset
 test_subset = torch.utils.data.Subset(dataset, range(mid_batch * 2))
 _ = generator.generate_distributions(test_subset, progress=False)

 # Success → Try larger batch_size
 optimal = mid_batch
 min_batch = mid_batch + 1

 # Clean up
 torch.cuda.empty_cache()

 except RuntimeError as e:
 if "out of memory" in str(e):
 # OOM → Try smaller batch_size
 max_batch = mid_batch - 1
 torch.cuda.empty_cache()
 else:
 raise e

 return optimal

optimal_batch_size = find_optimal_batch_size(model, dataset, 'cuda')
print(f"Optimal batch_size: {optimal_batch_size}")
`````

---


Troubleshooting
--------------



Problem 1: "Model must return a torch.Tensor"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


**Cause:** Your model returns something other than ``torch.Tensor`` (e.g. Dictionary, Tuple, List).

**Solution:**

`````python
Before: Model returns dictionary
====================================

class MyModel(nn.Module):
 def forward(self, x):
 logits = self.network(x)
 return {'logits': logits, 'features': features}

Solution 1: Use output_transform
====================================

def extract_logits(output):
 return output['logits']

generator = FirstOrderDataGenerator(
 model=model,
 output_transform=extract_logits,
 device='cuda'
)

Solution 2: Model wrapper
=======================

class ModelWrapper(nn.Module):
 def __init__(self, model):
 super().__init__()
 self.model = model

 def forward(self, x):
 output = self.model(x)
 return output['logits']

wrapped_model = ModelWrapper(model)
generator = FirstOrderDataGenerator(model=wrapped_model, device='cuda')
`````


Problem 2: Warning about different lengths
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


**Warning:**
`````
[FirstOrderDataGenerator] generated 960 distributions, but dataset length is 1000.
`````

**Cause:** DataLoader with ``drop_last=True`` or uneven batch size.

**Solution:**

`````python
Option 1: drop_last=False (recommended)
=====================================

loader = DataLoader(dataset, batch_size=64, drop_last=False)

Option 2: Ignore (usually harmless)
====================================

The first 960 samples have distributions, the last 40 don't
===============================================================

`````


Problem 3: Memory error with large datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


**Error:**
`````
RuntimeError: CUDA out of memory
`````

**Solutions:**

`````python
Solution 1: Reduce batch_size
==============================

generator = FirstOrderDataGenerator(model=model, batch_size=16)

Solution 2: Batch-wise generation
=================================

distributions = {}
for batch_idx, batch in enumerate(dataloader):
 batch_dists = generator.generate_distributions(batch, progress=False)
 distributions.update(batch_dists)

 # Periodic saving
 if (batch_idx + 1) % 100 == 0:
 generator.save_distributions(
 f'checkpoint_{batch_idx}.json',
 distributions
 )

 # Clear cache
 torch.cuda.empty_cache()

Solution 3: CPU fallback
======================

try:
 generator = FirstOrderDataGenerator(model=model, device='cuda')
 distributions = generator.generate_distributions(dataset)
except RuntimeError as e:
 if "out of memory" in str(e):
 print("CUDA OOM, fallback to CPU")
 model = model.to('cpu')
 generator = FirstOrderDataGenerator(model=model, device='cpu')
 distributions = generator.generate_distributions(dataset)
`````


Problem 4: Distributions don't sum to 1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


**Symptom:**
`````python
dist = distributions[0]
print(sum(dist)) # 0.23 or 145.67 instead of ~1.0
`````

**Cause:** Wrong ``output_mode``.

**Solution:**

`````python
If your model outputs logits:
================================

generator = FirstOrderDataGenerator(
 model=model,
 output_mode='logits' # Applies Softmax
)

If your model already outputs normalized probs:
=====================================================

generator = FirstOrderDataGenerator(
 model=model,
 output_mode='probs' # No transformation
)

Unsure? Use 'auto' (recommended)
==========================================

generator = FirstOrderDataGenerator(
 model=model,
 output_mode='auto' # Detects automatically
)

Verify
============

distributions = generator.generate_distributions(dataset)
first_dist = distributions[0]
print(f"Sum: {sum(first_dist):.6f}") # Should be ≈ 1.0
`````


Problem 5: Index mismatch
~~~~~~~~~~~~~~~~~~~~~~~~~


**Symptom:** Labels and distributions don't match.

**Cause:** Dataset was changed between generation and usage.

**Solution:**

`````python
IMPORTANT: Same dataset state!
==================================


Step 1: Generation
======================

dataset = MyDataset(split='train', shuffle=False) # shuffle=False!
generator = FirstOrderDataGenerator(model=model)
distributions = generator.generate_distributions(dataset)
generator.save_distributions('dists.json', distributions)

Step 2: Usage (later)
==============================

dataset = MyDataset(split='train', shuffle=False) # Exactly the same!
dists, _ = generator.load_distributions('dists.json')
fo_dataset = FirstOrderDataset(dataset, dists)

Verification
=============

for i in range(min(5, len(fo_dataset))):
 original_input, original_label = dataset[i]
 fo_input, fo_label, fo_dist = fo_dataset[i]

 # Should be identical!
 assert torch.equal(original_input, fo_input)
 assert original_label == fo_label
`````

---


API Quick Reference
----------------



FirstOrderDataGenerator
~~~~~~~~~~~~~~~~~~~~~~~


`````python
generator = FirstOrderDataGenerator(
 model: Callable, # Model
 device: str = 'cpu', # Device
 batch_size: int = 64, # Batch size
 output_mode: str = 'auto', # 'auto', 'logits', 'probs'
 output_transform: Callable = None,
 input_getter: Callable = None,
 model_name: str = None
)

Methods
========

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
`````


FirstOrderDataset
~~~~~~~~~~~~~~~~~


`````python
fo_dataset = FirstOrderDataset(
 base_dataset: Dataset,
 distributions: dict[int, list],
 input_getter: Callable = None
)

input, label, dist = fo_dataset[idx] # With labels
input, dist = fo_dataset[idx] # Without labels
`````


output_dataloader
~~~~~~~~~~~~~~~~~


`````python
fo_loader = output_dataloader(
 base_dataset: Dataset,
 distributions: dict[int, list],
 batch_size: int = 64,
 shuffle: bool = False,
 num_workers: int = 0,
 pin_memory: bool = False,
 input_getter: Callable = None
)
`````

---


Further Resources
------------------


- **API Reference**: ``api_reference.md`` - Complete API documentation
- **Multi-Framework Guide**: ``multi_framework_guide.md`` - JAX, TensorFlow support
- **Examples**: ``simple_usage.py`` - Executable example
- **Tutorial Notebook**: ``first_order_tutorial.ipynb`` - Interactive tutorial
- **Tests**: ``test_first_order_generator.py`` - Test examples

---


Support & Contact
-----------------


- **Issues**: Create an issue in the repository
- **Documentation**: Consult the detailed documentation
- **Tests**: Look at the tests for more examples

---


License
------


Part of the ``probly`` project. More information in the main README.
