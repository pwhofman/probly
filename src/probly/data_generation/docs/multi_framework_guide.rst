Multi-Framework Guide - First-Order Data Generator
==================================================



Overview
---------


The First-Order Data Generator supports three main frameworks: **PyTorch**, **TensorFlow/Keras** and **JAX**, plus a framework-agnostic Pure Python implementation. This guide explains the differences, best practices and migration strategies.


Table of Contents
--------------------


- `Framework Selection <#framework-selection>`_
- `PyTorch Implementation <#pytorch-implementation>`_
- `JAX Implementation <#jax-implementation>`_
- `TensorFlow Implementation <#tensorflow-implementation>`_
- `Framework-Agnostic Implementation <#framework-agnostic-implementation>`_
- `Cross-Framework Compatibility <#cross-framework-compatibility>`_
- `Migration between Frameworks <#migration-between-frameworks>`_

---


Framework Selection
-------------------



Factory Pattern
~~~~~~~~~~~~~~~


The simplest method is using the Factory Pattern:

````python
from probly.data_generator.factory import create_data_generator

Automatic framework selection
==============================

generator = create_data_generator(
 framework='pytorch', # or 'tensorflow', 'jax'
 model=model,
 dataset=dataset,
 batch_size=32,
 device='cuda'
)
`````


Direct Imports
~~~~~~~~~~~~~~~


Alternatively, you can import framework-specific modules directly:

`````python
PyTorch
=======

from probly.data_generator.torch_first_order_generator import FirstOrderDataGenerator

JAX
===

from probly.data_generator.jax_first_order_generator import FirstOrderDataGenerator

TensorFlow
==========

from probly.data_generator.tensorflow_generator import TensorFlowDataGenerator

Framework-Agnostic
==================

from probly.data_generator.first_order_datagenerator import FirstOrderDataGenerator
`````

---


PyTorch Implementation
----------------------



Overview
~~~~~~~~~


**Module**: ``torch_first_order_generator.py``

**Features:**
- Full ``torch.nn.Module`` integration
- Native ``torch.utils.data.DataLoader`` support
- GPU/CUDA support
- Automatic Mixed Precision compatible


Quick Start
~~~~~~~~~~~


`````python
import torch
from torch.utils.data import Dataset, DataLoader
from probly.data_generator.torch_first_order_generator import (
 FirstOrderDataGenerator,
 FirstOrderDataset,
 output_dataloader
)

1. Model setup
==============

model = torch.load('pretrained_model.pt')
model.eval()

2. Create generator
======================

generator = FirstOrderDataGenerator(
 model=model,
 device='cuda' if torch.cuda.is_available() else 'cpu',
 batch_size=64,
 output_mode='logits',
 model_name='resnet50'
)

3. Generate distributions
==========================

distributions = generator.generate_distributions(dataset, progress=True)

4. Save
============

generator.save_distributions(
 'pytorch_dists.json',
 distributions,
 meta={'framework': 'pytorch', 'device': 'cuda'}
)
`````


PyTorch-Specific Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~



1. Device Management
~~~~~~~~~~~~~~~~~~~~


`````python
Automatic device detection
=============================

generator = FirstOrderDataGenerator(
 model=model,
 device='cuda' # Automatically cuda:0
)

Specific GPU device
=======================

generator = FirstOrderDataGenerator(
 model=model,
 device='cuda:1' # GPU 1
)

CPU
===

generator = FirstOrderDataGenerator(
 model=model,
 device='cpu'
)
`````


2. DataLoader Integration
~~~~~~~~~~~~~~~~~~~~~~~~~


`````python
Create DataLoader with distributions
====================================

fo_loader = output_dataloader(
 base_dataset=dataset,
 distributions=distributions,
 batch_size=32,
 shuffle=True,
 num_workers=4, # Multi-process loading
 pin_memory=True, # Faster GPU transfer
)

Training loop
=============

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
`````


3. Bayesian Neural Networks Support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


`````python
For BayesLinear layers
======================

posterior_dists = generator.get_posterior_distributions()

posterior_dists contains:
========================

{
=

'layer_name': {
===============

'mu': torch.Tensor,
===================

'rho': torch.Tensor
===================

},
==

...
===

}
=


Save for later use
================================

torch.save(posterior_dists, 'posterior_params.pt')
`````


4. Tensor Operations
~~~~~~~~~~~~~~~~~~~~


`````python
Automatic device handling for nested structures
===========================================================

data = {
 'images': torch.randn(10, 3, 224, 224),
 'metadata': {
 'features': torch.randn(10, 128)
 }
}

All tensors are automatically moved to the correct device
===============================================================

data_on_device = generator.to_device(data)
`````


PyTorch Best Practices
~~~~~~~~~~~~~~~~~~~~~~


`````python
1. Model in eval() mode
========================

model.eval()

2. Don't compute gradients
=============================

with torch.no_grad():
 distributions = generator.generate_distributions(dataset)

3. Adapt batch size to GPU memory
=======================================

generator = FirstOrderDataGenerator(
 model=model,
 device='cuda',
 batch_size=256 if torch.cuda.get_device_properties(0).total_memory > 16e9 else 64
)

4. Memory-efficient for large datasets
======================================

torch.cuda.empty_cache()
distributions = generator.generate_distributions(dataset, progress=True)
torch.cuda.empty_cache()
`````

---


JAX Implementation
------------------



Overview
~~~~~~~~~


**Module**: ``jax_first_order_generator.py``

**Features:**
- Native ``jnp.ndarray`` support
- TPU support
- JIT-compilation compatible
- Functional programming paradigm


Quick Start
~~~~~~~~~~~


`````python
import jax
import jax.numpy as jnp
from probly.data_generator.jax_first_order_generator import (
 FirstOrderDataGenerator,
 FirstOrderDataset,
 output_dataloader
)

1. JAX Model Function
=====================

def model_fn(x):
 # Your JAX model
 return jnp.dot(x, params['W']) + params['b']

2. Create generator
======================

generator = FirstOrderDataGenerator(
 model=model_fn,
 device='gpu', # or 'cpu', 'tpu'
 batch_size=64,
 output_mode='logits'
)

3. Generate
=============

distributions = generator.generate_distributions(jax_dataset, progress=True)

4. Save
============

generator.save_distributions('jax_dists.json', distributions)
`````


JAX-Specific Features
~~~~~~~~~~~~~~~~~~~~~~~~



1. Device Management
~~~~~~~~~~~~~~~~~~~~


`````python
Platform-based
==============

generator = FirstOrderDataGenerator(
 model=model_fn,
 device='gpu' # Finds first GPU
)

Specific device ID
==================

generator = FirstOrderDataGenerator(
 model=model_fn,
 device='gpu:1' # GPU 1
)

TPU
===

generator = FirstOrderDataGenerator(
 model=model_fn,
 device='tpu' # Finds first TPU
)

List available devices
========================

devices = jax.devices()
print(f"Available: {[str(d) for d in devices]}")
`````


2. JIT-Compiled Models
~~~~~~~~~~~~~~~~~~~~~~


`````python
import jax

JIT-compiled model
==================

@jax.jit
def model_fn(x):
 return forward_pass(x, params)

Works directly with generator
=================================

generator = FirstOrderDataGenerator(model=model_fn, device='gpu')
distributions = generator.generate_distributions(dataset)
`````


3. Vmap for Batch Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~


`````python
Vmap-based model
====================

def single_prediction(x):
 return model_fn(x)

Automatic batching with vmap
===============================

batched_model = jax.vmap(single_prediction)

generator = FirstOrderDataGenerator(model=batched_model, device='gpu')
`````


4. Device Placement
~~~~~~~~~~~~~~~~~~~


`````python
Automatic device placement
==============================

data = {
 'images': jnp.array(images),
 'labels': jnp.array(labels)
}

Moves all arrays to configured device
================================================

data_on_device = generator.to_device(data)

Works with nested structures
===========================================

nested_data = {
 'inputs': {
 'images': jnp.array(images),
 'features': jnp.array(features)
 },
 'metadata': jnp.array(metadata)
}
data_on_device = generator.to_device(nested_data)
`````


JAX Best Practices
~~~~~~~~~~~~~~~~~~


`````python
1. Pre-allocate arrays
======================

images = jnp.zeros((batch_size, 224, 224, 3), dtype=jnp.float32)

2. Use functional style
=======================

def pure_model(params, x):
 return forward(params, x)

3. Leverage JIT
===============

@jax.jit
def generate_batch_dists(model, x):
 logits = model(x)
 return jax.nn.softmax(logits, axis=-1)

4. Handle static vs traced arrays
=================================

generator = FirstOrderDataGenerator(
 model=lambda x: model_fn(x, static_params=True),
 device='gpu'
)
`````


JAX vs PyTorch Differences
~~~~~~~~~~~~~~~~~~~~~~~~~~~


| Feature | PyTorch | JAX |
|---------|---------|-----|
| Arrays | ``torch.Tensor`` | ``jnp.ndarray`` |
| Device API | ``.to(device)`` | ``jax.device_put`` |
| Gradients | ``.backward()`` | ``jax.grad`` |
| JIT | ``torch.jit.script`` | ``@jax.jit`` |
| Parallel | ``DataLoader(num_workers)`` | ``jax.vmap`` |
| TPU | Limited | Native |

---


TensorFlow Implementation
-------------------------



Overview
~~~~~~~~~


**Module**: ``tensorflow_generator.py``

**Features:**
- ``tf.keras.Model`` integration
- ``tf.data.Dataset`` support
- TPU/GPU support
- Eager execution & Graph mode


Quick Start
~~~~~~~~~~~


`````python
import tensorflow as tf
from probly.data_generator.tensorflow_generator import TensorFlowDataGenerator

1. Keras Model
==============

model = tf.keras.models.load_model('pretrained_model.h5')

2. tf.data.Dataset
==================

dataset = tf.data.Dataset.from_tensor_slices((x, y))
dataset = dataset.batch(32)

3. Generator
============

generator = TensorFlowDataGenerator(
 model=model,
 dataset=dataset,
 batch_size=32,
 device='GPU:0' # or 'CPU:0'
)

4. Generate (metrics-based)
===========================

results = generator.generate() # Returns accuracy, confidence, etc.

5. Save
=======

generator.save('tf_results.json')
`````


TensorFlow-Specific Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



1. tf.data.Dataset Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


`````python
Preprocessing pipeline
======================

dataset = tf.data.Dataset.from_tensor_slices((images, labels))
dataset = dataset.map(preprocess_fn)
dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(tf.data.AUTOTUNE)

Direct use with generator
=========================

generator = TensorFlowDataGenerator(
 model=model,
 dataset=dataset,
 batch_size=32
)
`````


2. Strategy API
~~~~~~~~~~~~~~~


`````python
Multi-GPU with MirroredStrategy
==============================

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
 model = create_model()
 generator = TensorFlowDataGenerator(
 model=model,
 dataset=dataset,
 batch_size=32 * strategy.num_replicas_in_sync
 )
`````


3. TPU Support
~~~~~~~~~~~~~~


`````python
TPU configuration
=================

resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)

strategy = tf.distribute.TPUStrategy(resolver)

with strategy.scope():
 model = create_model()
 generator = TensorFlowDataGenerator(model=model, dataset=dataset)
`````


TensorFlow Best Practices
~~~~~~~~~~~~~~~~~~~~~~~~~


`````python
1. Use tf.function for performance
==================================

@tf.function
def model_predict(model, x):
 return model(x, training=False)

2. Prefetch data
================

dataset = dataset.prefetch(tf.data.AUTOTUNE)

3. Mixed precision
==================

tf.keras.mixed_precision.set_global_policy('mixed_float16')

4. Disable training mode
========================

model.trainable = False
`````

---


Framework-Agnostic Implementation
---------------------------------



Overview
~~~~~~~~~


**Module**: ``first_order_datagenerator.py``

**Features:**
- No framework dependencies
- Pure Python implementation
- Maximum compatibility
- Fallback for unknown frameworks


Quick Start
~~~~~~~~~~~


`````python
from probly.data_generator.first_order_datagenerator import (
 FirstOrderDataGenerator,
 SimpleDataLoader
)

1. Any callable model
============================

def my_model(inputs):
 # Your implementation (numpy, custom, etc.)
 return predictions

2. Generator
============

generator = FirstOrderDataGenerator(
 model=my_model,
 batch_size=64,
 output_mode='logits'
)

3. Simple dataset
=================

class MyDataset:
 def __len__(self):
 return len(self.data)

 def __getitem__(self, idx):
 return self.data[idx], self.labels[idx]

4. Generate
=============

distributions = generator.generate_distributions(MyDataset())

5. SimpleDataLoader
===================

loader = SimpleDataLoader(dataset, batch_size=32, shuffle=True)
for batch in loader:
 # Process batch
 pass
`````


When to use Framework-Agnostic?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


**Suitable for:**
- Custom ML Frameworks
- Scikit-learn Models
- NumPy-based Models
- Legacy Code
- Prototyping

**Not suitable for:**
- Production PyTorch/TensorFlow/JAX Code
- Performance-critical applications
- GPU-accelerated training

---


Cross-Framework Compatibility
-----------------------------



JSON format is framework-independent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


`````python
Generate with PyTorch
======================

from probly.data_generator.torch_first_order_generator import FirstOrderDataGenerator
torch_gen = FirstOrderDataGenerator(model=torch_model)
torch_gen.save_distributions('dists.json', distributions)

Load with JAX
=============

from probly.data_generator.jax_first_order_generator import FirstOrderDataGenerator
jax_gen = FirstOrderDataGenerator(model=jax_model)
dists, meta = jax_gen.load_distributions('dists.json')
Works! Distributions are framework-independent
====================================================

`````


Metadata Convention
~~~~~~~~~~~~~~~~~~~


Add framework info to metadata:

`````python
generator.save_distributions(
 'dists.json',
 distributions,
 meta={
 'framework': 'pytorch', # or 'jax', 'tensorflow'
 'device': 'cuda',
 'model_name': 'resnet50',
 # ... additional metadata
 }
)
`````


Conversion Utilities
~~~~~~~~~~~~~~~~~~~~


`````python
PyTorch → NumPy → JAX
=====================

torch_tensor = torch.randn(10, 10)
numpy_array = torch_tensor.cpu().numpy()
jax_array = jnp.array(numpy_array)

JAX → NumPy → PyTorch
=====================

jax_array = jnp.ones((10, 10))
numpy_array = np.array(jax_array)
torch_tensor = torch.from_numpy(numpy_array)
`````

---


Migration between Frameworks
-----------------------------



PyTorch → JAX
~~~~~~~~~~~~~


`````python
Before (PyTorch)
================

import torch
from probly.data_generator.torch_first_order_generator import FirstOrderDataGenerator

torch_model = torch.load('model.pt')
torch_gen = FirstOrderDataGenerator(model=torch_model, device='cuda')
dists = torch_gen.generate_distributions(torch_dataset)

After (JAX)
=============

import jax
from probly.data_generator.jax_first_order_generator import FirstOrderDataGenerator

def jax_model_fn(x):
 # Port PyTorch model to JAX
 return jax_forward_pass(x, converted_params)

jax_gen = FirstOrderDataGenerator(model=jax_model_fn, device='gpu')
dists = jax_gen.generate_distributions(jax_dataset)
`````


Important differences to note:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


| Aspect | PyTorch | JAX |
|--------|---------|-----|
| Model Type | ``nn.Module`` | Function |
| Arrays | ``torch.Tensor`` | ``jnp.ndarray`` |
| Device API | ``.to('cuda')`` | ``jax.device_put()`` |
| In-place Ops | Allowed | Not allowed (functional) |
| Gradients | ``loss.backward()`` | ``jax.grad(loss_fn)`` |


Dataset Conversion
~~~~~~~~~~~~~~~~~~


`````python
PyTorch Dataset → JAX
=====================

class PyTorchDataset(torch.utils.data.Dataset):
 def __getitem__(self, idx):
 return self.data[idx], self.labels[idx]

JAX-compatible version
======================

class JAXDataset:
 def __getitem__(self, idx):
 # Convert to JAX arrays
 return jnp.array(self.data[idx]), jnp.array(self.labels[idx])
`````

---


Performance Comparison
----------------------



Benchmark Setup
~~~~~~~~~~~~~~~


`````python
Same task: Generate distributions for 10,000 samples, 10 classes
================================================================

dataset_size = 10000
num_classes = 10
batch_size = 64
device = 'GPU'
`````


Results (Approximate)
~~~~~~~~~~~~~~~~~~~~~


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


Optimization Tips
~~~~~~~~~~~~~~~~~


`````python
PyTorch
=======

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

JAX
===

@jax.jit
def optimized_model(x):
 return model(x)

TensorFlow
==========

tf.config.optimizer.set_jit(True)
`````

---


Framework Selection Guide
-------------------------



Choose PyTorch when:
~~~~~~~~~~~~~~~~~~~~~~~~

✓ You already use PyTorch
✓ Large community & resources are important
✓ Flexibility in model definition is important
✓ Debugging-friendliness is a priority


Choose JAX when:
~~~~~~~~~~~~~~~~~~~~

✓ Performance is critical
✓ You prefer functional programming
✓ TPU support is needed
✓ Automatic differentiation is central


Choose TensorFlow when:
~~~~~~~~~~~~~~~~~~~~~~~~~~~

✓ Production deployment is important
✓ TensorFlow Serving is used
✓ TPU training is needed
✓ Mobile deployment (TFLite) is planned


Choose Framework-Agnostic when:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

✓ No dependencies are allowed
✓ Custom framework is used
✓ Maximum portability is needed
✓ Only CPU inference is needed

---


Troubleshooting
---------------



PyTorch Issues
~~~~~~~~~~~~~~


**Problem**: CUDA out of memory
`````python
Solution: Reduce batch_size
=================================

generator = FirstOrderDataGenerator(model=model, batch_size=32) # instead of 128

Or: Clear cache periodically
============================

torch.cuda.empty_cache()
`````

**Problem**: Model not on same device as input
`````python
Solution: Explicit device management
====================================

model = model.to(device)
generator = FirstOrderDataGenerator(model=model, device=device)
`````


JAX Issues
~~~~~~~~~~


**Problem**: "Array has been deleted"
`````python
Solution: Copy arrays
===========================

x_copy = jnp.array(x) # Creates copy
`````

**Problem**: JIT compilation error
`````python
Solution: Disable JIT temporarily
=====================================

with jax.disable_jit():
 distributions = generator.generate_distributions(dataset)
`````


Cross-Framework Issues
~~~~~~~~~~~~~~~~~~~~~~


**Problem**: Different precision
`````python
PyTorch: float32 (default)
==========================

JAX: float32 (default)
======================

Solution: Explicitly cast
=======================

jax_array = jnp.array(torch_tensor.numpy(), dtype=jnp.float32)
`````

---


Best Practices Summary
----------------------



General (all frameworks)
~~~~~~~~~~~~~~~~~~~~~~~~~~~


`````python
1. Model in eval mode
======================

model.eval() # PyTorch
model.trainable = False # TensorFlow
====================================


2. Consistent batch size
==========================

batch_size = 64 # Standard, good for most GPUs

3. Progress tracking
====================

distributions = generator.generate_distributions(dataset, progress=True)

4. Document metadata
==========================

meta = {
 'framework': 'pytorch',
 'device': device,
 'timestamp': datetime.now().isoformat(),
 'model_name': 'resnet50',
 'accuracy': 0.95
}
`````


Framework-Specific
~~~~~~~~~~~~~~~~~~~~


`````python
PyTorch
=======

with torch.no_grad():
 distributions = generator.generate_distributions(dataset)

JAX
===

@jax.jit
def optimized_forward(x):
 return model(x)

TensorFlow
==========

@tf.function
def predict(x):
 return model(x, training=False)
````
