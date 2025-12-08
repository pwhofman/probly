##########################
Examples and tutorials
##########################

This section contains practical, end-to-end examples that demonstrate how **probly** can be used in real applications. Each tutorial provides a guided workflow from start to finish, including model transformation, execution and interpretation of the results. The examples also directly correspond to the advanced modeling patterns discussed in :doc:`Advanced Topics <advanced_topics>`
, providing at least one worked example for concepts such as uncertainty-aware transformations, ensemble methods, and mixed-model workflows.They are self-contained and can be adapted to individual projects and datasets.

Users who are new to probly are encouraged to begin with the introductory Dropout example before exploring ensemble-based methods and more advanced uncertainty-aware workflows.

.. contents::
   :local:
   :depth: 2


1. Uncertainty estimation with Dropout on MNIST
===============================================

What you will learn
-------------------

In this tutorial, you will learn how to use **probly** to make a standard neural network uncertainty-aware with the Dropout transformation. You start from a conventional PyTorch model trained on MNIST and then apply probly so that Dropout remains active during inference. By running multiple stochastic forward passes, you obtain a distribution of predictions and estimate predictive uncertainty.

Prerequisites
-------------

This example requires Python 3.8 or later and the packages ``probly``, ``torch`` and ``torchvision``:

.. code-block:: bash

   pip install probly torch torchvision

Step 1: Load the MNIST dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from torch.utils.data import DataLoader
   from torchvision import datasets, transforms

   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   transform = transforms.ToTensor()

   train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
   test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

   train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
   test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

Step 2: Define a base convolutional model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch.nn as nn

   class SimpleCNN(nn.Module):
       def __init__(self):
           super().__init__()
           self.net = nn.Sequential(
               nn.Conv2d(1, 32, 3, padding=1),
               nn.ReLU(),
               nn.MaxPool2d(2),
               nn.Conv2d(32, 64, 3, padding=1),
               nn.ReLU(),
               nn.MaxPool2d(2),
               nn.Flatten(),
               nn.Linear(64 * 7 * 7, 10),
           )

       def forward(self, x):
           return self.net(x)

   model = SimpleCNN().to(device)

Step 3: Train the model briefly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch.optim as optim
   import torch.nn.functional as F

   optimizer = optim.Adam(model.parameters(), lr=1e-3)

   model.train()
   for epoch in range(1):
       for x, y in train_loader:
           x, y = x.to(device), y.to(device)
           optimizer.zero_grad()
           logits = model(x)
           loss = F.cross_entropy(logits, y)
           loss.backward()
           optimizer.step()

Step 4: Apply probly’s Dropout transformation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from probly.transformation import dropout

   prob_model = dropout(model, p=0.5, enable_at_eval=True)

Step 5: Perform Monte Carlo inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch.nn.functional as F
   import torch

   @torch.no_grad()
   def mc_predict(model, x, samples=30):
       model.eval()  # Dropout remains active
       probs = []
       for _ in range(samples):
           logits = model(x)
           p = F.softmax(logits, dim=-1)
           probs.append(p.unsqueeze(0))
       probs = torch.cat(probs, dim=0)
       return probs.mean(0), probs.std(0)

   x_batch, _ = next(iter(test_loader))
   x_batch = x_batch.to(device)

   mean_probs, std_probs = mc_predict(prob_model, x_batch[0:1])

   print("Mean probabilities:", mean_probs.squeeze().cpu())
   print("Std probabilities:", std_probs.squeeze().cpu())

Step 6: Visualize uncertainty
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: mc_dropout_example.png
   :width: 550px
   :align: center
   :alt: Monte Carlo Dropout uncertainty visualization

Summary
-------

In this example, probly was used to transform a standard neural network into an uncertainty-aware model. Dropout remains active during inference and multiple forward passes allow you to obtain predictive uncertainty without modifying the original architecture.


2. Creating a SubEnsemble with probly
=====================================

What you will learn
-------------------

In this tutorial, you will learn how to construct an ensemble using probly and how to derive a smaller ``SubEnsemble`` without retraining. This allows you to trade inference speed for accuracy in a controlled way.

Step 1: Define a simple base model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   import torch.nn as nn
   from torch.utils.data import DataLoader
   from torchvision import datasets, transforms

   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   transform = transforms.ToTensor()

   train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
   test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

   train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
   test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

   class SmallMLP(nn.Module):
       def __init__(self):
           super().__init__()
           self.net = nn.Sequential(
               nn.Flatten(),
               nn.Linear(28 * 28, 128),
               nn.ReLU(),
               nn.Linear(128, 10),
           )

       def forward(self, x):
           return self.net(x)

Step 2: Create an Ensemble with probly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from probly.ensemble import Ensemble

   num_members = 5
   members = [SmallMLP().to(device) for _ in range(num_members)]
   ensemble = Ensemble(members)

Step 3: Train ensemble members
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch.nn.functional as F
   import torch.optim as optim

   def train_member(model, loader, epochs=1):
       optimizer = optim.Adam(model.parameters(), lr=1e-3)
       model.train()
       for _ in range(epochs):
           for x, y in loader:
               x, y = x.to(device), y.to(device)
               optimizer.zero_grad()
               logits = model(x)
               loss = F.cross_entropy(logits, y)
               loss.backward()
               optimizer.step()

   for m in members:
       train_member(m, train_loader, epochs=1)

Step 4: Evaluate the Ensemble
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @torch.no_grad()
   def evaluate(model, loader):
       model.eval()
       correct = 0
       total = 0
       for x, y in loader:
           x, y = x.to(device), y.to(device)
           preds = model(x).argmax(dim=1)
           correct += (preds == y).sum().item()
           total += x.size(0)
       return correct / total

   full_acc = evaluate(ensemble, test_loader)
   print("Ensemble accuracy:", full_acc)

Step 5: Create and evaluate a SubEnsemble
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from probly.ensemble import SubEnsemble

   sub = SubEnsemble(ensemble, indices=[0, 1])
   sub_acc = evaluate(sub, test_loader)
   print("SubEnsemble accuracy:", sub_acc)

Visual result
~~~~~~~~~~~~~

.. image:: subensemble_comparison.png
   :width: 500px
   :align: center
   :alt: Accuracy comparison between full ensemble and SubEnsemble

Summary
-------

In this example, probly was used to create both a full Ensemble and a SubEnsemble without retraining. The SubEnsemble provides faster inference while maintaining useful accuracy, enabling deployment-time flexibility.


3. MixedEnsemble with probly
============================

What you will learn
-------------------

In this tutorial, you will learn how to build a MixedEnsemble using probly by combining different neural network architectures into a single probabilistic ensemble. You will compare it to a homogeneous ensemble and observe how model diversity may influence performance.

Step 1: Prepare data
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from torch.utils.data import DataLoader
   from torchvision import datasets, transforms

   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   transform = transforms.ToTensor()

   train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
   test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

   train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
   test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

Step 2: Define different architectures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch.nn as nn
   import torch.nn.functional as F

   class SmallCNN(nn.Module):
       def __init__(self):
           super().__init__()
           self.conv = nn.Sequential(
               nn.Conv2d(1, 32, 3, padding=1),
               nn.ReLU(),
               nn.MaxPool2d(2),
               nn.Conv2d(32, 64, 3, padding=1),
               nn.ReLU(),
               nn.MaxPool2d(2),
           )
           self.head = nn.Sequential(
               nn.Flatten(),
               nn.Linear(64 * 7 * 7, 128),
               nn.ReLU(),
               nn.Linear(128, 10),
           )

       def forward(self, x):
           x = self.conv(x)
           x = self.head(x)
           return x

   class SmallMLP(nn.Module):
       def __init__(self):
           super().__init__()
           self.net = nn.Sequential(
               nn.Flatten(),
               nn.Linear(28 * 28, 256),
               nn.ReLU(),
               nn.Linear(256, 10),
           )

       def forward(self, x):
           return self.net(x)

Step 3: Create Ensemble and MixedEnsemble
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from probly.ensemble import Ensemble, MixedEnsemble

   cnn_members = [SmallCNN().to(device) for _ in range(3)]
   cnn_ensemble = Ensemble(cnn_members)

   mixed_members = [
       SmallCNN().to(device),
       SmallCNN().to(device),
       SmallMLP().to(device),
   ]
   mixed_ensemble = MixedEnsemble(mixed_members)

Step 4: Train all members
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch.optim as optim

   def train(model, loader, epochs=1):
       optimizer = optim.Adam(model.parameters(), lr=1e-3)
       model.train()
       for _ in range(epochs):
           for x, y in loader:
               x, y = x.to(device), y.to(device)
               optimizer.zero_grad()
               logits = model(x)
               loss = F.cross_entropy(logits, y)
               loss.backward()
               optimizer.step()

   for m in cnn_members:
       train(m, train_loader, epochs=1)

   for m in mixed_members:
       train(m, train_loader, epochs=1)

Step 5: Evaluate both ensembles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @torch.no_grad()
   def evaluate(model, loader):
       model.eval()
       correct = 0
       total = 0
       for x, y in loader:
           x, y = x.to(device), y.to(device)
           logits = model(x)
           preds = logits.argmax(dim=1)
           correct += (preds == y).sum().item()
           total += x.size(0)
       return correct / total

   acc_cnn = evaluate(cnn_ensemble, test_loader)
   acc_mixed = evaluate(mixed_ensemble, test_loader)

   print("Homogeneous CNN Ensemble accuracy:", acc_cnn)
   print("MixedEnsemble accuracy:", acc_mixed)

Visual result
-------------

.. image:: mixed_ensemble_comparison.png
   :width: 500px
   :align: center
   :alt: Accuracy comparison between homogeneous and mixed ensembles

Summary
-------

In this example, you used probly to construct both a homogeneous ensemble and a MixedEnsemble combining different model types. The MixedEnsemble may capture complementary model behaviour, which can improve robustness and calibration in some settings.
