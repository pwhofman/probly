Uncertainty estimation with Dropout on MNIST
===========================================

What you will learn
-------------------

In this tutorial, you will learn how to train a simple convolutional neural network with Dropout on the MNIST dataset using PyTorch. You will also learn how to perform Monte Carlo Dropout during inference and how to estimate predictive uncertainty by computing the mean and standard deviation across multiple stochastic forward passes.

Prerequisites
-------------

This example requires Python 3.8 or later, as well as the packages ``torch``, ``torchvision``, and ``matplotlib``.

You can install the dependencies by running:

.. code-block:: bash

   pip install torch torchvision matplotlib

Step 1: Load the MNIST dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We begin by loading the MNIST dataset and creating dataloaders for both training and testing.

.. code-block:: python

   import torch
   from torch import nn, optim
   from torch.utils.data import DataLoader
   from torchvision import datasets, transforms

   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.1307,), (0.3081,))
   ])

   train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
   test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

   train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
   test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

Step 2: Define a CNN with Dropout
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next, we define a small convolutional network that includes several Dropout layers. During training, these layers behave normally; however, during evaluation we will keep them active in order to obtain uncertainty estimates.

.. code-block:: python

   class DropoutCNN(nn.Module):
       def __init__(self, p: float = 0.5):
           super().__init__()
           self.conv = nn.Sequential(
               nn.Conv2d(1, 32, 3, padding=1),
               nn.ReLU(),
               nn.MaxPool2d(2),
               nn.Dropout(p),

               nn.Conv2d(32, 64, 3, padding=1),
               nn.ReLU(),
               nn.MaxPool2d(2),
               nn.Dropout(p),
           )
           self.fc = nn.Sequential(
               nn.Flatten(),
               nn.Linear(64 * 7 * 7, 128),
               nn.ReLU(),
               nn.Dropout(p),
               nn.Linear(128, 10),
           )

       def forward(self, x):
           x = self.conv(x)
           x = self.fc(x)
           return x


   model = DropoutCNN(p=0.5).to(device)

Step 3: Train the model
~~~~~~~~~~~~~~~~~~~~~~~

For demonstration purposes, we train the model for only a few epochs. In a real application, you would typically train for longer to achieve higher accuracy.

.. code-block:: python

   def train_one_epoch(model, loader, optimizer, criterion):
       model.train()
       running_loss = 0.0
       correct = 0
       total = 0

       for x, y in loader:
           x, y = x.to(device), y.to(device)
           optimizer.zero_grad()
           logits = model(x)
           loss = criterion(logits, y)
           loss.backward()
           optimizer.step()

           running_loss += loss.item() * x.size(0)
           preds = logits.argmax(dim=1)
           correct += (preds == y).sum().item()
           total += x.size(0)

       return running_loss / total, correct / total


   criterion = nn.CrossEntropyLoss()
   optimizer = optim.Adam(model.parameters(), lr=1e-3)

   for epoch in range(3):
       train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
       print(f"Epoch {epoch+1}: loss={train_loss:.4f}, acc={train_acc:.4f}")

Step 4: Enable Monte Carlo Dropout
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To estimate uncertainty, we perform multiple forward passes with dropout activated during inference. The function below forces all ``Dropout`` layers into training mode and returns both mean probabilities and standard deviations.

.. code-block:: python

   import torch.nn.functional as F

   def enable_dropout(model):
       for m in model.modules():
           if isinstance(m, nn.Dropout):
               m.train()


   def mc_dropout_predict(model, x, n_samples: int = 20):
       model.eval()
       enable_dropout(model)

       probs = []
       with torch.no_grad():
           for _ in range(n_samples):
               logits = model(x)
               p = F.softmax(logits, dim=-1)
               probs.append(p.unsqueeze(0))

       probs = torch.cat(probs, dim=0)
       mean_probs = probs.mean(dim=0)
       std_probs = probs.std(dim=0)
       return mean_probs, std_probs

Step 5: Visualize uncertainty
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

[Platzhalter: MC Dropout PNG einfügen]

Summary
-------

In this example, you learned that Dropout can be used not only as a regularization technique but also as an approximate Bayesian inference method. By enabling dropout during evaluation and performing multiple forward passes, you obtained a distribution over predictions. The mean probability reflects the predicted class, while the standard deviation indicates how uncertain the model is. This approach is useful in real-world scenarios where uncertainty awareness is important, such as medical diagnosis or autonomous decision-making.

---

Creating a SubEnsemble for large models
=======================================

What you will learn
-------------------

In this tutorial, you will learn how to build an ensemble of models in PyTorch and how to create a smaller ``SubEnsemble`` that uses only a subset of the full ensemble. You will also see how this approach enables a trade-off between accuracy and inference speed without requiring additional training.

Concept overview
----------------

Large ensembles can provide strong performance but may be too slow or expensive during inference. A ``SubEnsemble`` allows you to reuse selected members of a trained ensemble and create a lighter version that produces predictions more quickly. This is especially useful in latency-sensitive environments such as real-time applications or deployment on resource-limited hardware.

Step 1: Define a simple base model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We begin by defining a small fully connected network that can be used as a base model. This network expects MNIST-style input with a flattened 28×28 pixel image.

.. code-block:: python

   import torch
   from torch import nn
   import torch.nn.functional as F

   class SmallMLP(nn.Module):
       def __init__(self, input_dim=28*28, hidden_dim=128, num_classes=10):
           super().__init__()
           self.net = nn.Sequential(
               nn.Linear(input_dim, hidden_dim),
               nn.ReLU(),
               nn.Linear(hidden_dim, num_classes),
           )

       def forward(self, x):
           x = x.view(x.size(0), -1)
           return self.net(x)

Step 2: Create an Ensemble wrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next, we define an ``Ensemble`` class that contains multiple independent models and averages their outputs.

.. code-block:: python

   class Ensemble(nn.Module):
       def __init__(self, base_models):
           super().__init__()
           self.models = nn.ModuleList(base_models)

       def forward(self, x):
           logits_list = []
           for m in self.models:
               logits_list.append(m(x))
           logits = torch.stack(logits_list, dim=0)
           return logits.mean(dim=0)

Step 3: Create a SubEnsemble
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A ``SubEnsemble`` references only a subset of the full ensemble’s models. This allows you to choose which members to include without modifying or retraining them.

.. code-block:: python

   class SubEnsemble(nn.Module):
       def __init__(self, full_ensemble: Ensemble, indices):
           super().__init__()
           selected = [full_ensemble.models[i] for i in indices]
           self.models = nn.ModuleList(selected)

       def forward(self, x):
           logits_list = [m(x) for m in self.models]
           logits = torch.stack(logits_list, dim=0)
           return logits.mean(dim=0)

Step 4: Instantiate and train the ensemble
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In a real scenario, each model would be trained independently using different seeds or data shuffling. The following example only shows how to set up the full ensemble.

.. code-block:: python

   num_members = 5
   base_models = [SmallMLP() for _ in range(num_members)]
   full_ensemble = Ensemble(base_models)

   def train_member(model, train_loader, epochs=1):
       optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
       criterion = nn.CrossEntropyLoss()
       model.train()
       for epoch in range(epochs):
           for x, y in train_loader:
               optimizer.zero_grad()
               logits = model(x)
               loss = criterion(logits, y)
               loss.backward()
               optimizer.step()

   # In practice:
   # for m in full_ensemble.models:
   #     train_member(m, train_loader, epochs=5)

Step 5: Evaluate the SubEnsemble
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We now create a smaller SubEnsemble that uses only two of the five original models and compare its performance to the full ensemble.

.. code-block:: python

   fast_subensemble = SubEnsemble(full_ensemble, indices=[0, 1])

   def evaluate(model, data_loader):
       model.eval()
       correct = 0
       total = 0
       with torch.no_grad():
           for x, y in data_loader:
               logits = model(x)
               preds = logits.argmax(dim=1)
               correct += (preds == y).sum().item()
               total += x.size(0)
       return correct / total

   # Example usage:
   # full_acc = evaluate(full_ensemble, test_loader)
   # sub_acc = evaluate(fast_subensemble, test_loader)

Visual result
-------------

[Platzhalter: SubEnsemble PNG einfügen]

Summary
-------

In this example, you learned how to construct an ensemble of models and how to derive a smaller ``SubEnsemble`` from it without retraining. The full ensemble provides the highest accuracy, while the SubEnsemble offers faster inference by using fewer models. This approach enables flexible deployment strategies, allowing you to choose between maximum performance and reduced computational cost depending on the application requirements.
