.. probly documentation master file, created by
   sphinx-quickstart on Tue Apr  8 15:10:07 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

The `probly` Python package
===========================

`probly` is a Python package for **uncertainty representation** and **quantification** for machine learning.

Installation
~~~~~~~~~~~~
`probly` is intended to work with **Python 3.10 and above**. Installation can be done via `pip` and
or `uv`:

.. code-block:: sh

   pip install probly

or

.. code-block:: sh

   uv add probly

Quickstart
~~~~~~~~~~~~
probly makes it very easy to make models uncertainty-aware and perform several downstream tasks:

.. code-block:: python

   import probly
   import torch.nn.functional as F

   net = ... # get neural network
   model = probly.representation.Dropout(net) # make neural network a Dropout model
   train(model) # train model as usual

   data = ... # get data
   preds = model.predict_representation(data) # predict an uncertainty representation
   eu = probly.quantification.classification.mutual_information(preds) # compute model's epistemic uncertainty

   data_ood = ... # get out of distribution data
   preds_ood = model.predict_representation(data_ood)
   eu_ood = probly.quantification.classification.mutual_information(preds_ood)
   auroc = probly.tasks.out_of_distribution_detection(eu, eu_ood) # compute the AUROC score for out of distribution detection



.. toctree::
   :maxdepth: 1
   :caption: Content
   :hidden:

   methods
   references

.. toctree::
   :maxdepth: 0
   :caption: Tutorials
   :hidden:

   notebooks/examples/fashionmnist_ood_ensemble
   notebooks/examples/label_relaxation_calibration
   notebooks/examples/sklearn_selective_prediction
   notebooks/examples/synthetic_regression_dropout
   notebooks/examples/temperature_scaling_calibration
   notebooks/examples/train_bnn_classification
   notebooks/examples/train_evidential_classification
   notebooks/examples/train_evidential_regression

.. toctree::
   :maxdepth: 2
   :caption: API
   :hidden:

   api

.. toctree::
   :maxdepth: 1
   :caption: Development
   :hidden:

   contributing
