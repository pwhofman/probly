# probly: Uncertainty Representation and Quantification for Machine Learning
<div align="center">
<picture>
  <source srcset="docs/source/_static/logo/logo_dark.png" media="(prefers-color-scheme: dark)">
  <source srcset="docs/source/_static/logo/logo_light.png" media="(prefers-color-scheme: light)">
  <img src="docs/source/_static/logo/logo_light.png" alt="probly logo" width="300" />
</picture>

[![PyPI version](https://badge.fury.io/py/probly.svg)](https://badge.fury.io/py/probly)
[![PyPI status](https://img.shields.io/pypi/status/probly.svg?color=blue)](https://pypi.org/project/probly)
[![PePy](https://static.pepy.tech/badge/probly?style=flat-square)](https://pepy.tech/project/probly)
[![codecov](https://codecov.io/gh/pwhofman/probly/branch/main/graph/badge.svg)](https://codecov.io/gh/pwhofman/probly)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen)](.github/CONTRIBUTING.md)
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)
</div>

## 🛠️ Install
`probly` is intended to work with **Python 3.12 and above**. Installation can be done via `pip` and
or `uv`:

```sh
pip install probly
```

```sh
uv add probly
```

## ⭐ Quickstart

`probly` makes it very easy to make models uncertainty-aware and perform several downstream tasks:

```python
import probly
import torch.nn.functional as F

net = ...  # get neural network
model = probly.transformation.dropout(net)  # make neural network a Dropout model
train(model)  # train model as usual

data = ...  # get data
data_ood = ...  # get out of distribution data
sampler = probly.representation.Sampler(model, num_samples=20)
sample = sampler.predict(data) # predict an uncertainty representation
sample_ood = sampler.predict(data_ood)

eu = probly.quantification.classification.mutual_information(sample)  # quantify model's epistemic uncertainty
eu_ood = probly.quantification.classification.mutual_information(sample_ood)

auroc = probly.evaluation.tasks.out_of_distribution_detection(eu, eu_ood)  # evaluate model's uncertainty
```

## 📜 License
This project is licensed under the [MIT License](https://github.com/pwhofman/probly/blob/main/LICENSE).

---
Built with ❤️ by the probly team.
