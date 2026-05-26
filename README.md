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
`probly` is intended to work with **Python 3.13 and above**. Installation can be done via `pip` and
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
from probly.transformation.ensemble import ensemble
from probly.representer import representer
from probly.quantification import quantify
from probly.evaluation.ood import evaluate_ood

net = ...  # get neural network

# transform model
model = dropout(net, predictor_type='logit_classifier')

train(model)  # train model as usual

# get data
data_id = ...
data_ood = ...

# represent uncertainty
rep = representer(model, num_samples=10)
out_id = rep.represent(data_id)
out_ood = rep.represent(data_ood)

# quantify uncertainty
eu_id = quantify(out_id).epistemic.detach().numpy()
eu_ood = quantify(out_ood).epistemic.detach().numpy()

# evaluate uncertainty
auroc = evaluate_ood(eu_id, eu_ood)
print(auroc)
```

## 🤝 Contributing
Contributions are welcome - see [CONTRIBUTING.md](.github/CONTRIBUTING.md) for guidelines on adding methods, representations, or evaluation protocols.

## 📜 License
This project is licensed under the [MIT License](https://github.com/pwhofman/probly/blob/main/LICENSE).

---
Built with ❤️ by the probly team.
