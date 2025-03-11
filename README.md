# probly: Uncertainty Quantification for Machine Learning

[![PyPI version](https://badge.fury.io/py/probly.svg)](https://badge.fury.io/py/probly)
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)
[![Coverage Status](https://coveralls.io/repos/github/KIuML/probly/badge.svg?branch=main)](https://coveralls.io/github/mmschlk/probly?branch=main)
[![Tests](https://github.com/KIuML/probly/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/mmschlk/probly/actions/workflows/unit-tests.yml)
[![Read the Docs](https://readthedocs.org/projects/probly/badge/?version=latest)](https://probly.readthedocs.io/en/latest/?badge=latest)

[![PyPI Version](https://img.shields.io/pypi/pyversions/probly.svg)](https://pypi.org/project/probly)
[![PyPI status](https://img.shields.io/pypi/status/probly.svg?color=blue)](https://pypi.org/project/probly)
[![PePy](https://static.pepy.tech/badge/probly?style=flat-square)](https://pepy.tech/project/probly)

[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen)
![Last Commit](https://img.shields.io/github/last-commit/KIuML/probly)

> This is **probly** a dog, could also be a cat though ...


## 🛠️ Install
`probly` is intended to work with **Python 3.13 and above**. Installation can be done via `pip`:

```sh
pip install probly
```

## ⭐ Quickstart

Using `probly` is as easy as 1, 2, 3:

```python
import probly
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor

# get your model and data
data = fetch_california_housing()
X, y = data.data, data.target
model = RandomForestRegressor()
model.fit(X, y)

# quantify the uncertainty
quantifyer = probly.Quantifyer(model, X)
quantifyer.quantify()
```

## 📜 License
This project is licensed under the [MIT License](https://github.com/KIuML/probly/blob/main/LICENSE).

---
Built with ❤️ by the probly team.
