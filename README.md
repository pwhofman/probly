<div align="center">
<picture>
  <source srcset="https://raw.githubusercontent.com/pwhofman/probly/main/docs/source/_static/logo/logo_dark.png" media="(prefers-color-scheme: dark)">
  <source srcset="https://raw.githubusercontent.com/pwhofman/probly/main/docs/source/_static/logo/logo_light.png" media="(prefers-color-scheme: light)">
  <img src="https://raw.githubusercontent.com/pwhofman/probly/main/docs/source/_static/logo/logo_light.png" alt="probly logo" width="300" />
</picture>

### Uncertainty Representation and Quantification for Machine Learning

[![PyPI version](https://badge.fury.io/py/probly.svg)](https://badge.fury.io/py/probly)
[![PyPI status](https://img.shields.io/pypi/status/probly.svg?color=blue)](https://pypi.org/project/probly)
[![Python versions](https://img.shields.io/pypi/pyversions/probly.svg)](https://pypi.org/project/probly)
[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://pwhofman.github.io/probly)
[![PePy](https://static.pepy.tech/badge/probly?style=flat-square)](https://pepy.tech/project/probly)
[![codecov](https://codecov.io/gh/pwhofman/probly/branch/main/graph/badge.svg)](https://codecov.io/gh/pwhofman/probly)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen)](.github/CONTRIBUTING.md)
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)
</div>

> *probably this is the right answer.*

`probly` turns ordinary machine learning models into **uncertainty-aware** ones. Transform a model in one line, represent its predictive uncertainty, quantify it with principled measures, and evaluate it on downstream tasks like out-of-distribution detection — from logistic regression to large language models.

- 🧩 **Library-agnostic** — one API across PyTorch, Flax/JAX, scikit-learn, River, and Hugging Face Transformers.
- 🎯 **Model-agnostic** — the same workflow for a linear model, a CNN, a GNN, or an LLM.
- 🔁 **Ante-hoc and post-hoc** — make your existing model uncertainty-aware, or build uncertainty-native models from scratch.

## 🛠️ Install

`probly` is intended to work with **Python 3.13 and above**. Installation can be done via `pip` or `uv`:

```sh
pip install probly
```

```sh
uv add probly
```

Backends are optional: `probly` only imports PyTorch, JAX, or scikit-learn if your model actually needs them.

## 🎲 Methods at a glance

`probly` ships implementations of a wide range of uncertainty quantification methods, organized around a common transform → represent → quantify → evaluate workflow:

| Method | PyTorch | Flax | scikit-learn | River |
| :--- | :---: | :---: | :---: | :---: |
| **Ensembles & sampling** | | | | |
| Deep ensembles (`ensemble`) | ✅ | ✅ | ✅ | ✅ |
| Sub-ensembles (`subensemble`) | ✅ | ✅ | | |
| BatchEnsemble (`batchensemble`) | ✅ | ✅ | | |
| MC dropout (`dropout`) | ✅ | ✅ | | |
| MC dropconnect (`dropconnect`) | ✅ | ✅ | | |
| Class-bias ensembles (`class_bias_ensemble`) | ✅ | | | |
| **Bayesian & variational** | | | | |
| Variational Bayesian networks (`bayesian`) | ✅ | | | |
| Laplace approximation (`laplace`) | ✅ | | | |
| Spectral-normalized GP heads (`sngp`) | ✅ | | | |
| **Conformal prediction** | | | | |
| Split conformal prediction (`conformal_absolute_error`) | ✅ | ✅ | ✅ | |
| Conformal credal sets (`conformal_inner_product`, …) | ✅ | | | |
| **Calibration** | | | | |
| Temperature & Platt scaling (`calibration`) | ✅ | | ✅ | |
| **Evidential & Dirichlet** | | | | |
| Evidential classification & regression (`evidential_*`) | ✅ | | | |
| Prior networks (`prior_network`) | ✅ | | | |
| Posterior networks (`posterior_network`, `natural_posterior_network`) | ✅ | | | |
| Graph posterior networks (`graph_posterior_network`, …) | ✅ | | | |
| Dirichlet activations & NIG heads (`dirichlet_*`, `normal_inverse_gamma_head`) | ✅ | | | |
| **Deterministic & distance-based** | | | | |
| DDU, DUQ, DEUP, HetNet, Mahalanobis (`ddu`, `duq`, `deup`, `het_net`, `mahalanobis`) | ✅ | | | |
| **Credal & imprecise probabilities** | | | | |
| Credal wrappers, nets & ensembling (`credal_*`)¹ | ✅ | | | |
| Density-ratio credal regions (`dare`)¹ | ✅ | | | |
| Efficient credal prediction (`efficient_credal_prediction`)² | ✅ | | | |

¹ Built on the `ensemble` transformation and usable wherever it is. ² Also has a pure NumPy implementation.

🤖 **LLM uncertainty**, too: semantic entropy and spectral uncertainty for Hugging Face text generation models — [see below](#-uncertainty-for-llms).

Browse the full [API reference](https://pwhofman.github.io/probly/stable/api.html) and the [examples gallery](https://pwhofman.github.io/probly/stable/examples.html) for the complete picture.

## ⭐ Quickstart

`probly` makes it very easy to make models uncertainty-aware and perform downstream tasks:

```python
from probly.transformation import dropout
from probly.representer import representer
from probly.quantification import quantify
from probly.evaluation.ood import evaluate_ood

net = ...  # your neural network

# make the model uncertainty-aware: keep dropout active at inference (MC dropout)
model = dropout(net, p=0.25, predictor_type="logit_classifier")

train(model)  # train as usual

# represent uncertainty: turn stochastic forward passes into a predictive distribution
rep = representer(model, num_samples=50)
out_id = rep.represent(data_id)
out_ood = rep.represent(data_ood)

# quantify epistemic (model) uncertainty
eu_id = quantify(out_id).epistemic.detach().numpy()
eu_ood = quantify(out_ood).epistemic.detach().numpy()

# evaluate: does uncertainty separate in-distribution from out-of-distribution?
print(evaluate_ood(eu_id, eu_ood))  # {'auroc': ...}
```

Swap `dropout` for `ensemble`, `bayesian`, `laplace`, or any other method above — the rest of the pipeline stays the same.

## 🤖 Uncertainty for LLMs

Does your language model *know* when it doesn't know? `probly` brings uncertainty quantification to text generation: sample multiple answers, cluster them by meaning with an NLI model, and decompose the resulting **semantic entropy** into its aleatoric and epistemic parts.

```python
from probly.quantification import decompose
from probly.representation.distribution.torch_categorical import TorchCategoricalDistributionSample
from probly.representer.clarifier.huggingface import HFQuestionClarifier
from probly.representer.sampler.huggingface import HFTextGenerationSampler, load_model
from probly.representer.semantic_clustering.huggingface import HFGreedySemanticClusterer

model, tokenizer = load_model("google/gemma-4-E2B-it")
clarifier = HFQuestionClarifier(model, tokenizer, num_samples=2)  # rephrase each question
sampler = HFTextGenerationSampler(model, tokenizer, num_samples=10, temperature=0.7)
clusterer = HFGreedySemanticClusterer.from_model_name("microsoft/deberta-base-mnli")

questions = ["What is the capital of France?", "Who was the first person to walk on Mars?"]

answers = sampler(clarifier(questions))  # sample answers per clarified question
semantic = clusterer(answers)            # cluster answers by meaning (NLI)

# densify the semantic clusters and decompose the semantic entropy
dense = TorchCategoricalDistributionSample(tensor=semantic.tensor.to_dense(), sample_dim=semantic.sample_dim)
uq = decompose(dense)

for question, tu, au, eu in zip(questions, uq.total, uq.aleatoric, uq.epistemic):
    print(f"{question:<45} TU={tu:.3f}  AU={au:.3f}  EU={eu:.3f}")
```

```text
What is the capital of France?                TU=0.000  AU=0.000  EU=0.000
Who was the first person to walk on Mars?     TU=1.561  AU=1.386  EU=0.175
```

The factual question collapses into a single semantic cluster — the model is certain. The trick question scatters across many clusters: high uncertainty, flagging a likely hallucination. See [`examples/llm/semantic_entropy.py`](examples/llm/semantic_entropy.py) for the full pipeline and [`examples/llm/spectral_uncertainty.py`](examples/llm/spectral_uncertainty.py) for an embedding-based alternative.

## 📈 Conformal regression

Prediction intervals with guaranteed coverage in four lines — wrap any trained regressor, calibrate on held-out data, and predict:

```python
from probly.calibrator import calibrate
from probly.method.conformal import conformal_absolute_error
from probly.representer import representer
from probly.evaluation import coverage

conformal_model = conformal_absolute_error(model)               # wrap a trained regressor
calibrated = calibrate(conformal_model, 0.1, y_calib, X_calib)  # target 90% coverage
intervals = representer(calibrated).predict(X_test)             # [pred - q, pred + q]
print(coverage(intervals, y_test))                              # 0.916
```

<div align="center">
<img src="https://raw.githubusercontent.com/pwhofman/probly/main/docs/source/_static/readme/conformal_regression.png" alt="Conformal regression with uncertainty bands" width="700" />
</div>

## 📖 Documentation

The [documentation](https://pwhofman.github.io/probly) covers the full workflow, including a [user guide](https://pwhofman.github.io/probly/stable/user_guide.html), an [examples gallery](https://pwhofman.github.io/probly/stable/examples.html), and the [API reference](https://pwhofman.github.io/probly/stable/api.html).

## 🤝 Contributing

Contributions are welcome — see [CONTRIBUTING.md](.github/CONTRIBUTING.md) for guidelines on adding methods, representations, or evaluation protocols.

## 📜 License

This project is licensed under the [MIT License](https://github.com/pwhofman/probly/blob/main/LICENSE).

---
Built with ❤️ by the probly team.
