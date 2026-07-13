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
from probly.transformation import dropout
from probly.representer import representer
from probly.quantification import quantify
from probly.evaluation.ood import evaluate_ood

net = ...  # get neural network

# transform model: keep dropout active at inference (MC dropout)
model = dropout(net, p=0.25, predictor_type="logit_classifier")

train(model)  # train model as usual

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

Swap `dropout` for `ensemble`, `bayesian`, `laplace`, or any other method below — the rest of the pipeline stays the same.

## 🎲 Methods at a glance

`probly` turns ordinary machine learning models into **uncertainty-aware** ones — transform a model in one line, represent its predictive uncertainty, quantify it with principled measures, and evaluate it on downstream tasks:

- 🧩 **Library-agnostic** — one API across PyTorch, Flax/JAX, scikit-learn, River, and Hugging Face Transformers.
- 🎯 **Model-agnostic** — the same workflow for a linear model, a CNN, a GNN, or an LLM.
- 🔁 **Ante-hoc and post-hoc** — make your existing model uncertainty-aware, or build uncertainty-native models from scratch.

### 👯 Ensembles & sampling

| Method | Reference | PyTorch | Flax | scikit-learn | River |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Deep ensembles (`ensemble`) | [Lakshminarayanan et al., 2017](https://proceedings.neurips.cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html) | ✅ | ✅ | ✅ | ✅ |
| Sub-ensembles (`subensemble`) | [Valdenegro-Toro, 2019](https://arxiv.org/abs/1910.08168) | ✅ | ✅ | | |
| BatchEnsemble (`batchensemble`) | [Wen et al., 2020](https://openreview.net/forum?id=Sklf1yrYDr) | ✅ | ✅ | | |
| MC dropout (`dropout`) | [Gal & Ghahramani, 2016](http://proceedings.mlr.press/v48/gal16.html) | ✅ | ✅ | | |
| MC dropconnect (`dropconnect`) | [Mobiny et al., 2021](https://doi.org/10.1038/s41598-021-84854-x) | ✅ | ✅ | | |
| Class-bias ensembles (`class_bias_ensemble`) | [Löhr et al., 2025](https://doi.org/10.48550/arXiv.2505.22332) | ✅ | | | |

### 🧠 Bayesian & variational

| Method | Reference | PyTorch | Flax | scikit-learn | River |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Variational Bayesian networks (`bayesian`) | [Blundell et al., 2015](http://proceedings.mlr.press/v37/blundell15.html) | ✅ | | | |
| Laplace approximation (`laplace`) | [Daxberger et al., 2021](https://arxiv.org/abs/2106.14806) | ✅ | | | |
| Spectral-normalized GP heads (`sngp`) | [Liu et al., 2020](https://proceedings.neurips.cc/paper/2020/hash/543e83748234f7cbab21aa0ade66565f-Abstract.html) | ✅ | | | |

### 📏 Conformal prediction & calibration

| Method | Reference | PyTorch | Flax | scikit-learn | River |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Split conformal prediction (`conformal_absolute_error`) | [Angelopoulos & Bates, 2021](https://arxiv.org/abs/2107.07511) | ✅ | ✅ | ✅ | |
| Conformal credal sets (`conformal_inner_product`, …) | [Sale et al., 2024](https://openreview.net/forum?id=VJjjNrUi8j) | ✅ | | | |
| Temperature & Platt scaling (`calibration`) | [Guo et al., 2017](http://proceedings.mlr.press/v70/guo17a.html) | ✅ | | ✅ | |

### 📜 Evidential & Dirichlet

| Method | Reference | PyTorch | Flax | scikit-learn | River |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Evidential classification (`evidential_classification`) | [Sensoy et al., 2018](https://proceedings.neurips.cc/paper/2018/hash/a981f2b708044d6fb4a71a1463242520-Abstract.html) | ✅ | | | |
| Evidential regression (`evidential_regression`) | [Amini et al., 2020](https://proceedings.neurips.cc/paper/2020/hash/aab085461de182608ee9f607f3f7d18f-Abstract.html) | ✅ | | | |
| Prior networks (`prior_network`) | [Malinin & Gales, 2018](https://proceedings.neurips.cc/paper/2018/hash/3ea2db50e62ceefceaf70a9d9a56a6f4-Abstract.html) | ✅ | | | |
| Posterior networks (`posterior_network`) | [Charpentier et al., 2020](https://proceedings.neurips.cc/paper/2020/hash/0eac690d7059a8de4b48e90f14510391-Abstract.html) | ✅ | | | |
| Natural posterior networks (`natural_posterior_network`) | [Charpentier et al., 2022](https://openreview.net/forum?id=tV3N0DWMxCg) | ✅ | | | |
| Graph posterior networks (`graph_posterior_network`, …) | [Stadler et al., 2021](https://arxiv.org/abs/2110.14012) | ✅ | | | |
| Dirichlet activations & NIG heads (`dirichlet_*`, `normal_inverse_gamma_head`) | [Malinin et al., 2020](https://arxiv.org/abs/2006.11590) | ✅ | | | |

### 🔮 Deterministic & distance-based

| Method | Reference | PyTorch | Flax | scikit-learn | River |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Deep deterministic uncertainty (`ddu`) | [Mukhoti et al., 2023](https://doi.org/10.1109/CVPR52729.2023.02336) | ✅ | | | |
| Deterministic uncertainty quantification (`duq`) | [van Amersfoort et al., 2020](http://proceedings.mlr.press/v119/van-amersfoort20a.html) | ✅ | | | |
| Mahalanobis distance (`mahalanobis`) | [Lee et al., 2018](https://proceedings.neurips.cc/paper/2018/hash/abdeb6f575ac5c6676b747bca8d09cc2-Abstract.html) | ✅ | | | |
| Direct epistemic uncertainty prediction (`deup`) | [Lahlou et al., 2023](https://openreview.net/forum?id=eGLdVRvvfQ) | ✅ | | | |
| Heteroscedastic networks (`het_net`) | [Collier et al., 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Collier_Correlated_Input-Dependent_Label_Noise_in_Large-Scale_Image_Classification_CVPR_2021_paper.html) | ✅ | | | |

### ☁️ Credal & imprecise probabilities

| Method | Reference | PyTorch | Flax | scikit-learn | River |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Credal wrapper (`credal_wrapper`)¹ | [Wang et al., 2025](https://openreview.net/forum?id=cv2iMNWCsh) | ✅ | | | |
| Credal Bayesian deep learning (`credal_bnn`)¹ | [Caprio et al., 2024](https://openreview.net/forum?id=4NHF9AC5ui) | ✅ | | | |
| Credal ensembling (`credal_ensembling`)¹ | [Nguyen et al., 2025](https://doi.org/10.1007/s10994-024-06703-y) | ✅ | | | |
| Credal nets (`credal_net`)¹ | [Sale et al., 2024](https://openreview.net/forum?id=VJjjNrUi8j) | ✅ | | | |
| Relative-likelihood credal prediction (`credal_relative_likelihood`)¹ | [Löhr et al., 2025](https://doi.org/10.48550/arXiv.2505.22332) | ✅ | | | |
| Density-ratio credal regions (`dare`)¹ | [de Mathelin et al., 2023](https://doi.org/10.48550/arXiv.2304.04042) | ✅ | | | |
| Efficient credal prediction (`efficient_credal_prediction`)² | [Hofman et al., 2026](https://doi.org/10.48550/arXiv.2603.08495) | ✅ | | | |

¹ Built on the `ensemble` transformation and usable wherever it is. ² Also has a pure NumPy implementation.

🤖 **LLM uncertainty**, too: semantic entropy and spectral uncertainty for Hugging Face text generation models — [see below](#-uncertainty-for-llms).

Browse the full [API reference](https://pwhofman.github.io/probly/stable/api.html) and the [examples gallery](https://pwhofman.github.io/probly/stable/examples.html) for the complete picture.

## 🤖 Uncertainty for LLMs

Does your language model *know* when it doesn't know? `probly` brings uncertainty quantification to text generation: sample multiple answers, cluster them by meaning with an NLI model, and decompose the resulting **semantic entropy** ([Kuhn et al., 2023](https://arxiv.org/abs/2302.09664)) into its aleatoric and epistemic parts.

<div align="center">
  <img src="docs/source/_static/readme/llm_uncertainty_demo.svg" alt="Animated demo: a factual question collapses to one meaning with zero uncertainty, while a trick question scatters into seven meanings with high uncertainty, flagging a likely hallucination" width="100%" />
</div>

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

The factual question collapses into a single semantic cluster — the model is certain. The trick question scatters across many clusters: high uncertainty, flagging a likely hallucination. See [`examples/llm/semantic_entropy.py`](examples/llm/semantic_entropy.py) for the full pipeline and [`examples/llm/spectral_uncertainty.py`](examples/llm/spectral_uncertainty.py) for an embedding-based alternative.

## 📖 Documentation

The [documentation](https://pwhofman.github.io/probly) covers the full workflow, including a [user guide](https://pwhofman.github.io/probly/stable/user_guide.html), an [examples gallery](https://pwhofman.github.io/probly/stable/examples.html), and the [API reference](https://pwhofman.github.io/probly/stable/api.html).

## 🤝 Contributing
Contributions are welcome - see [CONTRIBUTING.md](.github/CONTRIBUTING.md) for guidelines on adding methods, representations, or evaluation protocols.

## 📜 License
This project is licensed under the [MIT License](https://github.com/pwhofman/probly/blob/main/LICENSE).

---
Built with ❤️ by the probly team.
