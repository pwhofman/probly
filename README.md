# probly: Uncertainty Representation and Quantification for Machine Learning
<div align="center">
<picture>
  <source srcset="docs/source/_static/readme/ecosystem_dark.png" media="(prefers-color-scheme: dark)">
  <source srcset="docs/source/_static/readme/ecosystem_light.png" media="(prefers-color-scheme: light)">
  <img src="docs/source/_static/readme/ecosystem_light.png" alt="probly at the center of a ring of the libraries it works with: PyTorch, Flax/JAX, scikit-learn, River, Hugging Face and NumPy as native backends, plus Lightning, torch-uncertainty, PyTorch Geometric and laplace-torch as integrations" width="68%" />
</picture>

[![PyPI version](https://badge.fury.io/py/probly.svg)](https://badge.fury.io/py/probly)
[![Python Versions](https://img.shields.io/pypi/pyversions/probly.svg)](https://pypi.org/project/probly)
[![PyPI status](https://img.shields.io/pypi/status/probly.svg?color=blue)](https://pypi.org/project/probly)
[![PePy](https://static.pepy.tech/badge/probly?style=flat-square)](https://pepy.tech/project/probly)
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)

[![Tests](https://github.com/pwhofman/probly/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/pwhofman/probly/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/pwhofman/probly/branch/main/graph/badge.svg)](https://codecov.io/gh/pwhofman/probly)
[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://pwhofman.github.io/probly)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen)](.github/CONTRIBUTING.md)
[![Last Commit](https://img.shields.io/github/last-commit/pwhofman/probly)](https://github.com/pwhofman/probly/commits/main)
</div>

<div align="center">
<em>Turn any model into one that knows what it doesn’t know.</em>
</div>

`probly` is a **library-agnostic** toolkit for **uncertainty representation and
quantification** in machine learning. Make any PyTorch, Flax/JAX, scikit-learn, River, or
Hugging Face model uncertainty-aware in a single line, then **represent**, **quantify**,
and **decompose** its predictive uncertainty into **aleatoric** and **epistemic** components.
It ships 40+ methods ranging from Bayesian nets and deep ensembles to evidential, credal, and
conformal prediction, all accessible via the same unified API.


## 🛠️ Install
`probly` is intended to work with **Python 3.13 and above**. Installation can be done via `pip`
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
from probly.method import dropout
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
print(evaluate_ood(eu_id, eu_ood))
```

Output:

```text
{'auroc': 0.94}
```

Swap `dropout` for `ensemble`, `bayesian`, `laplace`, or any other method listed [below](#-methods-at-a-glance) while the rest of the pipeline stays the same. Comparing methods is then a matter of changing one line, which allows results such as the ImageNet numbers portrayed in the figures below: one pipeline, 20+ methods.

<div align="center">
  <picture>
    <source srcset="docs/source/_static/readme/from_paper/paper_benchmark_dark.png" media="(prefers-color-scheme: dark)">
    <source srcset="docs/source/_static/readme/from_paper/paper_benchmark_light.png" media="(prefers-color-scheme: light)">
    <img src="docs/source/_static/readme/from_paper/paper_benchmark_light.png" alt="Bar charts of out-of-distribution detection AUROC for 20+ uncertainty methods on ImageNet with a ResNet50, far-OoD on the left and near-OoD on the right" width="100%" />
  </picture>
  <br />
  <em>Out-of-distribution detection on ImageNet, far and near, mean over three runs.</em>
</div>

<div align="center">
  <picture>
    <source srcset="docs/source/_static/readme/from_paper/paper_ood_histogram_dark.png" media="(prefers-color-scheme: dark)">
    <source srcset="docs/source/_static/readme/from_paper/paper_ood_histogram_light.png" media="(prefers-color-scheme: light)">
    <img src="docs/source/_static/readme/from_paper/paper_ood_histogram_light.png" alt="Histogram of the uncertainty scores a credal Bayesian neural network assigns to in-distribution ImageNet images and to out-of-distribution iNaturalist images, with the ImageNet scores piling up near zero" width="55%" />
  </picture>
  <br />
  <em>The same result up close: a credal BNN scores in-distribution ImageNet near zero and out-of-distribution iNaturalist much higher.<br />See the <a href="https://pwhofman.github.io/probly">docs</a> for the full benchmark.</em>
</div>

## 🤔 What "uncertainty" means here

In a two-class scenario between "dog" and "cat," assigning 0.51 probability to "dog" only marginally prefers "dog" to "cat," barely counting as a confident decision.
In such a case, the classifier can be unsure for two very different reasons;
distinguishing between them changes what to do next.

**Aleatoric uncertainty** is noise in the data inherent to the data-generating process;
be it because of blurry images, classes genuinely overlapping, or an imprecise sensor.
More data won’t reduce the aleatoric uncertainty, as the ambiguity lies in the problem itself,
not in the model’s ignorance of it.

**Epistemic uncertainty** reflects missing knowledge. The model hasn’t encountered similar inputs
during training, so it’s operating outside its experience. Unlike aleatoric uncertainty,
this can be reduced with more data or a better model.

This distinction is especially useful in practice. High epistemic uncertainty signals where you can
take action either by gathering more data, expanding the model capacity, or sending the case to a human.
In contrast, high aleatoric uncertainty reflects irreducible noise in the problem itself;
collecting more samples won’t help. A single confidence score cannot distinguish between these situations, which is why
`probly` addresses this in its four-stage process.

<div align="center">
  <picture>
    <source srcset="docs/source/_static/readme/from_paper/paper_workflow_dark.png" media="(prefers-color-scheme: dark)">
    <source srcset="docs/source/_static/readme/from_paper/paper_workflow_light.png" media="(prefers-color-scheme: light)">
    <img src="docs/source/_static/readme/from_paper/paper_workflow_light.png" alt="The four-stage probly workflow: transform a model to make it uncertainty-aware, represent its predictive uncertainty, quantify and decompose it into total, epistemic and aleatoric parts, decide based on it, and evaluate the result in a downstream task" width="100%" />
  </picture>
  <br />
  <em><strong>transform</strong> a model to carry uncertainty, <strong>represent</strong> its predictions, <strong>quantify</strong> and <strong>decide</strong>, then <strong>evaluate</strong> on a task.</em>
</div>

Each stage is one import, and the stages compose freely. The
[user guide](https://pwhofman.github.io/probly/stable/user_guide.html) walks through them in
order.

## 🎲 Methods at a glance

Every method below integrates with a **single line of code** regardless of the type of model: Linear, CNN, GNN, or LLM. You can apply them **post-hoc** by wrapping an existing model to make it uncertainty-aware, or **ante-hoc** by building an uncertainty-native model from scratch.

What differs between each method is how it represents uncertainty. `probly` covers the range from simple point predictions to full distributional representations. The methods below are organized by representation type, falling within this range.

<div align="center">
  <picture>
    <source srcset="docs/source/_static/readme/from_paper/paper_representations_dark.png" media="(prefers-color-scheme: dark)">
    <source srcset="docs/source/_static/readme/from_paper/paper_representations_light.png" media="(prefers-color-scheme: light)">
    <img src="docs/source/_static/readme/from_paper/paper_representations_light.png" alt="Uncertainty representations arranged from zeroth to second order, shown for classification on a probability simplex over dog, fox and cat, and for regression in the mean-standard-deviation plane: single outcome, set of outcomes, probability distribution, samples of a distribution over distributions, a distribution over distributions, and a set of distributions" width="100%" />
  </picture>
  <br />
  <em>Range of uncertainty representations: single outcome to set of probability distributions. All methods implemented in probly fall along this range.</em>
</div>

#### 🧠 Second-order distributions

These methods transform a point predictor into a model that outputs a distribution over possible distributions. Instead of returning a single prediction, they learn a higher-order probabilistic model where predictions are expressed as probability distributions. This can be achieved by stochastic sampling, ensembling, estimating feature-space distance to training data, or by parameterizing it with an evidential output head.

<details>
<summary><strong>Show all 21 methods</strong></summary>

<a id="second-order-table"></a>

| Method | Reference | Backends |
| :--- | :--- | :--- |
| MC dropout (`dropout`) | [Gal & Ghahramani, 2016](http://proceedings.mlr.press/v48/gal16.html) | torch · flax |
| MC dropconnect (`dropconnect`) | [Mobiny et al., 2021](https://doi.org/10.1038/s41598-021-84854-x) | torch · flax |
| Mean-field Bayesian networks (`bayesian`) | [Blundell et al., 2015](http://proceedings.mlr.press/v37/blundell15.html) | torch |
| Laplace approximation (`laplace`) | [Daxberger et al., 2021](https://arxiv.org/abs/2106.14806) | torch |
| Deep ensembles (`ensemble`) | [Lakshminarayanan et al., 2017](https://proceedings.neurips.cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html) | torch · flax · sklearn · river |
| BatchEnsemble (`batchensemble`) | [Wen et al., 2020](https://openreview.net/forum?id=Sklf1yrYDr) | torch · flax |
| Sub-ensembles (`subensemble`) | [Valdenegro-Toro, 2019](https://arxiv.org/abs/1910.08168) | torch · flax |
| Deep anti-regularized ensembles (`dare`)[¹](#fn-so-1) | [de Mathelin et al., 2023](https://doi.org/10.48550/arXiv.2304.04042) | torch |
| Deterministic uncertainty quantification (`duq`) | [van Amersfoort et al., 2020](http://proceedings.mlr.press/v119/van-amersfoort20a.html) | torch |
| Deep deterministic uncertainty (`ddu`) | [Mukhoti et al., 2023](https://doi.org/10.1109/CVPR52729.2023.02336) | torch |
| Mahalanobis distance (`mahalanobis`) | [Lee et al., 2018](https://proceedings.neurips.cc/paper/2018/hash/abdeb6f575ac5c6676b747bca8d09cc2-Abstract.html) | torch |
| Direct epistemic uncertainty prediction (`deup`) | [Lahlou et al., 2023](https://openreview.net/forum?id=eGLdVRvvfQ) | torch |
| Spectral-normalized GP heads (`sngp`) | [Liu et al., 2020](https://proceedings.neurips.cc/paper/2020/hash/543e83748234f7cbab21aa0ade66565f-Abstract.html) | torch |
| Evidential classification (`evidential_classification`) | [Sensoy et al., 2018](https://proceedings.neurips.cc/paper/2018/hash/a981f2b708044d6fb4a71a1463242520-Abstract.html) | torch |
| Posterior networks (`posterior_network`) | [Charpentier et al., 2020](https://proceedings.neurips.cc/paper/2020/hash/0eac690d7059a8de4b48e90f14510391-Abstract.html) | torch |
| Natural posterior networks (`natural_posterior_network`) | [Charpentier et al., 2022](https://openreview.net/forum?id=tV3N0DWMxCg) | torch |
| Graph posterior networks (`graph_posterior_network`, `cuq_graph_neural_network`, `lop_graph_posterior_network`) | [Stadler et al., 2021](https://arxiv.org/abs/2110.14012) | torch |
| Prior networks (`prior_network`) | [Malinin & Gales, 2018](https://proceedings.neurips.cc/paper/2018/hash/3ea2db50e62ceefceaf70a9d9a56a6f4-Abstract.html) | torch |
| Evidential regression (`evidential_regression`) | [Amini et al., 2020](https://proceedings.neurips.cc/paper/2020/hash/aab085461de182608ee9f607f3f7d18f-Abstract.html) | torch |
| Heteroscedastic networks (`het_net`) | [Collier et al., 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Collier_Correlated_Input-Dependent_Label_Noise_in_Large-Scale_Image_Classification_CVPR_2021_paper.html) | torch |
| Dirichlet activations & NIG heads (`dirichlet_*`, `normal_inverse_gamma_head`) | [Malinin et al., 2020](https://arxiv.org/abs/2006.11590) | torch |

<a id="fn-so-1"></a>¹ Built on the `ensemble` transformation and is applicable in the same context as `ensemble`. [↩](#second-order-table)

</details>

#### ☁️ Credal sets

Instead of committing to a single second-order distribution, credal methods represent uncertainty as a set of plausible probability distributions. Expressing results as lower/upper probability bounds rather than point probabilities allows capturing ignorance when a model does not have a conclusive answer.

<details>
<summary><strong>Show all 12 methods</strong></summary>

<a id="credal-table"></a>

| Method                                                                                          | Reference | Backends |
|:------------------------------------------------------------------------------------------------| :--- | :--- |
| Credal wrapper (`credal_wrapper`)[¹](#fn-credal-1)                                              | [Wang et al., 2025](https://openreview.net/forum?id=cv2iMNWCsh) | torch |
| Credal ensembling (`credal_ensembling`)[¹](#fn-credal-1)                                        | [Nguyen et al., 2025](https://doi.org/10.1007/s10994-024-06703-y) | torch |
| Credal Bayesian deep learning (`credal_bnn`)[¹](#fn-credal-1)                                   | [Caprio et al., 2024](https://openreview.net/forum?id=4NHF9AC5ui) | torch |
| Credal nets (`credal_net`)[¹](#fn-credal-1)                                                     | [Sale et al., 2024](https://openreview.net/forum?id=VJjjNrUi8j) | torch |
| Relative-likelihood credal prediction (`credal_relative_likelihood`)[¹](#fn-credal-1)           | [Löhr et al., 2025](https://doi.org/10.48550/arXiv.2505.22332) | torch |
| Class-bias ensembles (`class_bias_ensemble`)[²](#fn-credal-2)                                   | [Löhr et al., 2025](https://doi.org/10.48550/arXiv.2505.22332) | torch |
| Efficient credal prediction (`efficient_credal_prediction`)[³](#fn-credal-3)                    | [Hofman et al., 2026](https://doi.org/10.48550/arXiv.2603.08495) | torch |
| Conformal credal set, inner product (`conformal_inner_product`)                                 | [Sale et al., 2024](https://openreview.net/forum?id=VJjjNrUi8j) | torch |
| Conformal credal set, Kullback-Leibler (`conformal_kullback_leibler`)                           | [Sale et al., 2024](https://openreview.net/forum?id=VJjjNrUi8j) | torch |
| Conformal credal set, total variation (`conformal_total_variation`)                             | [Sale et al., 2024](https://openreview.net/forum?id=VJjjNrUi8j) | torch |
| Conformal credal set, Wasserstein (`conformal_wasserstein_distance`)                            | [Sale et al., 2024](https://openreview.net/forum?id=VJjjNrUi8j) | torch |
| Conformal credal set, Dirichlet relative likelihood (`conformal_dirichlet_relative_likelihood`) | [Sale et al., 2024](https://openreview.net/forum?id=VJjjNrUi8j) | torch |

<a id="fn-credal-1"></a>¹ Built on the `ensemble` transformation and is applicable in the same context as `ensemble`. [↩](#credal-table)<br>
<a id="fn-credal-2"></a>² The ensembling basis for `credal_relative_likelihood`. [↩](#credal-table)<br>
<a id="fn-credal-3"></a>³ Also has a pure NumPy implementation. [↩](#credal-table)

</details>

#### 📏 Conformal prediction

Distribution-free prediction sets/intervals that provide finite-sample coverage guarantees. Those sets/intervals are guaranteed to contain the true value.

<details>
<summary><strong>Show all 8 methods</strong></summary>

| Method | Reference | Backends |
| :--- | :--- | :--- |
| LAC (`conformal_lac`) | [Sadinle et al., 2019](https://arxiv.org/abs/1609.00451) | torch · flax · sklearn |
| APS (`conformal_aps`) | [Romano et al., 2020](https://arxiv.org/abs/2006.02544) | torch · flax · sklearn |
| SAPS (`conformal_saps`) | [Huang et al., 2024](https://arxiv.org/abs/2310.06430) | torch · flax · sklearn |
| RAPS (`conformal_raps`) | [Angelopoulos et al., 2021](https://arxiv.org/abs/2009.14193) | torch · flax · sklearn |
| Absolute-error CP (`conformal_absolute_error`) | [Angelopoulos & Bates, 2021](https://arxiv.org/abs/2107.07511) | torch · flax · sklearn |
| CQR (`conformal_cqr`) | [Romano et al., 2019](https://arxiv.org/abs/1905.03222) | torch · flax · sklearn |
| CQR-r (`conformal_cqr_r`) | [Sesia & Candès, 2020](https://arxiv.org/abs/1909.05433) | torch · flax · sklearn |
| UACQR (`conformal_uacqr`) | [Rossellini et al., 2024](https://arxiv.org/abs/2306.08693) | torch · flax · sklearn |

</details>

#### 🌡️ Calibration

Post-hoc methods that ensure that a model’s predicted probabilities match the statistical likelihood, fixing over-confident probabilities. The methods are fitted on a held-out split, a separate validation dataset.

<details>
<summary><strong>Show all 4 methods</strong></summary>

| Method | Reference | Backends |
| :--- | :--- | :--- |
| Temperature scaling (`temperature_scaling`) | [Guo et al., 2017](http://proceedings.mlr.press/v70/guo17a.html) | torch · sklearn |
| Platt scaling (`platt_scaling`) | [Platt, 1999](https://www.semanticscholar.org/paper/Probabilistic-Outputs-for-Support-vector-Machines-Platt/42e5ed832d4310ce4378c44d05570439df28a393) | torch · sklearn |
| Vector scaling (`vector_scaling`) | [Guo et al., 2017](http://proceedings.mlr.press/v70/guo17a.html) | torch · sklearn |
| Isotonic regression (`isotonic_regression`) | [Zadrozny & Elkan, 2002](https://doi.org/10.1145/775047.775151) | torch · sklearn |

</details>

Calibration-aware training losses ship too: label smoothing, [label relaxation](https://doi.org/10.1609/aaai.v35i10.17041), and [focal loss](https://doi.org/10.1109/ICCV.2017.324) in `probly.train.calibration.torch`.

#### 📐 Uncertainty quantification

The preceding methods produce a **representation**. To interpret these, `quantify` converts them to numbers, and `decompose` splits the value into its **aleatoric** and **epistemic** parts, wherever possible. The measures below are organized by the type of representation they process.

- **Distributions**: `entropy`, `mutual_information`, `conditional_entropy`, `sample_variance`, `vacuity`, `dempster_shafer_uncertainty`
- **Credal sets**: `upper_entropy`, `lower_entropy`, `generalized_hartley`, `min_expected_total_variation`
- **Conformal sets**: `conformal_set_size`
- **Embeddings and text**: `spectral_entropy`, `semantic_entropy`
- **Scoring rules**: `BrierLoss`, `LogLoss`, `SphericalLoss`, `ZeroOneLoss`

🤖 **LLM uncertainty**, too: semantic entropy and spectral uncertainty for Hugging Face text generation models. [See below](#-uncertainty-for-llms).

Browse the full [API reference](https://pwhofman.github.io/probly/stable/api.html) and the [examples gallery](https://pwhofman.github.io/probly/stable/examples.html) for the complete picture.

## 🤖 Uncertainty for LLMs

The same machinery applies to text generation. Sample several answers to a question, cluster them by meaning with an NLI model, and decompose the resulting **semantic entropy** ([Kuhn et al., 2023](https://arxiv.org/abs/2302.09664)) into its aleatoric and epistemic parts. A model that answers the same thing in five different phrasings is confident; one that answers five different things is not.

<div align="center">
  <picture>
    <source srcset="docs/source/_static/readme/llm_uncertainty_demo_dark.svg" media="(prefers-color-scheme: dark)">
    <source srcset="docs/source/_static/readme/llm_uncertainty_demo_light.svg" media="(prefers-color-scheme: light)">
    <img src="docs/source/_static/readme/llm_uncertainty_demo_light.svg" alt="Animated demo: a factual question collapses to one meaning with zero uncertainty, while a trick question scatters into seven meanings with high uncertainty, flagging a likely hallucination" width="100%" />
  </picture>
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

The factual question collapses into a single semantic cluster, so the entropy is near zero. The trick question scatters across many clusters, which flags a likely hallucination. See [`examples/llm/semantic_entropy.py`](examples/llm/semantic_entropy.py) for the full pipeline and [`examples/llm/spectral_uncertainty.py`](examples/llm/spectral_uncertainty.py) for an embedding-based alternative.

## 📈 Uncertainty for regression

For regression, epistemic uncertainty shows up as a band around the prediction. Wrap any model in a **deep ensemble** ([Lakshminarayanan et al., 2017](https://proceedings.neurips.cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html)) and `probly` turns the disagreement between members into that band: tight near the training data, wide wherever the model extrapolates.

<div align="center">
  <picture>
    <source srcset="docs/source/_static/readme/regression_uncertainty_dark.svg" media="(prefers-color-scheme: dark)">
    <source srcset="docs/source/_static/readme/regression_uncertainty_light.svg" media="(prefers-color-scheme: light)">
    <img src="docs/source/_static/readme/regression_uncertainty_light.svg" alt="Deep-ensemble regression on a 1D sine dataset with a gap in the middle: the predicted mean tracks the data while the uncertainty band stays tight over the two training regions and balloons in the central gap and beyond the data range" width="100%" />
  </picture>
</div>

```python
import numpy as np
import torch
import torch.nn as nn
from probly.method import ensemble
from probly.representer import representer
from probly.quantification.decomposition.variance import SecondOrderVarianceDecomposition

# 1D data with a gap in the middle, so the model never sees x near 0
X = np.concatenate([np.random.uniform(-4.5, -1, 40), np.random.uniform(1, 4.5, 40)])
X = torch.from_numpy(X).float().reshape(-1, 1)
y = torch.sin(1.5 * X) + 0.12 * torch.randn_like(X)

net = nn.Sequential(nn.Linear(1, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))

# transform: build a deep ensemble, reinitializing each member for diversity
ens = ensemble(net, num_members=10, reset_params=True)

for member in ens:  # train each member as usual
    opt = torch.optim.Adam(member.parameters(), lr=0.01)
    for _ in range(400):
        opt.zero_grad()
        loss = nn.functional.mse_loss(member(X), y)
        loss.backward()
        opt.step()

# represent + quantify: mean prediction and its uncertainty over a wide grid
grid = torch.linspace(-7, 7, 400).reshape(-1, 1)
out = representer(ens).predict(grid)
unc = SecondOrderVarianceDecomposition(out)

mean = out.tensor.mean(dim=out.sample_axis)  # ensemble mean
std = unc.epistemic.sqrt()                   # wide where members disagree, i.e. away from data
# plot mean ± 2 * std to reproduce the band above
```

The band is tight where the model has seen data and wide in the gap and past the edges, which is exactly where an ensemble should be unsure. `SecondOrderVarianceDecomposition` also exposes `.total` and `.aleatoric`, so you can separate disagreement between members (epistemic) from irreducible noise (aleatoric). See [`examples/quantification/plot_ensemble_regression.py`](examples/quantification/plot_ensemble_regression.py) for the full walkthrough.

## 📖 Documentation with tutorials

New to `probly`? Start with the [user guide](https://pwhofman.github.io/probly/stable/user_guide.html) and the [examples gallery](https://pwhofman.github.io/probly/stable/examples.html). The full [documentation](https://pwhofman.github.io/probly) covers the entire workflow, including the [API reference](https://pwhofman.github.io/probly/stable/api.html).

## 🤝 Contributing
Contributions are welcome - see [CONTRIBUTING.md](.github/CONTRIBUTING.md) for guidelines on adding methods, representations, or evaluation protocols.

## 📜 License
This project is licensed under the [MIT License](https://github.com/pwhofman/probly/blob/main/LICENSE).

---
Built with ❤️ by the probly team.
