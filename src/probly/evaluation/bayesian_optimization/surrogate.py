"""Surrogate models for Bayesian optimization.

Each surrogate is a probly :class:`~probly.predictor.RepresentationPredictor`
returning whichever :class:`~probly.representation.Representation` is
natural to its underlying model:

* :class:`BotorchGPSurrogate` -- exact GP via botorch ``SingleTaskGP``;
  returns a :class:`~probly.representation.distribution.TorchGaussianDistribution`.
* :class:`RandomForestSurrogate` -- sklearn ``RandomForestRegressor``;
  returns a :class:`~probly.representation.sample.TorchSample` of per-tree
  predictions (the empirical posterior over trees).
* :class:`MCDropoutSurrogate` -- a torch MLP trained with MC-Dropout via
  probly's :func:`~probly.transformation.dropout.dropout` transformation;
  returns a :class:`~probly.representation.sample.TorchSample` of
  ``num_samples`` Monte-Carlo forward passes via probly's
  :func:`~probly.representer.representer`.
* :class:`BNNSurrogate` -- a Bayes-by-Backprop BNN via probly's
  :func:`~probly.transformation.bayesian.bayesian` transformation, trained
  with the standard ELBO (MSE + KL); returns a ``TorchSample`` of
  ``num_samples`` weight samples through the same Sampler machinery.

Acquisitions consume the posterior via :func:`posterior_mean_std`, a
flexdispatch that knows how to extract ``(mean, std)`` from each
representation type -- ``Sample`` uses ``sample_mean`` / ``sample_var``,
``GaussianDistribution`` reads ``.mean`` / ``.var``. This keeps the
non-GP surrogates honest about their empirical posteriors instead of
forcing every model into a Gaussian shape, while still letting the rest
of probly's UQ machinery (``quantify``, ``decompose``) consume the
representation directly. Adding another probly stochastic method
(e.g. ``dropconnect``, ``batchensemble``) is a one-line change inside
``fit``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
import torch
from torch import nn

from flextype import flexdispatch
from probly.predictor import RepresentationPredictor
from probly.representation import Representation
from probly.representation.distribution import GaussianDistribution
from probly.representation.distribution.torch_gaussian import TorchGaussianDistribution
from probly.representation.sample import Sample
from probly.representation.sample.torch import TorchSample
from probly.representer import Representer, representer
from probly.train.bayesian.torch import collect_kl_divergence
from probly.transformation.bayesian import bayesian
from probly.transformation.dropout import dropout

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sklearn.ensemble import RandomForestRegressor

_VAR_FLOOR: float = 1e-6
"""Lower bound applied to the variance of surrogate posteriors."""


@flexdispatch
def posterior_mean_std(representation: Representation) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(mean, std)`` tensors from a surrogate's posterior representation.

    Acquisitions and visualizations call this to consume any surrogate's
    posterior uniformly. Register a new representation type via
    ``@posterior_mean_std.register(...)`` to support new surrogates.
    """
    msg = f"No posterior_mean_std extractor registered for {type(representation)}"
    raise NotImplementedError(msg)


@posterior_mean_std.register(GaussianDistribution)
def _(distribution: GaussianDistribution) -> tuple[torch.Tensor, torch.Tensor]:
    return distribution.mean, distribution.var.clamp_min(_VAR_FLOOR).sqrt()


@posterior_mean_std.register(Sample)
def _(sample: Sample) -> tuple[torch.Tensor, torch.Tensor]:
    return sample.sample_mean(), sample.sample_var().clamp_min(_VAR_FLOOR).sqrt()


class Surrogate(
    RepresentationPredictor[[torch.Tensor], Representation],
    ABC,
):
    """Abstract base class for BO surrogates.

    A surrogate is a probly representation predictor returning whichever
    :class:`~probly.representation.Representation` is natural to the
    underlying model -- a :class:`~probly.representation.distribution.GaussianDistribution`
    for the GP, a :class:`~probly.representation.sample.Sample` for any
    ensemble or stochastic-NN surrogate. The acquisition pulls
    ``(mean, std)`` out via :func:`posterior_mean_std`.

    ``fit`` is called on the augmented observation pool between BO rounds;
    ``predict_representation`` is what ``predict(surrogate, x)`` ultimately
    dispatches to.

    Attributes:
        differentiable: Whether the posterior is differentiable through
            autograd. Acquisitions consult this to choose between gradient-
            based local optimization and a Sobol-only sweep.
    """

    differentiable: ClassVar[bool] = True

    @abstractmethod
    def fit(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Fit the surrogate on observed inputs and outputs.

        Args:
            x: Observed inputs of shape ``(n, dim)``.
            y: Observed outputs of shape ``(n,)`` or ``(n, 1)``.
        """

    @abstractmethod
    def predict_representation(self, x: torch.Tensor) -> Representation:
        """Return the posterior representation at query inputs.

        Args:
            x: Query inputs of shape ``(n, dim)``.

        Returns:
            A probly :class:`Representation` (a Gaussian distribution for
            the GP, a Sample for any ensemble / MC surrogate). Use
            :func:`posterior_mean_std` to extract ``(mean, std)`` tensors
            uniformly.
        """


class BotorchGPSurrogate(Surrogate):
    """Exact GP surrogate using botorch ``SingleTaskGP``.

    Wraps the GP with a :class:`~botorch.models.transforms.input.Normalize`
    input transform and a :class:`~botorch.models.transforms.outcome.Standardize`
    outcome transform so the GP fits roughly unit-scale data regardless of
    the objective.
    """

    differentiable: ClassVar[bool] = True

    def __init__(self) -> None:
        """Construct an unfitted GP surrogate."""
        self._model: SingleTaskGP | None = None

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Refit the GP on the full set of observations.

        Args:
            x: Observed inputs of shape ``(n, dim)``.
            y: Observed outputs of shape ``(n,)`` or ``(n, 1)``.
        """
        x_d = x.to(torch.float64)
        y_d = y.to(torch.float64).view(-1, 1)
        d = x_d.shape[-1]
        self._model = SingleTaskGP(
            train_X=x_d,
            train_Y=y_d,
            input_transform=Normalize(d=d),
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(self._model.likelihood, self._model)
        fit_gpytorch_mll(mll)
        self._model.eval()

    def predict_representation(self, x: torch.Tensor) -> TorchGaussianDistribution:
        """Return the GP posterior at ``x`` as a Gaussian distribution.

        Gradients flow through the result so the caller can run an autograd-
        based acquisition optimizer (e.g. L-BFGS-B with analytic gradients).
        """
        if self._model is None:
            msg = "BotorchGPSurrogate must be fit before calling predict_representation."
            raise RuntimeError(msg)
        x_d = x.to(torch.float64)
        posterior = self._model.posterior(x_d)
        mean = posterior.mean.squeeze(-1).to(x.dtype)
        var = posterior.variance.clamp_min(_VAR_FLOOR).squeeze(-1).to(x.dtype)
        return TorchGaussianDistribution(mean=mean, var=var)


class RandomForestSurrogate(Surrogate):
    """Random-forest surrogate using sklearn ``RandomForestRegressor``.

    The posterior at a query input is a :class:`~probly.representation.sample.TorchSample`
    of per-tree predictions -- the empirical posterior over the forest.
    The acquisition reads its mean and standard deviation via
    ``sample.sample_mean()`` / ``sample.sample_var().sqrt()``, the same
    way it reads them from the BNN / MC-Dropout sample posteriors. This
    is the canonical SMAC/MOE-style surrogate and a robust fallback when
    GP marginal-likelihood optimization struggles (e.g. very rough or
    high-dimensional objectives). probly does not have a tree-based UQ
    transformation, so this surrogate wraps sklearn directly rather than
    going through ``representer`` / ``Sampler`` -- but the *output* is
    still a probly ``Sample``, so it composes with ``quantify`` /
    ``decompose`` like any other ensemble-style posterior.

    The posterior is **not** differentiable in ``x`` (decision trees are
    piecewise constant), so acquisitions optimize via Sobol sweeps instead
    of L-BFGS-B.

    Attributes:
        n_estimators: Number of trees in the forest.
        max_depth: Optional cap on tree depth.
        min_samples_leaf: Minimum number of training points per leaf.
        seed: RNG seed for forest construction; bootstrap and feature
            sampling use this.
    """

    differentiable: ClassVar[bool] = False

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        seed: int = 0,
    ) -> None:
        """Construct an unfitted random-forest surrogate.

        Args:
            n_estimators: Number of trees in the forest.
            max_depth: Optional cap on tree depth.
            min_samples_leaf: Minimum number of training points per leaf.
            seed: RNG seed for forest construction.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.seed = seed
        self._rf: RandomForestRegressor | None = None

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Refit the random forest on the full set of observations."""
        from sklearn.ensemble import RandomForestRegressor  # noqa: PLC0415

        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().view(-1).numpy()
        rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.seed,
            bootstrap=True,
        )
        rf.fit(x_np, y_np)
        self._rf = rf

    def predict_representation(self, x: torch.Tensor) -> TorchSample[torch.Tensor]:
        """Return the forest posterior at ``x`` as a per-tree :class:`TorchSample`.

        The returned sample has shape ``(n_trees, n_query)`` with
        ``sample_dim=0``; ``sample_mean()`` and ``sample_var()`` give the
        cross-tree statistics. Lives on the same device and dtype as
        ``x``; gradients are not propagated.
        """
        rf = self._rf
        if rf is None:
            msg = "RandomForestSurrogate must be fit before calling predict_representation."
            raise RuntimeError(msg)
        import numpy as np  # noqa: PLC0415

        x_np = x.detach().cpu().numpy()
        per_tree = np.stack([t.predict(x_np) for t in rf.estimators_], axis=0)
        tensor = torch.from_numpy(per_tree).to(device=x.device, dtype=x.dtype)
        return TorchSample(tensor=tensor, sample_dim=0)


# ---------------------------------------------------------------------------
# NN-based surrogates: train a torch MLP, wrap with a probly stochastic
# transformation, then sample with probly's Sampler representer.
# ---------------------------------------------------------------------------


class _MLP(nn.Module):
    """A small feedforward regressor with no built-in dropout.

    Stochastic behavior is added externally by a probly transformation
    (``dropout()`` for MC-Dropout, ``bayesian()`` for Bayes-by-Backprop)
    rather than baked into this base architecture. Keeping the base bare
    ensures the stochastic transformation owns the noise mechanism end to
    end.
    """

    def __init__(self, in_dim: int, hidden_dims: Sequence[int]) -> None:
        """Build an MLP with the given hidden widths and a scalar output."""
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.GELU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the scalar prediction at ``x``."""
        return self.net(x).squeeze(-1)


def _standardize(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(x_n, y_n, x_mean, x_std, y_mean, y_std)`` for unit-scale fitting."""
    x_f = x.to(torch.float32).detach()
    y_f = y.to(torch.float32).detach().view(-1)
    x_mean = x_f.mean(dim=0)
    x_std = x_f.std(dim=0).clamp_min(1e-6)
    y_mean = y_f.mean()
    y_std = y_f.std().clamp_min(1e-6)
    return (x_f - x_mean) / x_std, (y_f - y_mean) / y_std, x_mean, x_std, y_mean, y_std


class MCDropoutSurrogate(Surrogate):
    """MC-Dropout surrogate built on probly's ``dropout`` transformation.

    Mirrors how the active-learning module uses MC-Dropout
    (``configs/method/dropout.yaml``): a base torch MLP, wrapped by
    :func:`~probly.transformation.dropout.dropout`, then sampled via
    probly's :func:`~probly.representer.representer` with ``num_samples``
    Monte-Carlo forward passes. The posterior at a query input is the
    cross-sample mean and variance, exactly as
    :class:`~probly.representer.IterableSampler` would produce for
    classification UQ -- only the final conversion to a Gaussian is
    BO-specific.

    The posterior is differentiable through autograd (each forward draws a
    fresh dropout mask but the resulting tensor still carries grad), so
    acquisitions can use gradient-based local optimization.

    Attributes:
        hidden_dims: Hidden layer widths of the base MLP.
        dropout_p: Dropout probability used in training and inference.
        lr: Adam learning rate.
        epochs: Full-batch gradient steps per fit.
        weight_decay: L2 regularization strength.
        num_samples: MC samples drawn per ``predict_representation`` call.
        seed: Base RNG seed for parameter initialization.
    """

    differentiable: ClassVar[bool] = True

    def __init__(
        self,
        hidden_dims: Sequence[int] = (64, 64),
        dropout_p: float = 0.1,
        lr: float = 1e-2,
        epochs: int = 200,
        weight_decay: float = 1e-4,
        num_samples: int = 20,
        seed: int = 0,
    ) -> None:
        """Construct an unfitted MC-Dropout surrogate.

        Args:
            hidden_dims: Hidden layer widths for the MLP.
            dropout_p: Dropout probability used in training and inference.
            lr: Adam learning rate.
            epochs: Full-batch gradient steps per fit.
            weight_decay: L2 regularization strength.
            num_samples: MC samples drawn per posterior query.
            seed: Base RNG seed for parameter initialization.
        """
        if not 0.0 < dropout_p < 1.0:
            msg = f"dropout_p must be in (0, 1), got {dropout_p}."
            raise ValueError(msg)
        self.hidden_dims = tuple(hidden_dims)
        self.dropout_p = dropout_p
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.num_samples = num_samples
        self.seed = seed

        self._sampler: Representer | None = None
        self._x_mean: torch.Tensor | None = None
        self._x_std: torch.Tensor | None = None
        self._y_mean: torch.Tensor | None = None
        self._y_std: torch.Tensor | None = None

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Train the MLP under MC-Dropout regularization on the current observations.

        The base MLP is wrapped by probly's ``dropout()`` transformation
        (which prepends ``nn.Dropout`` before each non-input ``nn.Linear``)
        before training, so dropout regularizes the MSE optimization. After
        training, probly's :func:`~probly.representer.representer` produces
        a :class:`~probly.representer.Sampler` that draws ``num_samples``
        stochastic forward passes per query.
        """
        x_n, y_n, x_mean, x_std, y_mean, y_std = _standardize(x, y)

        torch.manual_seed(self.seed)
        base = _MLP(in_dim=x_n.shape[-1], hidden_dims=self.hidden_dims)
        mc_model = dropout(base, p=self.dropout_p)

        mc_model.train()
        optim = torch.optim.Adam(mc_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for _ in range(self.epochs):
            optim.zero_grad()
            pred = mc_model(x_n)
            loss = nn.functional.mse_loss(pred, y_n)
            loss.backward()
            optim.step()

        mc_model.eval()
        # ``representer`` dispatches on ``DropoutPredictor`` (a RandomPredictor)
        # and returns a Sampler that forces dropout layers into train mode at
        # sample time -- the exact wiring AL uses for MC-Dropout UQ.
        self._sampler = representer(mc_model, num_samples=self.num_samples)
        self._x_mean = x_mean
        self._x_std = x_std
        self._y_mean = y_mean
        self._y_std = y_std

    def predict_representation(self, x: torch.Tensor) -> TorchSample[torch.Tensor]:
        """Return the MC-Dropout posterior at ``x`` as a :class:`TorchSample`."""
        if (
            self._sampler is None
            or self._x_mean is None
            or self._x_std is None
            or self._y_mean is None
            or self._y_std is None
        ):
            msg = "MCDropoutSurrogate must be fit before calling predict_representation."
            raise RuntimeError(msg)
        x_n = (x.to(torch.float32) - self._x_mean) / self._x_std
        sample = self._sampler.represent(x_n)
        # De-normalize the empirical samples back to the target scale and
        # cast to the caller's dtype; the Sample shape (sample_dim, weights)
        # is preserved so downstream `sample_mean` / `sample_var` work.
        denormalized = sample.tensor * self._y_std + self._y_mean
        return TorchSample(
            tensor=denormalized.to(x.dtype),
            sample_dim=sample.sample_dim,
            weights=sample.weights,
        )


class BNNSurrogate(Surrogate):
    """Bayes-by-Backprop BNN surrogate built on probly's ``bayesian`` transformation.

    Mirrors how the active-learning module uses BNNs
    (``configs/method/bayesian.yaml``): a base torch MLP, wrapped by
    :func:`~probly.transformation.bayesian.bayesian` to replace its
    ``nn.Linear`` layers with mean-field variational ``BayesLinear``
    layers, trained with the standard ELBO (MSE on regression targets +
    ``kl_penalty`` * sum of per-layer KL divergences against the Gaussian
    prior). At inference, probly's :func:`~probly.representer.representer`
    returns a Sampler that draws ``num_samples`` weight samples per query;
    the cross-sample mean / variance form the Gaussian posterior.

    The posterior is differentiable through autograd, so acquisitions can
    use gradient-based local optimization.

    Attributes:
        hidden_dims: Hidden layer widths of the base MLP.
        posterior_std: Initial posterior standard deviation of the
            variational weights (passed to ``bayesian()``).
        prior_std: Standard deviation of the Gaussian weight prior.
        kl_penalty: Weight on the KL term in the ELBO.
        lr: Adam learning rate.
        epochs: Full-batch gradient steps per fit.
        num_samples: MC samples drawn per ``predict_representation`` call.
        seed: Base RNG seed for parameter initialization.
    """

    differentiable: ClassVar[bool] = True

    def __init__(
        self,
        hidden_dims: Sequence[int] = (64, 64),
        posterior_std: float = 0.05,
        prior_std: float = 1.0,
        kl_penalty: float = 1e-3,
        lr: float = 1e-2,
        epochs: int = 300,
        num_samples: int = 20,
        seed: int = 0,
    ) -> None:
        """Construct an unfitted BNN surrogate.

        Args:
            hidden_dims: Hidden layer widths for the MLP.
            posterior_std: Initial posterior standard deviation for
                ``BayesLinear`` weights.
            prior_std: Standard deviation of the Gaussian weight prior.
            kl_penalty: Weight on the KL term in the ELBO loss.
            lr: Adam learning rate.
            epochs: Full-batch gradient steps per fit.
            num_samples: MC samples drawn per posterior query.
            seed: Base RNG seed for parameter initialization.
        """
        if posterior_std <= 0:
            msg = f"posterior_std must be > 0, got {posterior_std}."
            raise ValueError(msg)
        if prior_std <= 0:
            msg = f"prior_std must be > 0, got {prior_std}."
            raise ValueError(msg)
        self.hidden_dims = tuple(hidden_dims)
        self.posterior_std = posterior_std
        self.prior_std = prior_std
        self.kl_penalty = kl_penalty
        self.lr = lr
        self.epochs = epochs
        self.num_samples = num_samples
        self.seed = seed

        self._sampler: Representer | None = None
        self._x_mean: torch.Tensor | None = None
        self._x_std: torch.Tensor | None = None
        self._y_mean: torch.Tensor | None = None
        self._y_std: torch.Tensor | None = None

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Train the BNN with the ELBO on the current observations.

        Each gradient step samples weights from the variational posterior,
        evaluates MSE on standardized targets, and adds ``kl_penalty``
        times the KL divergence collected across all ``BayesLinear``
        layers via :func:`~probly.train.bayesian.torch.collect_kl_divergence`.
        """
        x_n, y_n, x_mean, x_std, y_mean, y_std = _standardize(x, y)

        torch.manual_seed(self.seed)
        base = _MLP(in_dim=x_n.shape[-1], hidden_dims=self.hidden_dims)
        bnn = bayesian(base, posterior_std=self.posterior_std, prior_std=self.prior_std)

        bnn.train()
        optim = torch.optim.Adam(bnn.parameters(), lr=self.lr)
        for _ in range(self.epochs):
            optim.zero_grad()
            pred = bnn(x_n)
            mse = nn.functional.mse_loss(pred, y_n)
            kl = collect_kl_divergence(bnn)
            loss = mse + self.kl_penalty * kl
            loss.backward()
            optim.step()

        bnn.eval()
        self._sampler = representer(bnn, num_samples=self.num_samples)
        self._x_mean = x_mean
        self._x_std = x_std
        self._y_mean = y_mean
        self._y_std = y_std

    def predict_representation(self, x: torch.Tensor) -> TorchSample[torch.Tensor]:
        """Return the BNN posterior at ``x`` as a :class:`TorchSample`."""
        if (
            self._sampler is None
            or self._x_mean is None
            or self._x_std is None
            or self._y_mean is None
            or self._y_std is None
        ):
            msg = "BNNSurrogate must be fit before calling predict_representation."
            raise RuntimeError(msg)
        x_n = (x.to(torch.float32) - self._x_mean) / self._x_std
        sample = self._sampler.represent(x_n)
        denormalized = sample.tensor * self._y_std + self._y_mean
        return TorchSample(
            tensor=denormalized.to(x.dtype),
            sample_dim=sample.sample_dim,
            weights=sample.weights,
        )
