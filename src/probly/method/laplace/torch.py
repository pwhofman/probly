"""Torch implementation of Laplace approximation, wrapping laplace-torch."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from laplace import Laplace
import torch
from torch import nn

from probly.representation.distribution import (
    CategoricalDistribution,
    DirichletDistribution,
    create_categorical_distribution,
    create_dirichlet_distribution_from_alphas,
)

from ._common import LaplaceGLMPredictor, LaplaceMCPredictor, laplace_generator

if TYPE_CHECKING:
    from typing import Any


@laplace_generator.register(nn.Module)
def _torch_laplace(
    base: nn.Module,
    pred_type: str = "glm",
    **laplace_kwargs: object,
) -> LaplaceGLMPredictor | LaplaceMCPredictor:
    """Pick a torch Laplace predictor subclass based on ``pred_type``."""
    if pred_type == "glm":
        return TorchLaplaceGLMPredictor(base, **laplace_kwargs)
    if pred_type == "nn":
        return TorchLaplaceMCPredictor(base, **laplace_kwargs)
    msg = f"pred_type must be 'glm' or 'nn', got {pred_type!r}"
    raise ValueError(msg)


def _fit_laplace(
    predictor: TorchLaplaceGLMPredictor | TorchLaplaceMCPredictor,
    loader: object,
    optimize_prior: bool,
    **kwargs: object,
) -> None:
    """Shared body for ``.fit`` on both torch Laplace predictor variants.

    Mutates ``predictor.la`` (fit) and ``predictor._fitted`` (flag).
    """
    cast("Any", predictor.la).fit(loader)
    predictor._fitted = True  # noqa: SLF001
    if optimize_prior:
        cast("Any", predictor.la).optimize_prior_precision(**kwargs)


def _predict_representation_categorical(
    predictor: TorchLaplaceGLMPredictor | TorchLaplaceMCPredictor,
    x: torch.Tensor,
    **kwargs: object,
) -> CategoricalDistribution:
    """Shared body for ``predict_representation`` on both torch Laplace variants.

    Wraps the underlying ``.predict(x, **kwargs)`` output in a
    :class:`CategoricalDistribution`. Only classification is supported today;
    regression raises :class:`NotImplementedError` until a torch-backed
    Gaussian distribution representation is added.
    """
    likelihood = getattr(predictor.la, "likelihood", None)
    if likelihood != "classification":
        msg = (
            "predict_representation is only implemented for "
            f"likelihood='classification', got {likelihood!r}. "
            "Call predictor.predict(x) directly to get the raw output."
        )
        raise NotImplementedError(msg)
    return create_categorical_distribution(predictor.predict(x, **kwargs))


class TorchLaplaceGLMPredictor(nn.Module, LaplaceGLMPredictor[[torch.Tensor], torch.Tensor]):
    """Torch Laplace predictor -- GLM (closed-form) prediction mode."""

    def __init__(self, model: nn.Module, **laplace_kwargs: object) -> None:
        """Construct an unfitted GLM Laplace wrapper around ``model``.

        Args:
            model: The trained torch ``nn.Module`` to wrap.
            **laplace_kwargs: Forwarded to ``laplace.Laplace(model, **kwargs)``.
        """
        super().__init__()
        self.la = cast("Any", Laplace)(model, **laplace_kwargs)
        self._fitted = False

    def fit(
        self,
        loader: object,
        optimize_prior: bool = False,
        **kwargs: object,
    ) -> None:
        """Fit the Laplace posterior on ``loader``.

        Args:
            loader: A torch data loader yielding ``(x, y)`` batches.
            optimize_prior: If ``True``, also call
                ``la.optimize_prior_precision(**kwargs)`` after fitting.
            **kwargs: Forwarded to ``optimize_prior_precision`` when
                ``optimize_prior=True``.
        """
        _fit_laplace(self, loader, optimize_prior, **kwargs)

    def predict(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """GLM closed-form prediction. Requires a prior call to ``.fit``.

        Args:
            x: Input batch.
            **kwargs: Forwarded to ``la(x, pred_type='glm', **kwargs)``
                (e.g. ``link_approx``, ``joint``).

        Returns:
            The GLM closed-form output (Gaussian mean+variance for
            regression; probit-approximated probabilities for
            classification).
        """
        if not self._fitted:
            msg = "Call .fit(loader) before predicting."
            raise RuntimeError(msg)
        return cast("Any", self.la)(x, pred_type="glm", **kwargs)

    def predict_representation(self, x: torch.Tensor, **kwargs: object) -> CategoricalDistribution:
        """Predict and wrap the result in a probly :class:`CategoricalDistribution`.

        Calls :meth:`predict` and wraps the resulting probabilities in a
        ``CategoricalDistribution``. This makes the predictor satisfy
        :class:`probly.predictor.RepresentationPredictor`, allowing it to
        flow through ``probly.predict`` and ``probly.representer.representer``
        without an explicit registration. Classification only.

        Args:
            x: Input batch.
            **kwargs: Forwarded to :meth:`predict`.

        Returns:
            A :class:`CategoricalDistribution` over the network's output
            classes.
        """
        return _predict_representation_categorical(self, x, **kwargs)

    def sample(
        self,
        x: torch.Tensor,
        n_samples: int = 100,
        **kwargs: object,
    ) -> torch.Tensor:
        """Sample from the closed-form GLM predictive distribution.

        For each input, the GLM Laplace approximation gives a Gaussian
        distribution over logits. This method samples ``n_samples`` draws
        from that Gaussian (and pushes them through the link function for
        classification, returning probability samples). Use this for
        downstream Monte Carlo metrics that need samples even though the
        GLM predictive is closed-form (e.g., epistemic/aleatoric
        decomposition via sample variance).

        Args:
            x: Input batch.
            n_samples: Number of samples to draw from the predictive
                Gaussian.
            **kwargs: Forwarded to ``la.predictive_samples(...)``.

        Returns:
            Predictive samples; shape follows ``la.predictive_samples``.
        """
        if not self._fitted:
            msg = "Call .fit(loader) before sampling."
            raise RuntimeError(msg)
        return cast("Any", self.la).predictive_samples(
            x,
            pred_type="glm",
            n_samples=n_samples,
            **kwargs,
        )

    def predict_dirichlet(self, x: torch.Tensor) -> DirichletDistribution:
        """Predict a closed-form Dirichlet over the simplex via the Laplace bridge.

        The Laplace bridge (Hobbhahn et al., 2022) approximates the integral
        of the GLM Gaussian over logits pushed through softmax with a
        Dirichlet over the simplex. Unlike :meth:`predict` with
        ``link_approx='probit'`` (which collapses to a single Categorical),
        this returns the full :class:`DirichletDistribution`, exposing both
        the predictive mean (~aleatoric) and the concentration
        (~epistemic) for downstream uncertainty decomposition.

        Classification only.

        .. note::

            laplace-torch's public ``link_approx='bridge'`` path normalizes
            the alphas before returning, discarding the concentration. To
            recover the raw Dirichlet alphas this method calls the
            ``BaseLaplace._glm_predictive_distribution`` private API and
            reimplements the bridge formula from
            ``laplace.baselaplace.BaseLaplace.predictive``. If laplace-torch
            ever exposes the alphas directly, this should be re-pointed.

        Args:
            x: Input batch.

        Returns:
            A :class:`DirichletDistribution` over the network's output
            classes.

        Raises:
            RuntimeError: If ``.fit(loader)`` has not been called.
            NotImplementedError: If ``likelihood != 'classification'``.
        """
        if not self._fitted:
            msg = "Call .fit(loader) before predicting."
            raise RuntimeError(msg)
        likelihood = getattr(self.la, "likelihood", None)
        if likelihood != "classification":
            msg = f"predict_dirichlet is only implemented for likelihood='classification', got {likelihood!r}."
            raise NotImplementedError(msg)

        la_any = cast("Any", self.la)
        f_mu, f_var = la_any._glm_predictive_distribution(x)  # noqa: SLF001

        # Zero-mean correction (mirrors laplace.baselaplace ll. 667-674).
        f_var_sum_last = f_var.sum(-1)
        f_var_sum_total = f_var.sum(dim=(1, 2)).reshape(-1, 1)
        f_mu = f_mu - f_var_sum_last * f_mu.sum(-1).reshape(-1, 1) / f_var_sum_total
        f_var = f_var - torch.einsum(
            "bi,bj->bij",
            f_var_sum_last,
            f_var.sum(-2),
        ) / f_var_sum_total.unsqueeze(-1)

        # Laplace bridge formula (mirrors laplace.baselaplace ll. 689-690).
        # We deliberately do NOT normalize alphas here: the public API divides
        # by ``alpha.sum(...)`` to return a Categorical, but that throws away
        # the concentration we want to expose.
        k = f_mu.size(-1)
        f_var_diag = torch.diagonal(f_var, dim1=1, dim2=2)
        sum_exp = torch.exp(-f_mu).sum(dim=1).unsqueeze(-1)
        alpha = (1 - 2 / k + f_mu.exp() / k**2 * sum_exp) / f_var_diag
        alpha = torch.nan_to_num(alpha, nan=1.0)

        return create_dirichlet_distribution_from_alphas(alpha)


class TorchLaplaceMCPredictor(nn.Module, LaplaceMCPredictor[[torch.Tensor], torch.Tensor]):
    """Torch Laplace predictor -- Monte Carlo posterior-sampling prediction mode."""

    def __init__(self, model: nn.Module, **laplace_kwargs: object) -> None:
        """Construct an unfitted MC Laplace wrapper around ``model``.

        Args:
            model: The trained torch ``nn.Module`` to wrap.
            **laplace_kwargs: Forwarded to ``laplace.Laplace(model, **kwargs)``.
        """
        super().__init__()
        self.la = cast("Any", Laplace)(model, **laplace_kwargs)
        self._fitted = False

    def fit(
        self,
        loader: object,
        optimize_prior: bool = False,
        **kwargs: object,
    ) -> None:
        """Fit the Laplace posterior on ``loader``.

        Args:
            loader: A torch data loader yielding ``(x, y)`` batches.
            optimize_prior: If ``True``, also call
                ``la.optimize_prior_precision(**kwargs)`` after fitting.
            **kwargs: Forwarded to ``optimize_prior_precision`` when
                ``optimize_prior=True``.
        """
        _fit_laplace(self, loader, optimize_prior, **kwargs)

    def predict(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """MC mean prediction (averages over posterior samples).

        Args:
            x: Input batch.
            **kwargs: Forwarded to ``la(x, pred_type='nn', **kwargs)``
                (e.g. ``n_samples``, ``link_approx``).

        Returns:
            The mean of the MC predictions.
        """
        if not self._fitted:
            msg = "Call .fit(loader) before predicting."
            raise RuntimeError(msg)
        return cast("Any", self.la)(x, pred_type="nn", **kwargs)

    def sample(
        self,
        x: torch.Tensor,
        n_samples: int = 100,
        **kwargs: object,
    ) -> torch.Tensor:
        """Posterior MC samples through the network.

        Args:
            x: Input batch.
            n_samples: Number of posterior samples to draw.
            **kwargs: Forwarded to ``la.predictive_samples(...)``.

        Returns:
            Predictive samples; shape follows ``la.predictive_samples``.
        """
        if not self._fitted:
            msg = "Call .fit(loader) before sampling."
            raise RuntimeError(msg)
        return cast("Any", self.la).predictive_samples(x, n_samples=n_samples, **kwargs)
