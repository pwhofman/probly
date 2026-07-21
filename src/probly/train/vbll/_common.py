"""Shared VBLL training utilities.

Provides the backend-agnostic :func:`vbll_loss` generic that dispatches to the
variant-specific negative ELBO based on the layer type. Use
:func:`probly.method.vbll.find_vbll_layer` to retrieve the VBLL layer from a
transformed predictor.
"""

from __future__ import annotations

from flextype import flexdispatch


@flexdispatch
def vbll_loss[T](layer: object, features: T, targets: T, regularization_weight: float) -> T:
    """Compute the negative ELBO of a VBLL layer, dispatching on the layer type.

    Routes to the variant-specific training objective of
    :cite:`harrisonVariationalBayesian2024` based on the type of ``layer``
    (e.g. the double-Jensen bound for a
    :class:`~probly.layers.torch.VBLLLayer` or the generative Jensen bound for
    a :class:`~probly.layers.torch.GVBLLLayer`). Use
    :func:`probly.method.vbll.find_vbll_layer` to retrieve the layer from a
    transformed predictor.

    Args:
        layer: The variational Bayesian last layer to fit.
        features: Backbone features feeding the layer, shape ``(batch, in_features)``.
        targets: Integer class labels, shape ``(batch,)``.
        regularization_weight: Weight on the KL/regularization terms
            (typically ``1 / dataset_size``).

    Returns:
        A scalar tensor with the negative ELBO to minimize.
    """
    msg = f"vbll_loss is not implemented for layers of type {type(layer)}."
    raise NotImplementedError(msg)
