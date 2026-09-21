"""Backend-agnostic training losses."""

from __future__ import annotations

from flextype import flexdispatch


@flexdispatch
def vbll_loss[T](layer: object, features: T, targets: T, regularization_weight: float) -> T:
    """Compute the negative VBLL ELBO from :cite:`harrisonVariationalBayesian2024`.

    Routes to the variant-specific training objective based on the type of ``layer``
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
