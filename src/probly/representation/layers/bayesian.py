import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class BayesLinear(nn.Module):
    """Implements a Bayesian linear layer.

    Attributes:
        in_features: int, number of input features
        out_features: int, number of output features
        bias: bool, whether to use a bias term
        weight_mu: torch.Tensor, mean of the posterior weights
        weight_rho: torch.Tensor, transformed standard deviation of the posterior weights
        weight_prior_mu: torch.Tensor, mean of the prior weights
        weight_prior_sigma: torch.Tensor, standard deviation of the prior weights
        bias_mu: torch.Tensor, mean of the posterior bias
        bias_rho: torch.Tensor, transformed standard deviation of the posterior bias
        bias_prior_mu: torch.Tensor, mean of the prior bias
        bias_prior_sigma: torch.Tensor, standard deviation of the prior bias
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        posterior_std: float = 0.05,
        prior_mean: float = 0.0,
        prior_std: float = 1.0,
        init_layer: nn.Module = None,
    ) -> None:
        """Initializes the Bayesian linear layer.

        Reparameterize the standard deviation of the posterior weights using the re-parameterization trick.
        Args:
            in_features: int, number of input features
            out_features: int, number of output features
            bias: bool, whether to use a bias term
            posterior_std: float, initial standard deviation of the posterior
            prior_mean: float, mean of the prior
            prior_std: float, standard deviation of the prior
            init_layer: nn.Module, layer to initialize the weights from
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        # transform standard deviation for the re-parametrization trick
        rho = torch.log(torch.exp(torch.tensor(posterior_std)) - 1)

        # posterior weights
        if init_layer is None:
            self.weight_mu = nn.Parameter(torch.empty((out_features, in_features)))
        else:
            self.weight_mu = nn.Parameter(init_layer.weight.data)
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), rho))

        # prior weights
        if init_layer is None:
            self.register_buffer(
                "weight_prior_mu", torch.full((out_features, in_features), prior_mean)
            )
        else:
            self.register_buffer("weight_prior_mu", init_layer.weight.data)
        self.register_buffer(
            "weight_prior_sigma", torch.full((out_features, in_features), prior_std)
        )

        if self.bias:
            # posterior bias
            if init_layer is None:
                self.bias_mu = nn.Parameter(torch.empty((out_features,)))
            else:
                self.bias_mu = nn.Parameter(init_layer.bias.data)
            self.bias_rho = nn.Parameter(torch.full((out_features,), rho))

            # prior bias
            if init_layer is None:
                self.register_buffer("bias_prior_mu", torch.full((out_features,), prior_mean))
            else:
                self.register_buffer("bias_prior_mu", init_layer.bias.data)
            self.register_buffer("bias_prior_sigma", torch.full((out_features,), prior_std))

        if init_layer is None:
            self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Bayesian linear layer.
        Args:
            x: torch.Tensor, input data
        Returns:
            torch.Tensor, layer output
        """
        eps_weight = torch.randn_like(self.weight_mu)
        weight = self.weight_mu + torch.log1p(torch.exp(self.weight_rho)) * eps_weight
        if self.bias:
            eps_bias = torch.randn_like(self.bias_mu)
            bias = self.bias_mu + torch.log1p(torch.exp(self.bias_rho)) * eps_bias
            x = F.linear(x, weight, bias)
        else:
            x = F.linear(x, weight)
        return x

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        if self.bias is not False:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_mu)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias_mu, -bound, bound)

    @property
    def kl_divergence(self) -> torch.Tensor:
        """
        Computes the KL-divergence between the posterior and prior.
        """
        kl = torch.sum(
            _kl_divergence_gaussian(
                self.weight_mu,
                torch.log1p(torch.exp(self.weight_rho)) ** 2,
                self.weight_prior_mu,
                self.weight_prior_sigma**2,
            )
        )
        if self.bias:
            kl += torch.sum(
                _kl_divergence_gaussian(
                    self.bias_mu,
                    torch.log1p(torch.exp(self.bias_rho)) ** 2,
                    self.bias_prior_mu,
                    self.bias_prior_sigma**2,
                )
            )
        return kl


class BayesConv2d(nn.Module):
    """Implementation of a Bayesian convolutional layer.

    Attributes:
        in_channels: int, number of input channels
        out_channels: int, number of output channels
        kernel_size: int or tuple, size of the convolutional kernel
        stride: int or tuple, stride of the convolution
        padding: int or tuple, padding of the convolution
        dilation: int or tuple, dilation of the convolution
        groups: int, number of groups for grouped convolution
        bias: bool, whether to use a bias term
        weight_mu: torch.Tensor, mean of the posterior weights
        weight_rho: torch.Tensor, transformed standard deviation of the posterior weights
        weight_prior_mu: torch.Tensor, mean of the prior weights
        weight_prior_sigma: torch.Tensor, standard deviation of the prior weights
        bias_mu: torch.Tensor, mean of the posterior bias
        bias_rho: torch.Tensor, transformed standard deviation of the posterior bias
        bias_prior_mu: torch.Tensor, mean of the prior bias
        bias_prior_sigma: torch.Tensor, standard deviation of the prior bias
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple,
        stride: int | tuple = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        posterior_std: float = 0.05,
        prior_mean: float = 0.0,
        prior_std: float = 1.0,
        init_layer: nn.Module = None,
    ) -> None:
        """Initializes the Bayesian convolutional layer.

        Reparameterize the standard deviation of the posterior weights using the re-parameterization trick.

        Args:
            in_channels: int, number of input channels
            out_channels: int, number of output channels
            kernel_size: int or tuple, size of the convolutional kernel
            stride: int or tuple, stride of the convolution
            padding: int or tuple, padding of the convolution
            dilation: int or tuple, dilation of the convolution
            groups: int, number of groups for grouped convolution
            bias: bool, whether to use a bias term
            posterior_std: float, initial standard deviation of the posterior
            prior_mean: float, mean of the prior
            prior_std: float, standard deviation of the prior
            init_layer: nn.Module, layer to initialize the weights from
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        # transform standard deviation for the re-parametrization trick
        rho = torch.log(torch.exp(torch.tensor(posterior_std)) - 1)

        # posterior weights
        if init_layer is None:
            self.weight_mu = nn.Parameter(
                torch.empty((out_channels, in_channels // groups, *kernel_size))
            )
        else:
            self.weight_mu = nn.Parameter(init_layer.weight.data)
        self.weight_rho = nn.Parameter(
            torch.full((out_channels, in_channels // groups, *kernel_size), rho)
        )

        # prior weights
        if init_layer is None:
            self.register_buffer(
                "weight_prior_mu",
                torch.full((out_channels, in_channels // groups, *kernel_size), prior_mean),
            )
        else:
            self.register_buffer("weight_prior_mu", init_layer.weight.data)

        self.register_buffer(
            "weight_prior_sigma",
            torch.full((out_channels, in_channels // groups, *kernel_size), prior_std),
        )

        if self.bias:
            # posterior bias
            if init_layer is None:
                self.bias_mu = nn.Parameter(torch.empty((out_channels,)))
            else:
                self.bias_mu = nn.Parameter(init_layer.bias.data)
            self.bias_rho = nn.Parameter(torch.full((out_channels,), rho))

            # prior bias
            if init_layer is None:
                self.register_buffer("bias_prior_mu", torch.full((out_channels,), prior_mean))
            else:
                self.register_buffer("bias_prior_mu", init_layer.bias.data)
            self.register_buffer("bias_prior_sigma", torch.full((out_channels,), prior_std))

        if init_layer is None:
            self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Bayesian conv2d layer.
        Args:
            x: torch.Tensor, input data
        Returns:
            torch.Tensor, layer output
        """
        eps_weight = torch.randn_like(self.weight_mu)
        weight = self.weight_mu + torch.log1p(torch.exp(self.weight_rho)) * eps_weight
        if self.bias:
            eps_bias = torch.randn_like(self.bias_mu)
            bias = self.bias_mu + torch.log1p(torch.exp(self.bias_rho)) * eps_bias
            x = F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            x = F.conv2d(
                x,
                weight,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        return x

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        if self.bias is not False:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_mu)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias_mu, -bound, bound)

    @property
    def kl_divergence(self) -> torch.Tensor:
        """
        Compute the KL-divergence between the posterior and prior.
        """
        kl = torch.sum(
            _kl_divergence_gaussian(
                self.weight_mu,
                torch.log1p(torch.exp(self.weight_rho)) ** 2,
                self.weight_prior_mu,
                self.weight_prior_sigma**2,
            )
        )
        if self.bias:
            kl += torch.sum(
                _kl_divergence_gaussian(
                    self.bias_mu,
                    torch.log1p(torch.exp(self.bias_rho)) ** 2,
                    self.bias_prior_mu,
                    self.bias_prior_sigma**2,
                )
            )
        return kl


def _kl_divergence_gaussian(
    mu1: torch.Tensor, sigma21: torch.Tensor, mu2: torch.Tensor, sigma22: torch.Tensor
) -> torch.Tensor:
    """
    Compute the KL-divergence between two Gaussian distributions.
    https://en.wikipedia.org/wiki/Kullback–Leibler_divergence#Examples
    Args:
        mu1: torch.Tensor, mean of the first Gaussian distribution
        sigma21: torch.Tensor, variance of the first Gaussian distribution
        mu2: torch.Tensor, mean of the second Gaussian distribution
        sigma22: torch.Tensor, variance of the second Gaussian distribution
    Returns:
        kl_div: float or numpy.ndarray shape (n_instances,), KL-divergence between the two Gaussian distributions
    """
    kl_div = 0.5 * torch.log(sigma22 / sigma21) + (sigma21 + (mu1 - mu2) ** 2) / (2 * sigma22) - 0.5
    return kl_div
