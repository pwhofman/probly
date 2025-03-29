import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from ...utils import kl_divergence_gaussian_torch


class BayesLinear(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 posterior_std=0.05,
                 prior_mean=0.0,
                 prior_std=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        # transform standard deviation for the re-parametrization trick
        rho = torch.log(torch.exp(torch.tensor(posterior_std)) - 1)

        # posterior weights
        self.weight_mu = nn.Parameter(torch.empty((out_features, in_features)))
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), rho))

        # prior weights
        self.weight_prior_mu = torch.full((out_features, in_features), prior_mean)
        self.weight_prior_sigma = torch.full((out_features, in_features), prior_std)

        if self.bias:
            # posterior bias
            self.bias_mu = nn.Parameter(torch.empty((out_features,)))
            self.bias_rho = nn.Parameter(torch.full((out_features,), rho))

            # prior bias
            self.bias_prior_mu = torch.full((out_features,), prior_mean)
            self.bias_prior_sigma = torch.full((out_features,), prior_std)

        self.reset_parameters()


    def forward(self, x):
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
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_mu)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias_mu, -bound, bound)

    @property
    def kl_divergence(self):
        """
        Computer the KL-divergence between the Gaussian posterior and prior.
        """
        kl = torch.sum(kl_divergence_gaussian_torch(self.weight_mu,
                                          torch.log1p(torch.exp(self.weight_rho)) ** 2,
                                          self.weight_prior_mu,
                                          self.weight_prior_sigma ** 2, base=torch.tensor(torch.e)))
        if self.bias:
            kl += torch.sum(kl_divergence_gaussian_torch(self.bias_mu,
                                                  torch.log1p(torch.exp(self.bias_rho)) ** 2,
                                                  self.bias_prior_mu,
                                                  self.bias_prior_sigma ** 2, base=torch.tensor(torch.e)))
        return kl

class BayesConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 posterior_std=0.05,
                 prior_mean=0.0,
                 prior_std=1.0):
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
        self.weight_mu = nn.Parameter(
            torch.empty((out_channels, in_channels // groups, *kernel_size)))
        self.weight_rho = nn.Parameter(
            torch.full((out_channels, in_channels // groups, *kernel_size), rho))

        # prior weights
        self.weight_prior_mu = (
            torch.full((out_channels, in_channels // groups, *kernel_size), prior_mean))
        self.weight_prior_sigma = (
            torch.full((out_channels, in_channels // groups, *kernel_size), prior_std))

        if self.bias:
            # posterior bias
            self.bias_mu = nn.Parameter(torch.empty((out_channels,)))
            self.bias_rho = nn.Parameter(torch.full((out_channels,), rho))

            # prior bias
            self.bias_prior_mu = torch.full((out_channels,), prior_mean)
            self.bias_prior_sigma = torch.full((out_channels,), prior_std)

        self.reset_parameters()

    def forward(self, x):
        eps_weight = torch.randn_like(self.weight_mu)
        weight = self.weight_mu + torch.log1p(torch.exp(self.weight_rho)) * eps_weight
        if self.bias:
            eps_bias = torch.randn_like(self.bias_mu)
            bias = self.bias_mu + torch.log1p(torch.exp(self.bias_rho)) * eps_bias
            x = F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            x = F.conv2d(x,
                         weight,
                         stride=self.stride,
                         padding=self.padding,
                         dilation=self.dilation,
                         groups=self.groups)
        return x

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_mu)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias_mu, -bound, bound)

    @property
    def kl_divergence(self):
        """
        Computer the KL-divergence between the Gaussian posterior and prior.
        """
        kl = torch.sum(kl_divergence_gaussian_torch(self.weight_mu,
                                                    torch.log1p(torch.exp(self.weight_rho)) ** 2,
                                                    self.weight_prior_mu,
                                                    self.weight_prior_sigma ** 2, base=torch.tensor(torch.e)))
        if self.bias:
            kl += torch.sum(kl_divergence_gaussian_torch(self.bias_mu,
                                                         torch.log1p(torch.exp(self.bias_rho)) ** 2,
                                                         self.bias_prior_mu,
                                                         self.bias_prior_sigma ** 2, base=torch.tensor(torch.e)))
        return kl
