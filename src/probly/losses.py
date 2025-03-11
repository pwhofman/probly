import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidentialLogLoss(nn.Module):
    """
    https://arxiv.org/pdf/1806.01768
    """
    def __init__(self):
        super(EvidentialLogLoss, self).__init__()

    def forward(self, inputs, targets):
        alphas = inputs + 1.0
        strengths = torch.sum(alphas, dim=1)
        loss = torch.mean(torch.log(strengths) - torch.log(alphas[torch.arange(targets.shape[0]), targets]))
        return loss

class EvidentialCELoss(nn.Module):
    """
    https://arxiv.org/pdf/1806.01768
    """
    def __init__(self):
        super(EvidentialCELoss, self).__init__()

    def forward(self, inputs, targets):
        alphas = inputs + 1.0
        strengths = torch.sum(alphas, dim=1)
        loss = torch.mean(torch.digamma(strengths) - torch.digamma(alphas[torch.arange(targets.shape[0]), targets]))
        return loss

class EvidentialMSELoss(nn.Module):
    """
    https://arxiv.org/pdf/1806.01768
    """

    def __init__(self):
        super(EvidentialMSELoss, self).__init__()

    def forward(self, inputs, targets):
        alphas = inputs + 1.0
        strengths = torch.sum(alphas, dim=1)
        y = F.one_hot(targets, inputs.shape[1])
        p = alphas / strengths[:, None]
        err = (y - p) ** 2
        var = p * (1 - p) / (strengths[:, None] + 1)
        loss = torch.mean(torch.sum(err + var, dim=1))
        return loss

class EvidentialKLDivergence(nn.Module):
    """
    https://arxiv.org/pdf/1806.01768
    """
    def __init__(self):
        super(EvidentialKLDivergence, self).__init__()

    def forward(self, inputs, targets):
        alphas = inputs + 1.0
        y = F.one_hot(targets, inputs.shape[1])
        alphas_tilde = y + (1 - y) * alphas
        strengths_tilde = torch.sum(alphas_tilde, dim=1)
        K = torch.full((inputs.shape[0],), inputs.shape[1])
        first = (torch.lgamma(strengths_tilde) -
                 torch.lgamma(K) -
                 torch.sum(torch.lgamma(alphas_tilde), dim=1)
                 )
        second = torch.sum((alphas_tilde - 1) * (torch.digamma(alphas_tilde) - torch.digamma(strengths_tilde)[:, None]), dim=1)
        loss = torch.mean(first + second)
        return loss
