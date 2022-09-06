#!/usr/bin/env python3

import torch

from ..likelihoods.multitask_gaussian_likelihood import MultitaskGaussianLikelihood
from .added_loss_term import AddedLossTerm


class InducingPointKernelAddedLossTerm(AddedLossTerm):
    def __init__(self, prior_dist, variational_dist, likelihood):
        self.prior_dist = prior_dist
        self.variational_dist = variational_dist
        self.likelihood = likelihood

    def loss(self, *params):
        prior_covar = self.prior_dist.lazy_covariance_matrix
        variational_covar = self.variational_dist.lazy_covariance_matrix
        diag = prior_covar.diagonal(dim1=-1, dim2=-2) - variational_covar.diagonal(dim1=-1, dim2=-2)
        shape = prior_covar.shape[:-1]
        if isinstance(self.likelihood, MultitaskGaussianLikelihood):
            shape = torch.Size([*shape, 1])
            diag = diag.unsqueeze(-1)
        noise_diag = self.likelihood._shaped_noise_covar(shape, *params).diagonal(dim1=-1, dim2=-2)
        if isinstance(self.likelihood, MultitaskGaussianLikelihood):
            noise_diag = noise_diag.reshape(*shape[:-1], -1)
            return -0.5 * (diag / noise_diag).sum(dim=[-1, -2])
        else:
            return -0.5 * (diag / noise_diag).sum(dim=-1)
