#!/usr/bin/env python3
from __future__ import annotations

import math

import torch
from gpytorch import settings
from linear_operator import to_linear_operator

from ..likelihoods import GaussianLikelihood, _GaussianLikelihoodBase
from ..models import ExactGP
from .marginal_log_likelihood import MarginalLogLikelihood


class ComputationAwareMarginalLogLikelihood(MarginalLogLikelihood):
    """
    Computation-aware marginal log-likelihood for a Gaussian process with a computation-aware Gaussian likelihood.


    :param ~gpytorch.likelihoods.GaussianLikelihood likelihood: The Gaussian likelihood for the model
    :param ~gpytorch.models.ExactGP model: The exact GP model

    Example:
        >>> mll = gpytorch.mlls.ComputationAwareMarginalLogLikelihood(likelihood, model)
        # TODO
    """

    def __init__(self, likelihood: GaussianLikelihood, model: ExactGP, linear_solver):
        if not isinstance(likelihood, _GaussianLikelihoodBase):
            raise RuntimeError("Likelihood must be Gaussian for exact inference.")
        self.linear_solver = linear_solver  # TODO: pass linear solver to `ComputationAwareGP` not to likelihood
        super().__init__(likelihood, model)

    def forward(self, output: torch.Tensor, target: torch.Tensor, **kwargs):

        with settings.prior_mode(True):
            Khat = self.likelihood(
                self.model(*self.model.train_inputs)
            ).covariance_matrix

        solver_state = self.linear_solver.solve(
            to_linear_operator(Khat), target  # TODO: do not omit prior mean here
        )
        repr_weights = solver_state.solution
        Khat_inv_approx = solver_state.inverse_op
        logdet_estimate = solver_state.logdet
        # TODO: do not do another linear solve in here, rather pass results from solve stored in model, after model(train_x) call

        # Implementing this via an autograd function is the recommended pattern by
        # PyTorch for extending nn.Module with a custom backward pass.
        # See also: https://pytorch.org/docs/stable/notes/extending.html#extending-torch-nn
        return _ComputationAwareMarginalLogLikelihoodFunction.apply(
            Khat, target, repr_weights, Khat_inv_approx, logdet_estimate
        )


class _ComputationAwareMarginalLogLikelihoodFunction(torch.autograd.Function):
    """Autograd function computing the computation-aware marginal log-likelihood."""

    @staticmethod
    def forward(
        ctx,
        Khat: torch.Tensor,
        y: torch.Tensor,
        repr_weights: torch.Tensor,
        Khat_inv_approx: torch.Tensor,
        logdet_estimate: torch.Tensor,
    ):
        lml = -0.5 * (
            torch.inner(y, repr_weights)
            + logdet_estimate
            + Khat.shape[-1] * math.log(2 * math.pi)
        )

        ctx.repr_weights = repr_weights
        ctx.prec_approx = Khat_inv_approx
        ctx.Khat = Khat

        return lml

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        def _pseudo_lml(Khat):

            # Computes d/dtheta (v' Khat v) = v' dK/dtheta v
            fit_term = ctx.repr_weights.T @ Khat @ ctx.repr_weights

            # Computes d/dtheta (\sum_{j=1}^i 1/eta_j d_j' Khat d_j) = \sum_{j=1}^i 1/eta_j d_j' dK/dtheta d_j
            logdet_term = torch.sum(
                torch.einsum(
                    "ni,nn,ni->i",
                    ctx.prec_approx.root.to_dense(),
                    Khat,
                    ctx.prec_approx.root.to_dense(),
                )
            )
            return 0.5 * (fit_term - logdet_term)

        return (
            torch.autograd.functional.vjp(_pseudo_lml, ctx.Khat, v=grad_output)[1],
            None,
            None,
            None,
            None,
        )
