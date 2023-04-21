#!/usr/bin/env python3
from __future__ import annotations

import math
from typing import Union

import torch
from linear_operator import operators, to_linear_operator

from ..likelihoods import GaussianLikelihood, _GaussianLikelihoodBase
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

    def __init__(self, likelihood: GaussianLikelihood, model: "ComputationAwareGP"):
        if not isinstance(likelihood, _GaussianLikelihoodBase):
            raise RuntimeError("Likelihood must be Gaussian for exact inference.")
        super().__init__(likelihood, model)

    def forward(self, output: torch.Tensor, target: torch.Tensor, **kwargs):

        Khat = self.likelihood(output).covariance_matrix  # TODO: creates n_train x n_train matrix, does not cause error
        # Khat = self.likelihood(output).lazy_covariance_matrix.evaluate_kernel() # TODO: causes error (detached from graph?)
        # Khat = to_linear_operator(self.likelihood(output).covariance_matrix)  # TODO: also causes error

        with torch.no_grad():
            solver_state = self.model.linear_solver.solve(to_linear_operator(Khat), target)
            if self.model.prediction_strategy is None:
                self.model._solver_state = solver_state

        repr_weights = solver_state.solution
        Khat_inv_approx = solver_state.inverse_op
        logdet_estimate = solver_state.logdet

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
        Khat: Union[torch.Tensor, operators.LinearOperator],
        y: torch.Tensor,
        repr_weights: torch.Tensor,
        Khat_inv_approx: torch.Tensor,
        logdet_estimate: torch.Tensor,
    ):
        lml = -0.5 * (torch.inner(y, repr_weights) + logdet_estimate + Khat.shape[-1] * math.log(2 * math.pi))

        ctx.repr_weights = repr_weights
        ctx.prec_approx = Khat_inv_approx
        ctx.Khat = Khat

        return lml

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        def _pseudo_lml(Khat):

            # Computes d/dtheta (v' Khat v) = v' dK/dtheta v
            fit_term = torch.inner(ctx.repr_weights, Khat @ ctx.repr_weights)

            # Computes d/dtheta (\sum_{j=1}^i 1/eta_j d_j' Khat d_j) = \sum_{j=1}^i 1/eta_j d_j' dK/dtheta d_j
            if isinstance(ctx.prec_approx, operators.ZeroLinearOperator):
                logdet_term = torch.as_tensor(0.0)
            else:
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