#!/usr/bin/env python3
from __future__ import annotations

import math
from collections import deque
from typing import Tuple

import torch
from linear_operator import operators

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
        Khat = self.likelihood(output).lazy_covariance_matrix.evaluate_kernel()

        with torch.no_grad():
            solver_state = self.model.linear_solver.solve(Khat, target)
            if self.model.prediction_strategy is None:
                self.model._solver_state = solver_state

        repr_weights = solver_state.solution
        Khat_inv_approx = solver_state.inverse_op
        logdet_estimate = solver_state.logdet

        # Implementing this via an autograd function is the recommended pattern by
        # PyTorch for extending nn.Module with a custom backward pass.
        # See also: https://pytorch.org/docs/stable/notes/extending.html#extending-torch-nn
        return _ComputationAwareMarginalLogLikelihoodFunction.apply(
            Khat.representation_tree(),
            target,
            repr_weights,
            Khat_inv_approx,
            logdet_estimate,
            *Khat.representation(),
        )


class _ComputationAwareMarginalLogLikelihoodFunction(torch.autograd.Function):
    """Autograd function computing the computation-aware marginal log-likelihood."""

    @staticmethod
    def forward(
        ctx,
        Khat_representation_tree,
        y: torch.Tensor,
        repr_weights: torch.Tensor,
        Khat_inv_approx: torch.Tensor,
        logdet_estimate: torch.Tensor,
        *linear_op_args: torch.Tensor,
    ):
        # Reconstruct Khat from representation tree
        Khat = Khat_representation_tree(*linear_op_args)

        # Log marginal likelihood
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
        # Create tensor for bilinear forms
        # Computes d/dtheta (v' Khat v) = v' dK/dtheta v
        bilinear_form_tensor_list_left = [-ctx.repr_weights.reshape(-1, 1)]
        bilinear_form_tensor_list_right = [ctx.repr_weights.reshape(-1, 1)]
        if not isinstance(ctx.prec_approx, operators.ZeroLinearOperator):
            # Computes d/dtheta (\sum_{j=1}^i 1/eta_j d_j' Khat d_j) = \sum_{j=1}^i 1/eta_j d_j' dK/dtheta d_j
            bilinear_form_tensor_list_left.append(ctx.prec_approx.root.to_dense())
            bilinear_form_tensor_list_right.append(ctx.prec_approx.root.to_dense())

        bilinear_form_tensors_left = 0.5 * torch.cat(bilinear_form_tensor_list_left, -1)
        bilinear_form_tensors_right = torch.cat(bilinear_form_tensor_list_right, -1)

        return (
            None,
            None,
            None,
            None,
            None,
            *ctx.Khat._bilinear_derivative(
                bilinear_form_tensors_left, bilinear_form_tensors_right
            ),
        )


class SparseComputationAwareMarginalLogLikelihood(MarginalLogLikelihood):
    def __init__(self, likelihood: GaussianLikelihood, model: "ComputationAwareGP"):
        if not isinstance(likelihood, _GaussianLikelihoodBase):
            raise RuntimeError("Likelihood must be Gaussian for exact inference.")
        super().__init__(likelihood, model)

    def forward(self, output: torch.Tensor, target: torch.Tensor, **kwargs):
        Khat = self.likelihood(output).lazy_covariance_matrix.evaluate_kernel()

        with torch.no_grad():
            solver_state = self.model.linear_solver.solve(Khat, target)
            if self.model.prediction_strategy is None:
                self.model._solver_state = solver_state

        compressed_repr_weights = solver_state.cache["compressed_solution"]
        repr_weights = solver_state.solution
        actions = solver_state.cache["actions"]
        cholfac_gram = solver_state.cache["cholfac_gram"]
        logdet_estimate = solver_state.logdet

        # Implementing this via an autograd function is the recommended pattern by
        # PyTorch for extending nn.Module with a custom backward pass.
        # See also: https://pytorch.org/docs/stable/notes/extending.html#extending-torch-nn
        return _SparseComputationAwareMarginalLogLikelihoodFunction.apply(
            Khat.representation_tree(),
            target,
            compressed_repr_weights,
            repr_weights,
            actions,
            cholfac_gram,
            logdet_estimate,
            *Khat.representation(),
        )


class _SparseComputationAwareMarginalLogLikelihoodFunction(torch.autograd.Function):
    """Autograd function computing the computation-aware marginal log-likelihood."""

    @staticmethod
    def forward(
        ctx,
        Khat_representation_tree,
        y: torch.Tensor,
        compressed_repr_weights: torch.Tensor,
        repr_weights: torch.Tensor,
        actions: torch.Tensor,
        cholfac_gram: torch.Tensor,
        logdet_estimate: torch.Tensor,
        *linear_op_args: torch.Tensor,
    ):
        # Reconstruct Khat from representation tree
        Khat = Khat_representation_tree(*linear_op_args)

        # Log marginal likelihood
        lml = -0.5 * (
            torch.inner(y, repr_weights)
            + logdet_estimate
            + Khat.shape[-1] * math.log(2 * math.pi)
        )

        ctx.Khat = Khat
        ctx.compressed_repr_weights = compressed_repr_weights
        ctx.repr_weights = repr_weights
        ctx.actions = actions
        ctx.cholfac_gram = cholfac_gram

        return lml

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            *_custom_derivative(
                ctx.Khat,
                ctx.actions,
                ctx.compressed_repr_weights,
                ctx.cholfac_gram,
            ),
        )


def _custom_derivative(
    Khat: operators.LinearOperator,
    actions: torch.Tensor,
    compressed_repr_weights: torch.Tensor,
    cholfac_gram: torch.Tensor,
) -> Tuple[torch.Tensor, ...]:
    args = tuple(Khat.representation())
    args_with_grads = tuple(arg for arg in args if arg.requires_grad)

    # No gradients required
    if not len(args_with_grads):
        return tuple(None for _ in args)

    def _neglml_derivative(*representation):
        lin_op_copy = Khat.representation_tree()(*representation)
        gram_SKS = actions.mT @ (lin_op_copy._matmul(actions))
        quadratic_loss_term = torch.inner(
            compressed_repr_weights, gram_SKS @ compressed_repr_weights
        )
        complexity_term = torch.trace(
            torch.cholesky_solve(gram_SKS, cholfac_gram, upper=False)
        )
        return -0.5 * (quadratic_loss_term - complexity_term)

    actual_grads = deque(
        torch.autograd.functional.vjp(_neglml_derivative, Khat.representation())[1]
    )

    # Now make sure that the object we return has one entry for every item in args
    grads = []
    for arg in args:
        if arg.requires_grad:
            grads.append(actual_grads.popleft())
        else:
            grads.append(None)

    return tuple(grads)
