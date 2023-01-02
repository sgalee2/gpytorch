#!/usr/bin/env python3
import math

import torch
from linear_operator import to_linear_operator


class ComputationAwareMarginalLogLikelihood(torch.autograd.Function):
    """
    TODO
    """

    @staticmethod
    def forward(ctx, Khat: torch.Tensor, y: torch.Tensor, linear_solver):
        repr_weights, prec_approx, etas = linear_solver.solve(
            to_linear_operator(Khat), y
        )

        lml = -0.5 * (
            torch.inner(y, repr_weights)
            + torch.sum(torch.log(etas))
            + Khat.shape[-1] * math.log(2 * math.pi)
        )

        ctx.repr_weights = repr_weights
        ctx.prec_approx = prec_approx
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
        )
