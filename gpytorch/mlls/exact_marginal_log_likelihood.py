#!/usr/bin/env python3

import math

from linear_operator.linear_solvers import CGGpytorch, LinearSolver

from ..distributions import MultivariateNormal
from ..likelihoods import _GaussianLikelihoodBase
from .marginal_log_likelihood import MarginalLogLikelihood


class ExactMarginalLogLikelihood(MarginalLogLikelihood):
    """
    The exact marginal log likelihood (MLL) for an exact Gaussian process with a
    Gaussian likelihood.

    .. note::
        This module will not work with anything other than a :obj:`~gpytorch.likelihoods.GaussianLikelihood`
        and a :obj:`~gpytorch.models.ExactGP`. It also cannot be used in conjunction with
        stochastic optimization.

    :param ~gpytorch.likelihoods.GaussianLikelihood likelihood: The Gaussian likelihood for the model
    :param ~gpytorch.models.ExactGP model: The exact GP model

    Example:
        >>> # model is a gpytorch.models.ExactGP
        >>> # likelihood is a gpytorch.likelihoods.Likelihood
        >>> mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        >>>
        >>> output = model(train_x)
        >>> loss = -mll(output, train_y)
        >>> loss.backward()
    """

    def __init__(self, likelihood, model):
        if not isinstance(likelihood, _GaussianLikelihoodBase):
            raise RuntimeError("Likelihood must be Gaussian for exact inference")
        super(ExactMarginalLogLikelihood, self).__init__(likelihood, model)

    def _add_other_terms(self, res, params):
        # Add additional terms (SGPR / learned inducing points, heteroskedastic likelihood models)
        for added_loss_term in self.model.added_loss_terms():
            res = res.add(added_loss_term.loss(*params))

        # Add log probs of priors on the (functions of) parameters
        res_ndim = res.ndim
        for name, module, prior, closure, _ in self.named_priors():
            prior_term = prior.log_prob(closure(module))
            res.add_(prior_term.view(*prior_term.shape[:res_ndim], -1).sum(dim=-1))

        return res

    def forward(self, function_dist, target, *params):
        r"""
        Computes the MLL given :math:`p(\mathbf f)` and :math:`\mathbf y`.

        :param ~gpytorch.distributions.MultivariateNormal function_dist: :math:`p(\mathbf f)`
            the outputs of the latent function (the :obj:`gpytorch.models.ExactGP`)
        :param torch.Tensor target: :math:`\mathbf y` The target values
        :rtype: torch.Tensor
        :return: Exact MLL. Output shape corresponds to batch shape of the model/input data.
        """
        if not isinstance(function_dist, MultivariateNormal):
            raise RuntimeError("ExactMarginalLogLikelihood can only operate on Gaussian random variables")

        # Get the log prob of the marginal distribution
        output = self.likelihood(function_dist, *params)
        res = output.log_prob(target)
        res = self._add_other_terms(res, params)

        # Scale by the amount of data we have
        num_data = function_dist.event_shape.numel()
        return res.div_(num_data)


class SLQMarginalLogLikelihood(ExactMarginalLogLikelihood):
    """Marginal log-likelihood computed via stochastic Lanczos quadrature and CG."""

    def __init__(self, likelihood, model, linear_solver: LinearSolver = None):

        if linear_solver is not None:
            self.linear_solver = linear_solver
        else:
            try:
                self.linear_solver = model.linear_solver
            except AttributeError:
                self.linear_solver = CGGpytorch()

        if not isinstance(likelihood, _GaussianLikelihoodBase):
            raise RuntimeError("Likelihood must be Gaussian for exact inference")
        super().__init__(likelihood, model)

    def forward(self, function_dist, target, *params):
        if not isinstance(function_dist, MultivariateNormal):
            raise RuntimeError("CGMarginalLogLikelihood can only operate on Gaussian random variables")

        # Get the log prob of the marginal distribution
        output = self.likelihood(function_dist, *params)
        res = _log_prob_via_slq(output, target, linear_solver=self.linear_solver)
        res = self._add_other_terms(res, params)

        # Scale by the amount of data we have
        num_data = function_dist.event_shape.numel()
        return res.div_(num_data)


def _log_prob_via_slq(self, value, linear_solver):

    if self._validate_args:
        self._validate_sample(value)

    mean, covar = self.loc, self.lazy_covariance_matrix
    diff = value - mean

    # Repeat the covar to match the batch shape of diff
    if diff.shape[:-1] != covar.batch_shape:
        if len(diff.shape[:-1]) < len(covar.batch_shape):
            diff = diff.expand(covar.shape[:-1])
        else:
            padded_batch_shape = (*(1 for _ in range(diff.dim() + 1 - covar.dim())), *covar.batch_shape)
            covar = covar.repeat(
                *(diff_size // covar_size for diff_size, covar_size in zip(diff.shape[:-1], padded_batch_shape)),
                1,
                1,
            )

    # Get log determininant and first part of quadratic form
    covar = covar.evaluate_kernel()
    covar.linear_solver = linear_solver  # Set linear solver for inv_quad_logdet computation

    inv_quad, logdet = covar.inv_quad_logdet(inv_quad_rhs=diff.unsqueeze(-1), logdet=True)

    res = -0.5 * sum([inv_quad, logdet, diff.size(-1) * math.log(2 * math.pi)])
    return res
