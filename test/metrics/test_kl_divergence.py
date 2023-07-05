# Tests for the KL divergence

import unittest

import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.metrics import kl_divergence


class TestKLDivergence(unittest.TestCase):
    def setUp(self):
        self.mean_0 = torch.as_tensor([1, 2])
        self.mean_1 = torch.as_tensor([-1.0, 0.0])
        self.cov_0 = 9.0 * torch.eye(2)
        self.cov_1 = torch.as_tensor([[3, 0.1], [0.1, 3]])
        self.q = MultivariateNormal(self.mean_0, self.cov_0)
        self.p = MultivariateNormal(self.mean_1, self.cov_1)
        self.q_batch = MultivariateNormal(
            mean=torch.stack((self.mean_0, self.mean_0)),
            covariance_matrix=torch.stack((self.cov_0, self.cov_0)),
        )
        self.p_batch = MultivariateNormal(
            mean=torch.stack((self.mean_1, torch.zeros((2,)))),
            covariance_matrix=torch.stack((self.cov_1, 3 * torch.eye(2))),
        )

    def test_kldiv_of_same_args_is_zero(self):
        self.assertEqual(kl_divergence(self.q, self.q), torch.as_tensor(0.0))

    def test_kldiv_greater_or_equal_zero(self):
        self.assertGreaterEqual(kl_divergence(self.q, self.p).item(), 0.0)

    def test_kldiv_of_batches_of_randvars(self):
        kldivs_batch = kl_divergence(self.q_batch, self.p_batch)
        self.assertEqual(kldivs_batch.size()[0], self.q_batch.batch_shape[0])
