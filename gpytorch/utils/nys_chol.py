#!/usr/bin/env python3

import torch

from .. import settings

def cholesky_helper(matrix, max_iter, alg):

    device = matrix.device
    n, k = matrix.shape[0], max_iter

    G = torch.empty((k, n), device = device)
    diags = matrix.diag().detach().clone()

    idx_ = []

    for i in range(k):

        if alg == 'rp':
            idx = torch.multinomial(diags/torch.sum(diags), 1)
        elif alg == 'greedy':
            idx = torch.argmax(diags).reshape(-1)
        else:
            raise RuntimeError("Algorithm {} not recognised".format(alg))
        
        idx_.append(idx.item())

        G[i,:] = (matrix[idx,:] - G[:i,idx].T @ G[:i,:]).evaluate() / torch.sqrt(diags[idx])
        diags -= G[i,:]**2
        diags = diags.clip(min=0)

    return G, idx_
        
