import torch

from .. import settings

"""
GPyTorch adapted function to compute several Krylov iteration algorithms.
"""


def bksvd(tensor):

    # Key values
    n, k, power = tensor.shape[0], settings.max_preconditioner_size.value(), settings.subspace_iters.value()
    dtype, device = tensor.dtype, tensor.device

    # Initialise tensors
    K = torch.zeros((n, k*power), dtype=dtype, device=device)
    T = torch.zeros((n, k), dtype=dtype, device=device)
    block = torch.randn(n, k, dtype=dtype, device=device)
    block, _ = torch.linalg.qr(block, mode="reduced")

    # Define matmul functions
    matmul = tensor.matmul
    tmatmul = tensor._t_matmul

    # Main recursion
    for i in range(power):
        T = matmul(block)
        block = tmatmul(T)
        block, _ = torch.linalg.qr(block, mode="reduced")
        K[:, i*k:(i+1)*k] = block
    # Final QR
    Q, _ = torch.linalg.qr(K, mode="reduced")
    T = matmul(Q)

    # Economy SVD
    Ut, St, _ = torch.linalg.svd(T, full_matrices=False)

    return Ut[:,:k], St[:k]
