"""
Linear algebra helpers.

Adapted from SVISE, reference/svise/svise/utils.py (GPL-3.0). Changes: bjorck creates its identity with the
dtype and device of its input.
"""
import math

import torch
from torch import Tensor
import torch.nn as nn


def finite_difference(func, t, dt):
    """Central finite difference approximation of the derivative of func at t."""
    return (func(t + dt) - func(t - dt)) / (2 * dt)


def sample_covariance(x: Tensor) -> Tensor:
    """
    Sample covariance of x.

    Args:
        x (Tensor): (..., bs, d) batch of inputs

    Returns:
        Tensor: (..., d, d) sample covariance
    """
    N = x.shape[-2]
    mu = x.mean(-2).unsqueeze(-2)
    diff = x - mu
    return 1 / (N - 1) * diff.transpose(-1, -2) @ diff


def make_random_matrix(dim: int, rank: int, random_seed: int = None) -> Tensor:
    """Random symmetric positive semi-definite (dim, dim) matrix of the given rank."""
    if random_seed is not None:
        torch.manual_seed(random_seed)
    assert dim >= rank, "rank cannot be greater than dim."
    u = torch.randn(rank, dim, 1)
    return (u @ u.transpose(-2, -1)).sum(0).div(rank)


def bjorck(A: Tensor, tol: float = 1e-9, max_iters: int = 50, order: int = 1, num_iters=None):
    """
    Closest orthonormal matrix to A by Bjorck orthonormalization.

    Returns:
        (Tensor, int): orthonormal matrix and the number of iterations used
    """
    N, M = A.shape[-2:]
    scale = math.sqrt(N * M)  # safe bjorck scaling
    I = torch.eye(M, dtype=A.dtype, device=A.device)
    Ak = A / scale
    for j in range(max_iters):
        Qk = I - Ak.transpose(-2, -1) @ Ak
        if num_iters is None:
            if Qk.pow(2).sum() < tol:
                break  # squared Frobenius norm below tol
        elif j == num_iters:
            break
        if order == 1:
            Ak = Ak @ (I + 1 / 2 * Qk)
        elif order == 2:
            Ak = Ak @ (I + 1 / 2 * Qk + 3 / 8 * (Qk @ Qk))
        else:
            raise ValueError(f"Order: {order} not supported.")
    return Ak, j


def solve_least_squares(A: Tensor, B: Tensor, gamma: float = 1e-6) -> Tensor:
    """
    Ridge-regularised least squares by SVD: W minimising ||A W - B||^2 + gamma ||W||^2.

    Args:
        A (Tensor): (n, m) design matrix
        B (Tensor): (n, d) right hand side

    Returns:
        Tensor: (m, d) solution W
    """
    assert A.dim() == 2, "A must be a matrix"
    assert B.dim() == 2, "B must be a matrix"
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    return Vh.t() @ (U.t() @ B).mul((S / (S.pow(2) + gamma)).unsqueeze(-1))


def solve_lyapunov_spectral(D: Tensor, Q: Tensor, RHS: Tensor) -> Tensor:
    """
    Solves X K^T + K X^T = RHS for symmetric X, with the spectral decomposition K = Q diag(D) Q^T.

    Args:
        D (Tensor): (..., n) eigenvalues of K
        Q (Tensor): (..., n, n) eigenvectors of K
        RHS (Tensor): (..., n, n) right hand side

    Returns:
        Tensor: (..., n, n) solution
    """
    n = D.shape[-1]
    if D.dim() > 1:
        bs = D.shape[:-1]
        squeeze = False
    else:
        bs = (1,)
        squeeze = True
    bs_rhs = RHS.shape[:-2]
    newRHS = Q.transpose(-2, -1) @ RHS @ Q
    # all eigenvalues of K (+) K
    eigs = (D.unsqueeze(-1) + D.unsqueeze(-2)).reshape(*bs, -1)
    vecX = newRHS.transpose(-2, -1).reshape(*bs_rhs, -1) / eigs
    X = Q @ vecX.reshape(*bs_rhs, n, n).transpose(-2, -1) @ Q.transpose(-2, -1)
    if squeeze:
        X = X.squeeze()
    return X


def solve_lyapunov_diag(K_diag: Tensor, RHS_diag: Tensor) -> Tensor:
    """
    Solves X K^T + K X^T = RHS when K and RHS are diagonal, in O(n).

    Args:
        K_diag (Tensor): (..., n) diagonal of K
        RHS_diag (Tensor): (..., n) diagonal of RHS

    Returns:
        Tensor: (..., n) diagonal of the solution
    """
    if RHS_diag.dim() >= 3:
        pass  # batch of samples, as in SVI; broadcasts against K_diag, any number of leading dims
    elif K_diag.shape[0] != RHS_diag.shape[0]:
        RHS_diag = RHS_diag.unsqueeze(1)
    return 0.5 * RHS_diag / K_diag


# ------------------------------------------------------------------------------
# Matrix exponential of skew-symmetric matrices, with an analytic gradient
# ------------------------------------------------------------------------------


def skew_sym_duplication_matrix(n: int) -> Tensor:
    """Duplication matrix of an (n, n) skew-symmetric matrix: Dn @ v(A) = vec(A)."""
    D = torch.zeros(n * (n - 1) // 2, n**2)
    k = 0
    for j in range(n):
        for i in range(n):
            if i > j:
                u = torch.zeros(n * (n - 1) // 2)
                u[k] = 1.0
                T = torch.zeros(n, n)
                T[i, j] = 1.0
                T[j, i] = -1.0
                D += u.unsqueeze(-1) * T.t().ravel().unsqueeze(0)
                k += 1
    return D.t()


def grad_skew_expm(V, exp_D, V_trns_conj, D, triu_indices, d_indices, dupl_mat):
    # https://www.janmagnus.nl/wips/expo-23.pdf
    d = len(d_indices)
    Delta = torch.zeros_like(V)
    diff = (exp_D[..., triu_indices[0]] - exp_D[..., triu_indices[1]]) / (
        D[..., triu_indices[0]] - D[..., triu_indices[1]]
    )
    Delta[..., triu_indices[0], triu_indices[1]] = diff
    Delta[..., triu_indices[1], triu_indices[0]] = diff
    Delta[..., d_indices, d_indices] = exp_D
    if V.dim() == 2:
        R = torch.kron(V.conj(), V)
        R_inv_dupl = torch.kron(V.transpose(-2, -1), V_trns_conj) @ dupl_mat
    elif V.dim() == 3:
        # batch-wise Kronecker product
        R = torch.einsum("ijk,ilm->ijlkm", V.conj(), V).view(-1, d**2, d**2)
        R_inv_dupl = (
            torch.einsum("ijk,ilm->ijlkm", V.transpose(-2, -1), V_trns_conj).view(-1, d**2, d**2)
            @ dupl_mat
        )
    else:
        raise ValueError("S must be 2 or 3 dimensional")
    grad_exp_S = R * Delta.flatten(start_dim=-2, end_dim=-1).unsqueeze(-2) @ (R_inv_dupl)
    return grad_exp_S.real


class FastSkewSymMatrixExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, S_vec, triu_indices, d_indicies, dupl_mat):
        d = len(d_indicies)
        S = torch.zeros((*S_vec.shape[:-1], d, d), dtype=S_vec.dtype, device=S_vec.device)
        S[..., triu_indices[0], triu_indices[1]] = -S_vec
        S[..., triu_indices[1], triu_indices[0]] = S_vec
        D, V = torch.linalg.eigh(1.0j * S)  # j * S is Hermitian
        D = D / 1.0j
        exp_D = torch.exp(D)
        V_trns_conj = V.transpose(-2, -1).conj()
        ctx.save_for_backward(V, exp_D, V_trns_conj, D, triu_indices, d_indicies, dupl_mat)
        return (V * exp_D.unsqueeze(-2) @ V_trns_conj).real

    @staticmethod
    def backward(ctx, gradoutputs):
        bs = gradoutputs.shape[:-2]
        grad = (
            gradoutputs.transpose(-2, -1).reshape(*bs, 1, -1) @ grad_skew_expm(*ctx.saved_tensors)
        ).squeeze(-2)
        return (grad, None, None, None, None)


fast_skew_sym_matrix_exp = FastSkewSymMatrixExp.apply


class SkewSymMatrixExp(nn.Module):
    """
    Matrix exponential of a skew-symmetric matrix and its gradient; about 10x-100x faster than autograd.

    Args:
        d (int): matrix size
    """

    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.register_buffer("dupl_mat", skew_sym_duplication_matrix(d) + 0j)
        self.register_buffer("ind", torch.triu_indices(d, d, offset=+1))
        self.register_buffer("d_ind", torch.arange(d))

    def forward(self, S_vec: Tensor, return_grad: bool = False):
        """
        Args:
            S_vec (Tensor): (..., d (d - 1) / 2) entries of the skew-symmetric matrices
            return_grad (bool): also return the Jacobian of exp(S) with respect to S_vec

        Returns:
            exp(S), or (exp(S), d exp(S) / d S_vec)
        """
        exp_S = fast_skew_sym_matrix_exp(S_vec, self.ind, self.d_ind, self.dupl_mat)
        if not return_grad:
            return exp_S
        grad_exp_S = grad_skew_expm(*exp_S.grad_fn.saved_tensors)
        return exp_S, grad_exp_S
