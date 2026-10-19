"""
Diffusion of LatentSDE: G(z, mu, t). Point estimate, not a Bayesian prior: discovery/diffusion.py's
DiffusionPrior stays scoped to the no-encoder path (see its docstring); a learned encoder makes the latent units
non-identifiable, so a sparse prior here would pick the latent scale instead of the data.

A diagonal G is stored as its diagonal (..., dim_z). A full G is stored as (..., dim_z, dim_z), lower triangular
with a positive diagonal: the Cholesky factor of Sigma = G G^T. diffuse, cov_mv and whiten apply G, Sigma and G^-1
to a vector for both forms.

The posterior drift needs div(Sigma), (div Sigma)_i = sum_j d Sigma_ij / d z_j (Bartosh et al. 2025, posterior SDE
drift). with_divergence gives G and div(Sigma) together, at O(dim_z^2) cost:

    ConstantDiagonalDiffusion  diagonal, independent of z: div = 0. The default.
    NetDiagonalDiffusion       diagonal, g_k depends on z only through z_k (as in SDE matching).
    NetCholeskyDiffusion       full lower triangular, row i depends on z only through z_i.
    AffineCholeskyDiffusion    full lower triangular, a constant diagonal, below it affine in all of z.

In the two network forms, one forward-mode pass with tangent ones gives the derivative of each entry by the one
coordinate it depends on, so div needs no full Jacobian.
"""
from typing import Optional, Protocol, Sequence, Tuple, runtime_checkable

import torch
from torch import Tensor, nn
from torch.nn.functional import softplus

from ..core.nn import mlp
from ..core.transforms import inverse_softplus_scalar


@runtime_checkable
class Diffusion(Protocol):
    def __call__(self, z: Tensor, mu: Tensor, t: Tensor) -> Tensor:
        """G: diagonal (..., dim_z), or lower triangular (..., dim_z, dim_z) with a positive diagonal."""
        ...

    def with_divergence(self, z: Tensor, mu: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """G, and div(G G^T) (it broadcasts to the shape of z)."""
        ...


def diffuse(G: Tensor, x: Tensor) -> Tensor:
    """G x."""
    return (G @ x.unsqueeze(-1)).squeeze(-1) if G.dim() > x.dim() else G * x


def cov_mv(G: Tensor, x: Tensor) -> Tensor:
    """G G^T x."""
    if G.dim() > x.dim():
        return (G @ (G.transpose(-1, -2) @ x.unsqueeze(-1))).squeeze(-1)
    return G.pow(2) * x


def whiten(G: Tensor, r: Tensor) -> Tensor:
    """G^-1 r."""
    if G.dim() > r.dim():
        return torch.linalg.solve_triangular(G, r.unsqueeze(-1), upper=False).squeeze(-1)
    return r / G


def _per_dimension(z: Tensor, mu: Tensor, t: Tensor, eye: Tensor) -> Tensor:
    """Inputs [z_k, e_k, mu, t] for each dimension k, with e_k the k-th unit vector: (..., dim_z, 2 + dim_z + dim_mu)."""
    t = (t + torch.zeros_like(z[..., 0])).unsqueeze(-1)
    context = torch.cat([mu, t], -1).unsqueeze(-2).expand(*z.shape, -1)
    return torch.cat([z.unsqueeze(-1), eye.expand(*z.shape, -1), context], -1)


class ConstantDiagonalDiffusion(nn.Module):
    """
    g = softplus(raw_g), one per latent dimension, independent of z, mu and t. fixed (dim_z values, None for a
    learned one) fixes g in some dimensions, e.g. a small g for a state with no noise of its own (dx = y dt): with a
    learned g, the posterior can make x a random walk and the prior then learns a large g for it
    (runs/simple_multimodal_fixed_noise_20261017-160921: g of x 1.0 for 0, the drive of x 0.5 for 1).
    """

    def __init__(self, dim_z: int, g_init: float = 0.1, fixed: Sequence[Optional[float]] = None):
        super().__init__()
        self.raw_g = nn.Parameter(torch.full((dim_z,), inverse_softplus_scalar(g_init)))
        if fixed is not None:
            if len(fixed) != dim_z:
                raise ValueError(f"fixed has {len(fixed)} values for {dim_z} latent dimensions")
            self.register_buffer("is_fixed", torch.tensor([g is not None for g in fixed]))
            self.register_buffer("g_fixed", torch.tensor([0.0 if g is None else g for g in fixed]))

    def release(self) -> None:
        """Lets the fixed dimensions learn from now on, starting at their fixed values (g stays continuous)."""
        if getattr(self, "is_fixed", None) is None:
            return
        with torch.no_grad():
            g = self.g_fixed[self.is_fixed].to(self.raw_g.dtype)
            self.raw_g[self.is_fixed] = torch.log(torch.expm1(g))  # inverse softplus
        self.is_fixed.zero_()

    def forward(self, z: Tensor, mu: Tensor, t: Tensor) -> Tensor:
        g = softplus(self.raw_g)
        if getattr(self, "is_fixed", None) is not None:  # runs saved before fixed have no attribute
            g = torch.where(self.is_fixed, self.g_fixed.to(g.dtype), g)
        return g.expand_as(z)

    def with_divergence(self, z: Tensor, mu: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        return self(z, mu, t), z.new_zeros(())


class NetDiagonalDiffusion(nn.Module):
    """g_k(z_k, mu, t) = softplus(MLP([z_k, e_k, mu, t])). All dimensions share one MLP."""

    def __init__(self, dim_z: int, dim_mu: int = 0, hidden: Sequence[int] = (64, 64)):
        super().__init__()
        self.net = mlp(2 + dim_z + dim_mu, 1, hidden)
        self.register_buffer("eye", torch.eye(dim_z), persistent=False)

    def forward(self, z: Tensor, mu: Tensor, t: Tensor) -> Tensor:
        return softplus(self.net(_per_dimension(z, mu, t, self.eye))).squeeze(-1)

    def with_divergence(self, z: Tensor, mu: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        g, dg = torch.func.jvp(lambda z: self(z, mu, t), (z,), (torch.ones_like(z),))
        return g, 2 * g * dg


class NetCholeskyDiffusion(nn.Module):
    """
    G lower triangular with a positive diagonal. Row i is MLP([z_i, e_i, mu, t]) (softplus on the diagonal entry,
    zero above it), and all rows share one MLP. Sigma = G G^T is full and depends on z.

    With G'_ik = dG_ik / dz_i (one forward-mode pass), div(Sigma) = G (G'^T 1) + (G * G') 1.
    """

    def __init__(self, dim_z: int, dim_mu: int = 0, hidden: Sequence[int] = (64, 64)):
        super().__init__()
        self.net = mlp(2 + dim_z + dim_mu, dim_z, hidden)
        self.register_buffer("eye", torch.eye(dim_z), persistent=False)
        self.register_buffer("lower", torch.ones(dim_z, dim_z).tril(-1), persistent=False)

    def forward(self, z: Tensor, mu: Tensor, t: Tensor) -> Tensor:
        rows = self.net(_per_dimension(z, mu, t, self.eye))
        return rows * self.lower + softplus(rows) * self.eye

    def with_divergence(self, z: Tensor, mu: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        G, dG = torch.func.jvp(lambda z: self(z, mu, t), (z,), (torch.ones_like(z),))
        return G, (G @ dG.sum(-2).unsqueeze(-1)).squeeze(-1) + (G * dG).sum(-1)


class AffineCholeskyDiffusion(nn.Module):
    """
    G(z) lower triangular: a constant positive diagonal softplus(raw_g), and below it C0 + sum_j z_j C_j, so a row
    can depend on any state. It holds noise that one state passes to another in proportion to a third, e.g. the exact
    Koopman lift of dx = (y1 + c (y2^2 - s^2)) dt + ...: w = y2^2 - s^2 has the noise 2 a y2 dW_2, the same dW_2 as
    y2 (row w, column y2: slope 2 a on z_y2), and a small diagonal of its own. Independent of mu and t.

    C0 and C start at zero, so G starts as ConstantDiagonalDiffusion(dim_z, g_init). dG/dz_j = C_j (below the
    diagonal), so div(Sigma)_i = sum_j (C_j G^T + G C_j^T)_ij in closed form.
    """

    def __init__(self, dim_z: int, g_init: float = 0.1):
        super().__init__()
        self.raw_g = nn.Parameter(torch.full((dim_z,), inverse_softplus_scalar(g_init)))
        self.C0 = nn.Parameter(torch.zeros(dim_z, dim_z))
        self.C = nn.Parameter(torch.zeros(dim_z, dim_z, dim_z))  # C[j] = dG/dz_j
        self.register_buffer("lower", torch.ones(dim_z, dim_z).tril(-1), persistent=False)

    def forward(self, z: Tensor, mu: Tensor, t: Tensor) -> Tensor:
        below = self.C0 * self.lower + torch.einsum("...j,jil->...il", z, self.C * self.lower)
        return below + torch.diag_embed(softplus(self.raw_g))

    def with_divergence(self, z: Tensor, mu: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        G, slopes = self(z, mu, t), self.C * self.lower
        div = torch.einsum("jil,...jl->...i", slopes, G) + G @ torch.einsum("jjl->l", slopes)
        return G, div
