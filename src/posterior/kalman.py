"""
Structured posterior: a Kalman smoother of the model's own prior SDE, given Gaussian potentials of the observations
in latent space (structured VAE, Johnson et al. 2016). A recognition network (GaussianPotentials) gives a factor
phi(z) ~ p(x_c | z) for each observed value; the smoother combines these factors with the CURRENT drift and
diffusion, so q follows the drift at once. A network that gives the marginals q(z(t)) itself must learn again after
each change of the drift, an EM-like loop that left the drive of x at 0.70 for the true 1 after 60 000 steps
(runs/simple_multimodal_linear_xnoise_dense_x_continued_20261017-203658), and in the trajectories that never observe
x it pulled the drive towards its own poor integral of y1.

Prior: dz = f(z, mu, u) dt + G dW with a ConstantDiagonalDiffusion G, and N(z0_mean, diag exp(z0_logvar)) at the
first step. A linear drift (DriftNet with no hidden layer) gives the exact transitions; any other drift is
linearized at the filter mean at the start of each gap, f(z) ~ F z + b (extended Kalman smoother), and the residual
of the loss then also measures the linearization error. Over a gap tau the transitions come from matrix
exponentials: exp(F tau), the integral of exp(F s) b, and the Van Loan form of the noise covariance.

At a time t between the steps k - 1 and k, the forward filter at k - 1 (the prior for k = 0) is predicted to t, and
the backward message of the steps k, ..., N - 1 (information form Lambda_b, eta_b, so a dimension with no potential
needs no inverse) is propagated back to t; then Sigma = (I + P Lambda_b)^-1 P and m = (I + P Lambda_b)^-1 (m_p +
P eta_b). The posterior drift is the h-transform F z + b + G G^T (eta_b - Lambda_b z) (exact_drift): no gauge and no
time derivatives. With exact potentials and a linear prior, the ELBO is the exact log-likelihood.

Linear drift (Koopman style: linear dynamics in the latent space, nonlinear potentials and decoder): q is the prior
times the potentials, so KL(q || p) of the paths has a closed form from the filter's normalizer (closed_form_terms),
with no quadrature times and no samples; LatentSDE.loss then uses it, and also the closed-form reconstruction of a
linear decoder. For a nonlinear drift, the loss estimates the residual by Monte Carlo, as before.

Without a potentials network, the potentials come from the decoder, which must then be a linear GaussianDecoder
of z only: exact, for checks. Computes in float64 (dtype). Cost: two loops over the N steps of the batch (O(N dim_z^3), N
sequential steps) for the extended filter; for a linear drift, both run as parallel scans
(Saerkkae and Garcia-Fernandez 2021) in 2 log2 N levels. The potentials are local, one network call per step.
"""
import math
from typing import Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn.functional import softplus

from ..core.nn import mlp
from ..decoders.gaussian import GaussianDecoder
from ..latent.diffusion import ConstantDiagonalDiffusion


def _sym(P: Tensor) -> Tensor:
    return 0.5 * (P + P.transpose(-1, -2))


def _mv(M: Tensor, v: Tensor) -> Tensor:
    return (M @ v.unsqueeze(-1)).squeeze(-1)


def _pick(v: Tensor, k: Tensor) -> Tensor:
    """v (B, N, ...) at the step indices k (B, T): (B, T, ...)."""
    B, T = k.shape
    return v.gather(1, k.view(B, T, *[1] * (v.dim() - 2)).expand(B, T, *v.shape[2:]))


def _expm(X: Tensor) -> Tensor:
    """
    exp(X) for a batch of small matrices: scaling and squaring with a Taylor polynomial of degree 8, with s such
    that the largest 1-norm / 2^s is at most 1/4 (error below 1e-11 before the squarings). Only matrix products, so
    its backward pass is cheap; torch.linalg.matrix_exp took half of a training step with its backward pass.
    """
    norm = X.detach().abs().sum(-2).amax(-1).max().item() if X.numel() else 0.0
    s = max(0, math.ceil(math.log2(norm / 0.25))) if norm > 0.25 else 0
    Y = X / 2 ** s
    I = torch.eye(X.shape[-1], dtype=X.dtype, device=X.device)
    E = I + Y / 8
    for k in range(7, 0, -1):
        E = I + Y @ E / k
    for _ in range(s):
        E = E @ E
    return E


def transition(F: Tensor, b: Tensor, GG: Tensor, tau: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Over gaps tau (...), z(t + tau) = A z(t) + c + w, w ~ N(0, Q), for dz = (F z + b) dt + G dW with GG = G G^T;
    F (..., d, d) and b (..., d) broadcast with tau. A = exp(F tau), c = int_0^tau exp(F s) ds b,
    Q = int_0^tau exp(F s) G G^T exp(F^T s) ds. One exponential of [[-F, GG, 0], [0, F^T, 0], [0, b^T, 0]] tau gives
    all three (Van Loan): exp(F^T tau) = A^T and A times the block above it is Q; the last row is c^T. tau = 0 gives
    A = I, c = 0, Q = 0.
    """
    d = F.shape[-1]
    shape = torch.broadcast_shapes(tau.shape, F.shape[:-2], b.shape[:-1])
    M = F.new_zeros(*shape, 2 * d + 1, 2 * d + 1)
    M[..., :d, :d], M[..., :d, d:2 * d], M[..., d:2 * d, d:2 * d] = -F, GG, F.transpose(-1, -2)
    M[..., 2 * d, d:2 * d] = b
    E = _expm(M * tau[..., None, None])
    A = E[..., d:2 * d, d:2 * d].transpose(-1, -2)
    return A, E[..., 2 * d, d:2 * d], _sym(A @ E[..., :d, d:2 * d])


def transition_shared(F: Tensor, b: Tensor, GG: Tensor, tau: Tensor, terms: int = 40):
    """
    transition for one F (d, d) and G for all gaps: the powers of M = [[-F, GG], [0, F^T]] once, and every gap from
    one product of its Taylor coefficients tau^k / k! with them (exp(M tau) gives A^T and the block for Q;
    int_0^tau exp(F s) ds = sum tau^(k+1) / (k+1)! F^k gives c). Exact to about 1e-12 for |M|_1 tau <= 8; longer
    gaps (e.g. evaluation times far after the last step) go to transition.
    """
    d = F.shape[-1]
    M = F.new_zeros(2 * d, 2 * d)
    M[:d, :d], M[:d, d:], M[d:, d:] = -F, GG, F.T
    if tau.numel() and tau.detach().abs().max().item() * M.detach().abs().sum(0).max().item() > 8:
        return transition(F, b, GG, tau)
    powers, step = torch.stack([torch.eye(2 * d, dtype=F.dtype, device=F.device), M]), M
    while len(powers) < terms:  # M^0..M^(2^j - 1), then times M^(2^j): doubling, log2(terms) products
        step = step @ step if len(powers) > 2 else M @ M
        powers = torch.cat([powers, powers @ step])
    upper = powers[:terms, :, d:].reshape(terms, 2 * d * d)  # blocks (1, 2) and (2, 2) of M^k
    k = torch.arange(1, terms + 1, dtype=tau.dtype, device=tau.device)
    ratio = tau.unsqueeze(-1) / k  # tau / 1, tau / 2, ...
    integral = torch.cumprod(ratio, -1)  # tau^(k+1) / (k+1)!
    exp_coef = torch.cat([torch.ones_like(ratio[..., :1]), integral[..., :-1]], -1)  # tau^k / k!
    E = (exp_coef @ upper).view(*tau.shape, 2 * d, d)
    A = E[..., d:, :].transpose(-1, -2)
    Phi1 = (integral @ upper.view(terms, 2 * d, d)[:, d:].reshape(terms, d * d)).view(*tau.shape, d, d).transpose(-1, -2)
    return A, _mv(Phi1, b), _sym(A @ E[..., :d, :])


def _combine(ei, ej):
    """
    a_i (x) a_j of the parallel Kalman filter (Saerkkae and Garcia-Fernandez 2021, lemma 8). An element (A, b, C,
    eta, J) of the steps i..j holds p(z_j | z_{i-1}, y_i..j) = N(A z_{i-1} + b, C) and p(y_i..j | z_{i-1}) ~
    exp(-z^T J z / 2 + eta^T z); all (..., d, d) or (..., d).
    """
    Ai, bi, Ci, hi, Ji = ei
    Aj, bj, Cj, hj, Jj = ej
    d = Ai.shape[-1]
    I = torch.eye(d, dtype=Ai.dtype, device=Ai.device)
    X = torch.linalg.solve(I + Ci @ Jj, torch.cat([Ai, (bi + _mv(Ci, hj)).unsqueeze(-1), Ci], -1))
    Y = torch.linalg.solve(I + Jj @ Ci, torch.cat([(hj - _mv(Jj, bi)).unsqueeze(-1), Jj @ Ai], -1))
    AiT = Ai.transpose(-1, -2)
    return (Aj @ X[..., :d], _mv(Aj, X[..., d]) + bj, _sym(Aj @ X[..., d + 1:] @ Aj.transpose(-1, -2)) + Cj,
            _mv(AiT, Y[..., 0]) + hi, _sym(AiT @ Y[..., 1:]) + Ji)


def _scan_both(fwd, bwd):
    """
    Inclusive scans of _combine along dim 1 in log2 N levels (Hillis-Steele), forward over the elements fwd and in
    reverse over the elements bwd (same shapes), both in one call per level: element k of the first result is
    a_0 (x) ... (x) a_k, of the second b_k (x) ... (x) b_{N-1}.
    """
    (B, N), offset = fwd[0].shape[:2], 1
    while offset < N:
        new = _combine([torch.cat([f[:, :-offset], r[:, :-offset]]) for f, r in zip(fwd, bwd)],
                       [torch.cat([f[:, offset:], r[:, offset:]]) for f, r in zip(fwd, bwd)])
        fwd = [torch.cat([f[:, :offset], c[:B]], 1) for f, c in zip(fwd, new)]
        bwd = [torch.cat([c[B:], r[:, -offset:]], 1) for r, c in zip(bwd, new)]
        offset *= 2
    return fwd, bwd


def _scan_filters(lam, eta, m0, P0, Phi, c, Q):
    """
    Filter means and covariances (B, N, ...) and the backward messages (precision, information) of the steps
    k, ..., N - 1 at each step k, for the potentials lam, eta (B, N, ...), the prior N(m0, P0) at the first step and
    the transitions Phi, c, Q (B, N - 1, ...) between the steps: a forward and a reverse parallel scan.
    """
    B, N, d = eta.shape
    I = torch.eye(d, dtype=eta.dtype, device=eta.device)
    sol = torch.linalg.solve(I + P0 @ lam[:, 0], torch.cat(
        [P0.expand(B, d, d), (m0 + _mv(P0, eta[:, 0])).unsqueeze(-1)], -1))
    zero_v, zero_m = eta.new_zeros(B, 1, d), lam.new_zeros(B, 1, d, d)
    elems = [zero_m, sol[..., d].unsqueeze(1), _sym(sol[..., :d]).unsqueeze(1), zero_v, zero_m]  # the first step
    if N > 1:
        L, h = lam[:, 1:], eta[:, 1:]
        X = torch.linalg.solve(I + Q @ L, torch.cat([Phi, (c + _mv(Q, h)).unsqueeze(-1), Q], -1))
        Y = torch.linalg.solve(I + L @ Q, torch.cat([h.unsqueeze(-1), L], -1))  # (I + L Q)^-1 [eta, lam]
        PhiT = Phi.transpose(-1, -2)
        steps = [X[..., :d], X[..., d], _sym(X[..., d + 1:]), _mv(PhiT, Y[..., 0] - _mv(Y[..., 1:], c)),
                 _sym(PhiT @ Y[..., 1:] @ Phi)]
        elems = [torch.cat([a, b], 1) for a, b in zip(elems, steps)]
    # the reverse scan reads the steps 1..N - 1 and the identity element behind them: its index k holds steps
    # k + 1..N - 1 as a function of z_k (0 at the last step); both scans in one pass
    eye = I.expand(B, 1, d, d)
    ident = [eye, zero_v, zero_m, zero_v, zero_m]
    if N == 1:
        return elems[1], elems[2], lam, eta
    rev = [torch.cat([e[:, 1:], i], 1) for e, i in zip(elems, ident)]
    fwd, bwd = _scan_both(elems, rev)
    return fwd[1], fwd[2], lam + bwd[4], eta + bwd[3]


class GaussianPotentials(nn.Module):
    """
    The recognition network: for each observed value x_c of a step, a Gaussian factor exp(-(w_c^T z - y_c)^2 /
    (2 s_c^2)) in latent space, a scalar pseudo-observation of z. The factors of the observed values multiply (a
    product of experts over the modalities, so a missing value adds nothing): the step's precision is
    sum_c mask_c w_c w_c^T / s_c^2 and its information sum_c mask_c w_c y_c / s_c^2. w_c = W_c + dw_c,
    y_c = x_c + x_scale_c dy_c and log s_c = log S_c + dlog s_c: learned constants and the output of an MLP of the
    step's standardized values, mask and mu, whose last layer starts at zero. The factor reads its own step only,
    not the drift and not the other steps. For a linear decoder x = C z + d + N(0, diag s^2), W = C, dy = -d / x_scale
    and S = s give the exact likelihood.

    init: "random" (W ~ N(0, 1 / dim_z)) or "identity" (W = I, each latent dimension starts on one column, as the
    decoder's init identity); s_init: one start value of S per column, "decoder" (the decoder's std, see
    KalmanPosterior.attach) or None (1).
    """

    def __init__(self, dim_x: int, dim_z: int, dim_mu: int = 0, hidden: Sequence[int] = (64, 64),
                 init: str = "random", s_init=None):
        super().__init__()
        self.dim_x, self.dim_z, self.s_init = dim_x, dim_z, s_init
        if init == "identity":
            W = torch.eye(dim_x, dim_z)
        elif init == "random":
            W = torch.randn(dim_x, dim_z) / dim_z ** 0.5
        else:
            raise ValueError(f"Unknown potentials init '{init}', choose 'random' or 'identity'")
        self.W = nn.Parameter(W)
        s = torch.ones(dim_x) if s_init is None or s_init == "decoder" else torch.as_tensor(s_init, dtype=torch.float)
        self.log_s = nn.Parameter(s.log())
        self.net = mlp(2 * dim_x + dim_mu, dim_x * (dim_z + 2), hidden)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("x_shift", torch.zeros(dim_x))
        self.register_buffer("x_scale", torch.ones(dim_x))

    def set_input_scaling(self, x_shift: Tensor, x_scale: Tensor, t_scale: float = None) -> None:
        """The standardization of the MLP's input (and the unit of dy): x_shift, x_scale (dim_x,)."""
        self.x_shift.copy_(x_shift)
        self.x_scale.copy_(x_scale)

    def forward(self, x: Tensor, mask: Tensor, mu: Tensor) -> Tuple[Tensor, Tensor]:
        """Precision (B, N, dim_z, dim_z) and information (B, N, dim_z) of each step; x, mask (B, N, dim_x)."""
        B, N, D = x.shape
        xs = (x - self.x_shift) / self.x_scale * mask
        out = self.net(torch.cat([xs, mask, mu.unsqueeze(1).expand(B, N, -1)], -1)).view(B, N, D, self.dim_z + 2)
        w = self.W + out[..., :self.dim_z]
        y = x + self.x_scale * out[..., self.dim_z]
        r = mask * torch.exp(-2 * (self.log_s + out[..., self.dim_z + 1]))
        return torch.einsum("bnci,bnc,bncj->bnij", w, r, w), torch.einsum("bnci,bnc->bni", w, r * y)


class KalmanPosterior(nn.Module):
    """
    condition(batch) runs the forward and the backward filter with the attached model's prior; forward(t) gives m,
    L (Sigma = L L^T) and None, None for the time derivatives, which the loss does not need with exact_drift.
    potentials: a GaussianPotentials, or None for the exact potentials of a linear decoder. extended: None picks the
    exact transitions for a linear drift and the extended filter otherwise; True forces the extended filter. dtype:
    the compute dtype, "float64" (default) or "float32" (much faster on a GPU with slow float64).
    """

    def __init__(self, dim_z: int, potentials: nn.Module = None, extended: bool = None, jitter: float = 1e-10,
                 dtype: str = "float64"):
        super().__init__()
        self.dim_z, self.potentials, self.extended, self.jitter = dim_z, potentials, extended, jitter
        self.dtype = dtype

    def _dtype(self) -> torch.dtype:
        return getattr(torch, getattr(self, "dtype", "float64"))

    def attach(self, model: nn.Module) -> None:
        """The LatentSDE whose drift, diffusion, decoder and initial state give the prior; kept outside the module
        tree, so its parameters are not registered twice. Starts the potentials' S at the decoder's std if asked."""
        self.__dict__["_prior_model"] = model
        pot = self.potentials
        if pot is not None and pot.s_init == "decoder":
            std = softplus(getattr(model.decoder, "raw_std", torch.ones(0)))
            if std.shape != pot.log_s.shape:
                raise ValueError("potentials s_init 'decoder' needs a decoder with one std per observed column")
            with torch.no_grad():
                pot.log_s.copy_(std.log())
            pot.s_init = None  # once, not again after a reload

    def set_input_scaling(self, x_shift: Tensor, x_scale: Tensor, t_scale: float) -> None:
        if self.potentials is not None:
            self.potentials.set_input_scaling(x_shift, x_scale, t_scale)

    def _model(self):
        model = self.__dict__.get("_prior_model")
        if model is None:
            raise RuntimeError("KalmanPosterior needs attach(model) first (LatentSDE does it)")
        return model

    def _linear_drift(self):
        """(W_z, W_[mu, u], bias) of a DriftNet with no hidden layer, linear in z, or None."""
        drift, d = self._model().drift, self.dim_z
        layers = getattr(drift, "net", None)
        if self.extended or layers is None or len(layers) != 1:
            return None
        W = layers[0].weight.to(self._dtype())
        return W[:, :d], W[:, d:], layers[0].bias.to(self._dtype())

    def _linearize(self, z: Tensor, mu: Tensor, u: Tensor) -> Tuple[Tensor, Tensor]:
        """F (B, d, d), b (B, d) with f(z') ~ F z' + b near z (B, d), in the compute dtype."""
        drift, dtype = self._model().drift, mu.dtype
        f = lambda zz, mm, uu: drift(zz[None], mm[None], uu[None])[0]
        z32 = z.to(dtype)
        J = torch.func.vmap(torch.func.jacfwd(f))(z32, mu, u)
        b = drift(z32, mu, u) - _mv(J, z32)
        return J.to(self._dtype()), b.to(self._dtype())

    def _decoder_potentials(self, x: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        decoder, d = self._model().decoder, self.dim_z
        if not isinstance(decoder, GaussianDecoder) or len(decoder.mean) != 1 or decoder.mean[0].weight.shape[1] != d:
            raise ValueError("exact potentials need a linear GaussianDecoder of z only (hidden (), no mu)")
        C, dvec = decoder.mean[0].weight.to(x.dtype), decoder.mean[0].bias.to(x.dtype)
        r = mask / softplus(decoder.raw_std).to(x.dtype) ** 2
        return torch.einsum("xi,bnx,xj->bnij", C, r, C), torch.einsum("xi,bnx->bni", C, r * (x - dvec))

    def _sequential_filters(self, lam, eta, m0, P0, GG, gaps, mu, u):
        """The extended filter, step by step: filter means and covariances (B, N, ...), the backward messages of the
        steps k, ..., N - 1 at each step k, and the linearization (F, b) of each interval, (B, N + 1, ...)."""
        B, N, d = eta.shape
        I = torch.eye(d, dtype=eta.dtype, device=eta.device)
        lin = [self._linearize(m0.expand(B, d), mu, u[:, 0])]
        m, P = m0.expand(B, d), P0.expand(B, d, d)
        mf, Pf, trans = [], [], []
        for k in range(N):
            if k:
                lin.append(self._linearize(m, mu, u[:, k - 1]))
                A, c, Q = transition(*lin[-1], GG, gaps[:, k - 1])
                trans.append((A, c, Q))
                m = _mv(A, m) + c
                P = A @ P @ A.transpose(-1, -2) + Q
            sol = torch.linalg.solve(I + P @ lam[:, k], torch.cat([P, (m + _mv(P, eta[:, k])).unsqueeze(-1)], -1))
            m, P = sol[..., d], _sym(sol[..., :d])
            mf.append(m)
            Pf.append(P)
        lin.append(self._linearize(m, mu, u[:, -1]))
        Lb, eb = lam[:, -1], eta[:, -1]
        Lbs, ebs = [Lb], [eb]
        for k in range(N - 2, -1, -1):
            A, c, Q = trans[k]
            X = torch.linalg.solve(I + Lb @ Q, torch.cat([Lb @ A, (eb - _mv(Lb, c)).unsqueeze(-1)], -1))
            Lb = _sym(A.transpose(-1, -2) @ X[..., :d]) + lam[:, k]
            eb = _mv(A.transpose(-1, -2), X[..., d]) + eta[:, k]
            Lbs.append(Lb)
            ebs.append(eb)
        return (torch.stack(mf, 1), torch.stack(Pf, 1), torch.stack(Lbs[::-1], 1), torch.stack(ebs[::-1], 1),
                torch.stack([F for F, _ in lin], 1), torch.stack([b for _, b in lin], 1))

    def condition(self, batch: dict) -> Tensor:
        """Runs the forward and the backward filter over the batch's steps; returns (B, N), True where observed."""
        model, d = self._model(), self.dim_z
        observed = batch["mask"].any(-1) > 0
        if not observed[:, 0].all():
            raise ValueError("KalmanPosterior needs an observation at the first step of each trajectory")
        if not isinstance(model.diffusion, ConstantDiagonalDiffusion):
            raise ValueError("KalmanPosterior needs a constant diagonal diffusion")
        f64 = lambda v: v.to(self._dtype())
        t, x, mask, mu, u = batch["t"], batch["x"], batch["mask"], batch["mu"], batch["u"]
        if self.potentials is None:
            lam, eta = self._decoder_potentials(f64(x), f64(mask))
        else:
            lam, eta = (f64(v) for v in self.potentials(x, mask, mu))
        t = f64(t)
        B, N = t.shape
        z = t.new_zeros(1, d)
        GG = torch.diag_embed(f64(model.diffusion(z, z[:, :0], z.new_zeros(()))[0]) ** 2)
        m0, P0 = f64(model.z0_mean), torch.diag_embed(f64(model.z0_logvar).exp())
        gaps = t[:, 1:] - t[:, :-1]
        linear = self._linear_drift()
        if linear is not None:  # exact transitions, all at once (b of a gap from u at its start); parallel scans
            Wz, Wmu_u, bias = linear
            P = mu.shape[-1]
            b_steps = bias + f64(mu).unsqueeze(1) @ Wmu_u[:, :P].T + f64(u) @ Wmu_u[:, P:].T  # (B, N, d)
            Phi, c, Q = transition_shared(Wz, b_steps[:, :-1], GG, gaps)
            mf, Pf, Lplus, eplus = _scan_filters(lam, eta, m0, P0, Phi, c, Q)
            lin_F = Wz.expand(B, N + 1, d, d)
            lin_b = torch.cat([b_steps[:, :1], b_steps[:, :-1], b_steps[:, -1:]], 1)
            # the prediction at each step (the prior at the first step), for closed_form_terms
            pred_m = torch.cat([m0.expand(B, 1, d), _mv(Phi, mf[:, :-1]) + c], 1)
            pred_P = torch.cat([P0.expand(B, 1, d, d), _sym(Phi @ Pf[:, :-1] @ Phi.transpose(-1, -2)) + Q], 1)
        else:  # extended: each gap linearized at the filter mean at its start, so the filter is sequential
            mf, Pf, Lplus, eplus, lin_F, lin_b = self._sequential_filters(lam, eta, m0, P0, GG, gaps, mu, u)
            pred_m = pred_P = None

        # index k in 0..N: the interval (t_{k-1}, t_k]; the prior at the first step in front, "nothing" behind
        self.lam, self.eta, self.pred_m, self.pred_P = lam, eta, pred_m, pred_P
        self.t, self.GG, self.shared_F = t, GG, Wz if linear is not None else None
        self.lin_F, self.lin_b = lin_F, lin_b
        self.fwd_m = torch.cat([m0.expand(B, 1, d), mf], 1)
        self.fwd_P = torch.cat([P0.expand(B, 1, d, d), Pf], 1)
        self.fwd_t = torch.cat([t[:, :1], t], 1)
        self.bwd_L = torch.cat([Lplus, Lplus.new_zeros(B, 1, d, d)], 1)
        self.bwd_e = torch.cat([eplus, eplus.new_zeros(B, 1, d)], 1)
        return observed

    def forward(self, t: Tensor):
        """m (B, T, dim_z), L (B, T, dim_z, dim_z) and None, None at the times t (B, T); keeps the linearization,
        Lambda_b and eta_b there for exact_drift. Times before the first step count as the first step."""
        dtype, d, steps = t.dtype, self.dim_z, self.t
        tq = torch.maximum(t.to(steps.dtype), steps[:, :1])
        k = torch.searchsorted(steps.contiguous(), tq.contiguous())  # first step at or after t, N after the last
        F, b = _pick(self.lin_F, k), _pick(self.lin_b, k)
        N = steps.shape[1]
        tau_b = torch.where(k < N, steps.gather(1, k.clamp(max=N - 1)) - tq, torch.zeros_like(tq))
        taus = torch.stack([tq - _pick(self.fwd_t, k), tau_b])
        shared = getattr(self, "shared_F", None)
        if shared is not None:
            (A, Ab), (c, cb), (Q, Qb) = transition_shared(shared, b, self.GG, taus)
        else:
            (A, Ab), (c, cb), (Q, Qb) = transition(F, b, self.GG, taus)
        m_p = _mv(A, _pick(self.fwd_m, k)) + c
        P_p = A @ _pick(self.fwd_P, k) @ A.transpose(-1, -2) + Q
        Lp, ep = _pick(self.bwd_L, k), _pick(self.bwd_e, k)
        I = torch.eye(d, dtype=tq.dtype, device=tq.device)
        X = torch.linalg.solve(I + Lp @ Qb, torch.cat([Lp @ Ab, (ep - _mv(Lp, cb)).unsqueeze(-1)], -1))
        Lam_b = _sym(Ab.transpose(-1, -2) @ X[..., :d])
        eta_b = _mv(Ab.transpose(-1, -2), X[..., d])
        sol = torch.linalg.solve(I + P_p @ Lam_b, torch.cat([P_p, (m_p + _mv(P_p, eta_b)).unsqueeze(-1)], -1))
        m, Sigma = sol[..., d], _sym(sol[..., :d])
        L = torch.linalg.cholesky(Sigma + self.jitter * I)
        self.q_F, self.q_b, self.Lam_b, self.eta_b = (v.to(dtype) for v in (F, b, Lam_b, eta_b))
        return m.to(dtype), L.to(dtype), None, None

    def filtered(self, t: Tensor):
        """
        The forward posterior (the filter) at the times t (B, T), after condition: N(m, Sigma) of z(t) given the
        potentials of the steps at or before t only. The updated filter at the last step at or before t, predicted to t
        by the prior (exact matrix exponentials for a linear drift): at a step it is the filter with that step's
        potential, between two steps and after the last one it is the forecast of the model's own SDE. No backward
        message, so no later observation enters; the smoother of forward(t) is used for training (its terms are exact),
        the filter for the state estimates and forecasts that must not see the future. Returns m (B, T, dim_z) and L.
        """
        dtype, d, steps = t.dtype, self.dim_z, self.t
        tq = torch.maximum(t.to(steps.dtype), steps[:, :1])
        k = torch.searchsorted(steps.contiguous(), tq.contiguous(), right=True)  # steps at or before t, at least 1
        F, b = _pick(self.lin_F, k), _pick(self.lin_b, k)
        tau = tq - _pick(self.fwd_t, k)  # from the last step at or before t; fwd_t[k] is the time of step k - 1
        shared = getattr(self, "shared_F", None)
        A, c, Q = transition_shared(shared, b, self.GG, tau) if shared is not None else transition(F, b, self.GG, tau)
        m = _mv(A, _pick(self.fwd_m, k)) + c
        P = A @ _pick(self.fwd_P, k) @ A.transpose(-1, -2) + Q
        I = torch.eye(d, dtype=tq.dtype, device=tq.device)
        return m.to(dtype), torch.linalg.cholesky(_sym(P) + self.jitter * I).to(dtype)

    def closed_form_terms(self):
        """
        After condition, for a linear drift (exact transitions): the smoothed marginals at the steps, m (B, N, d) and
        S (B, N, d, d), and KL(q || p) of the paths in closed form, as the KL at the first step and the KL of the
        rest of the path (B,): the terms that the loss otherwise estimates as kl0 and residual (chain rule of the
        KL). None after the extended filter: for a nonlinear drift, the linearized prior is not the prior.

        q is the prior times the potentials phi_k(z) = exp(-z^T lam_k z / 2 + eta_k^T z) at the steps, over their
        normalizer Z; so KL(q || p) = sum_k E_q log phi_k(z_k) - log Z. log Z is the sum over the steps of
        log int N(z; m_p, P_p) phi_k(z) dz, with the filter's prediction N(m_p, P_p) at the step (the prior at the
        first step). With h = eta - lam m_p, K = I + P_p lam and the smoothed N(m, S), delta = m - m_p, the term of
        a step is h^T delta - delta^T lam delta / 2 - tr(lam S) / 2 - h^T K^-1 P_p h / 2 + log det K / 2 (the large
        parts of the two logs cancel, so this form keeps the precision). A step with no potential gives 0.
        """
        if getattr(self, "pred_m", None) is None:
            return None
        lam, eta, m_p, P_p = self.lam, self.eta, self.pred_m, self.pred_P
        N, d = eta.shape[1], self.dim_z
        I = torch.eye(d, dtype=eta.dtype, device=eta.device)
        # smoothed: the prediction times the potential of the step and the backward message of the later steps
        Lb, eb = self.bwd_L[:, :N], self.bwd_e[:, :N]
        sol = torch.linalg.solve(I + P_p @ Lb, torch.cat([P_p, (m_p + _mv(P_p, eb)).unsqueeze(-1)], -1))
        m, S = sol[..., d], _sym(sol[..., :d])
        h, delta, K = eta - _mv(lam, m_p), m - m_p, I + P_p @ lam
        quad = (h * torch.linalg.solve(K, _mv(P_p, h).unsqueeze(-1)).squeeze(-1)).sum(-1)
        step_kl = ((h - 0.5 * _mv(lam, delta)) * delta).sum(-1) - 0.5 * ((lam * S).sum((-1, -2)) + quad
                                                                       - torch.linalg.slogdet(K)[1])
        model = self._model()
        m0, logv0 = model.z0_mean.to(eta.dtype), model.z0_logvar.to(eta.dtype)
        S0 = S[:, 0]
        logdet0 = 2 * torch.linalg.cholesky(S0 + self.jitter * I).diagonal(dim1=-2, dim2=-1).log().sum(-1)
        kl0 = 0.5 * (logv0.sum() - logdet0 - d
                     + ((S0.diagonal(dim1=-2, dim2=-1) + (m[:, 0] - m0).pow(2)) / logv0.exp()).sum(-1))
        return m, S, kl0, step_kl.sum(1) - kl0

    def exact_drift(self, z: Tensor, steps: slice) -> Tensor:
        """The posterior drift F z + b + G G^T (eta_b - Lambda_b z) at the query times `steps` of the last forward
        call; z (..., B, len(steps), dim_z)."""
        F, b, Lam_b, eta_b = (v[:, steps] for v in (self.q_F, self.q_b, self.Lam_b, self.eta_b))
        return _mv(F, z) + b + (eta_b - _mv(Lam_b, z)) @ self.GG.to(z.dtype)
