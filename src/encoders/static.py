"""
The static latent of a patient, w ~ N(0, I) a priori: q(w) = N(m_w, diag v_w) from the masked mean over the observed
steps of the per-step features of the encoder (src/encoders/multimodal.py). w holds what does not change over the
visits (the subpopulation, the rates of the patient); src/latent/model.py appends it to mu for the drift, the
diffusion and the decoder.
"""
import math
from typing import Sequence, Tuple

import torch
from torch import Tensor, nn

from ..core.nn import mlp


class StaticEncoder(nn.Module):
    """
    With n_components > 1 the prior of w is a mixture sum_k pi_k N(mu_k, diag s_k^2), all learned (the means start
    spread out, scale 2): a subpopulation is a component, and assign(m_w) gives the most likely one.
    """

    def __init__(self, dim_feat: int, dim_w: int, hidden: Sequence[int] = (64,), n_components: int = 1,
                 dim_cond: int = 0):
        super().__init__()
        self.net = mlp(dim_feat + dim_cond, 2 * dim_w, tuple(hidden))
        self.dim_w, self.dim_cond, self.n_components = dim_w, dim_cond, n_components
        if dim_cond:  # conditional prior p(w | mu) = N(prior_net(mu), I)
            if n_components > 1:
                raise ValueError("a mixture prior of w cannot also depend on mu")
            self.prior_net = mlp(dim_cond, dim_w, tuple(hidden))
        if n_components > 1:
            self.prior_mean = nn.Parameter(2.0 * torch.randn(n_components, dim_w))
            self.prior_logstd = nn.Parameter(torch.zeros(n_components, dim_w))
            self.prior_logits = nn.Parameter(torch.zeros(n_components))

    def component_logp(self, w: Tensor) -> Tensor:
        """log pi_k N(w; mu_k, s_k^2), (..., K), for a mixture prior."""
        z = (w.unsqueeze(-2) - self.prior_mean) / self.prior_logstd.exp()
        return (torch.log_softmax(self.prior_logits, 0) - 0.5 * z.pow(2).sum(-1) - self.prior_logstd.sum(-1)
                - 0.5 * w.shape[-1] * math.log(2 * math.pi))

    def log_prior(self, w: Tensor) -> Tensor:
        return torch.logsumexp(self.component_logp(w), -1)

    def assign(self, m_w: Tensor) -> Tensor:
        """Most likely mixture component of each m_w (B,)."""
        return self.component_logp(m_w).argmax(-1)

    def prior_mean(self, mu: Tensor) -> Tensor:
        """Mean of p(w | mu) (B, dim_w): 0 without conditioning."""
        return self.prior_net(mu) if getattr(self, "dim_cond", 0) else mu.new_zeros(mu.shape[0], self.dim_w)

    def forward(self, feat: Tensor, step_mask: Tensor, mu: Tensor = None) -> Tuple[Tensor, Tensor]:
        """feat (B, N, F), step_mask (B, N) with 1 at observed steps, mu (B, dim_cond) -> m_w, log v_w (B, dim_w)."""
        w = step_mask.unsqueeze(-1).to(feat.dtype)
        mean = (feat * w).sum(1) / w.sum(1).clamp_min(1.0)
        if getattr(self, "dim_cond", 0):
            mean = torch.cat([mean, mu], -1)
        m, logv = self.net(mean).chunk(2, -1)
        return m, logv.clamp(-8.0, 4.0)
