"""The batch dict of src/latent/model.py, and the batched linear interpolation it uses for the forcing u(t)."""
import torch
from torch import Tensor


def make_batch(t: Tensor, x: Tensor, mask: Tensor = None, mu: Tensor = None, u: Tensor = None) -> dict:
    """Batch with defaults: all values observed, no parameters (P = 0), no forcing (F = 0)."""
    B, N, _ = x.shape
    return {"t": t, "x": x, "mask": torch.ones_like(x) if mask is None else mask,
            "mu": x.new_zeros(B, 0) if mu is None else mu, "u": x.new_zeros(B, N, 0) if u is None else u}


def interp(t_query: Tensor, t: Tensor, y: Tensor) -> Tensor:
    """
    Linear interpolation of y (B, N, F) at sorted times t (B, N), for t_query (B, Q); holds the end values.
    In a padded row, the first copy of the last time is the end.
    """
    t = t.contiguous()
    i = torch.minimum(torch.searchsorted(t, t_query.contiguous()), torch.searchsorted(t, t[:, -1:])).clamp(min=1)
    t0, t1 = t.gather(1, i - 1), t.gather(1, i)
    w = ((t_query - t0) / (t1 - t0).clamp_min(1e-12)).clamp(0, 1).unsqueeze(-1)
    i = i.unsqueeze(-1).expand(-1, -1, y.shape[-1])
    y0, y1 = y.gather(1, i - 1), y.gather(1, i)
    return y0 + w * (y1 - y0)
