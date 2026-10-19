import math

import torch

from ..latent import (AngleEmbedding, DriftNet, Embedded, GaussianDecoder, IntegratedPosterior, LatentSDE, fit,
                      interp, make_batch)
from ..encoders.integrated import integrate_linear  # after src.latent: importing src.encoders first is circular
from ._reference import run_tests, use_float64

use_float64()


def padded_times():
    torch.manual_seed(0)
    t = torch.rand(3, 8).sort(-1).values * 4
    t[2, 5:] = t[2, 4]  # padded row: the last time repeated
    return t


def test_integrate_linear_matches_the_numerical_integral_of_interp():
    """The closed form against a fine trapezoid of the interpolation, also before the first step, after the last,
    and in a padded row; the value is interp's, the slope its finite difference."""
    t = padded_times()
    a = torch.randn(3, 8, 2)
    grid = torch.linspace(-0.5, 4.5, 20001).expand(3, -1).contiguous()
    value = interp(grid, t, a)
    numeric = torch.cat([torch.zeros(3, 1, 2), torch.cumulative_trapezoid(value, grid.unsqueeze(-1), dim=1)], 1)
    numeric = numeric - interp(t[:, :1], grid, numeric)  # from t[:, 0], not from the grid's start
    query = torch.rand(3, 50) * 5 - 0.5
    integral, at, slope = integrate_linear(t, a, query)
    torch.testing.assert_close(integral, interp(query, grid, numeric), atol=1e-6, rtol=0)
    torch.testing.assert_close(at, interp(query, t, a))
    h = 1e-6
    fd = (interp(query + h, t, a) - interp(query - h, t, a)) / (2 * h)
    torch.testing.assert_close(slope, fd, atol=1e-4, rtol=0)


def test_integrated_posterior_mean_derivative_is_its_rate():
    """m starts at m0 at the first step, dm/dt and dlog v/dt are the finite differences of m and log v."""
    t = padded_times()
    batch = make_batch(t, torch.randn(3, 8, 4), (torch.rand(3, 8, 4) > 0.3).double(), torch.randn(3, 1))
    batch["mask"][2, 5:] = 0
    post = IntegratedPosterior(4, 3, dim_mu=1, hidden=16)
    post.condition(batch)
    torch.testing.assert_close(post(t[:, :1])[0][:, 0], post.m0)
    query = torch.rand(3, 30) * 4
    m, logv, dm, dlogv = post(query)
    h = 1e-6
    (mp, lp, _, _), (mm, lm, _, _) = post(query + h), post(query - h)
    torch.testing.assert_close(dm, (mp - mm) / (2 * h), atol=1e-5, rtol=0)
    torch.testing.assert_close(dlogv, (lp - lm) / (2 * h), atol=1e-4, rtol=0)


def test_angle_embedding_makes_drift_and_decoder_periodic():
    torch.manual_seed(0)
    embed = AngleEmbedding(3, [0])
    drift = Embedded(embed, DriftNet(3, 1, hidden=(8,), dim_in=embed.dim_out))
    decoder = Embedded(embed, GaussianDecoder(embed.dim_out, 5, 1, (8,)))
    z, mu, u = torch.randn(4, 3), torch.randn(4, 1), torch.zeros(4, 0)
    turned = z + torch.tensor([2 * math.pi * 3, 0.0, 0.0])
    torch.testing.assert_close(drift(turned, mu, u), drift(z, mu, u))
    torch.testing.assert_close(decoder(turned, mu).mean, decoder(z, mu).mean)
    assert drift(z, mu, u).shape == (4, 3)


def test_latent_sde_trains_with_the_integrated_posterior_and_angles():
    torch.manual_seed(0)
    t = torch.rand(6, 12).sort(-1).values * 10
    x = torch.randn(6, 12, 4)
    batch = make_batch(t, x, (torch.rand(6, 12, 4) > 0.5).double(), torch.randn(6, 1))
    embed = AngleEmbedding(2, [0])
    model = LatentSDE(IntegratedPosterior(4, 2, dim_mu=1, hidden=16),
                      Embedded(embed, DriftNet(2, 1, hidden=(16,), dim_in=embed.dim_out)),
                      Embedded(embed, GaussianDecoder(embed.dim_out, 4, 1, (16,))), 2, n_samples=2, n_quad=8)
    losses = fit(model, batch, n_iter=60, lr=1e-2, n_warmup=0, log_every=0)
    assert all(math.isfinite(v) for v in losses) and sum(losses[-10:]) < sum(losses[:10])


if __name__ == "__main__":
    raise SystemExit(run_tests(globals()))
