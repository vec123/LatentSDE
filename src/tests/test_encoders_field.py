import math

import torch

from ..data.pca import PCA
from ..encoders.field import PCAFrameEncoder, ResidualUNetEncoder
from ..encoders.sequence import SequencePosterior
from ..latent import DriftNet, GaussianDecoder, LatentSDE, fit, make_batch
from ._reference import run_tests, use_float64

use_float64()
SHAPE = (8, 8)  # small, for a fast CNN forward/backward


def _frames(B=3, N=6, shape=SHAPE, seed=0):
    torch.manual_seed(seed)
    return torch.randn(B, N, shape[0] * shape[1])


def test_residual_unet_encoder_shape_and_grad():
    x = _frames()
    encoder = ResidualUNetEncoder(SHAPE, dim_feat=5, channels=(4, 8))
    feat = encoder(x)
    assert feat.shape == (3, 6, 5)
    feat.sum().backward()
    assert all(p.grad is not None for p in encoder.parameters())


def test_pca_frame_encoder_matches_pca_encode_and_is_not_trainable():
    y = torch.randn(200, SHAPE[0] * SHAPE[1])
    pca, _ = PCA.create(y, percent_cutoff=0.999, max_evecs=5)
    encoder = PCAFrameEncoder(pca)
    assert list(encoder.parameters()) == []
    x = _frames()
    torch.testing.assert_close(encoder(x), pca.encode(x))


def test_identity_encoder_matches_no_encoder():
    """LatentSDE(encoder=Identity()) must give the same loss as the default, encoder=None."""
    B, N, D = 3, 6, SHAPE[0] * SHAPE[1]
    batch = make_batch(torch.linspace(0, 1, N).repeat(B, 1), _frames(B, N))

    class Identity(torch.nn.Module):
        def forward(self, x):
            return x

    torch.manual_seed(0)
    model_a = LatentSDE(SequencePosterior(D, dim_z=2, hidden=8), DriftNet(2), GaussianDecoder(2, D), dim_z=2,
                        n_samples=2)
    torch.manual_seed(0)
    model_b = LatentSDE(SequencePosterior(D, dim_z=2, hidden=8), DriftNet(2), GaussianDecoder(2, D), dim_z=2,
                        encoder=Identity(), n_samples=2)

    torch.manual_seed(1)
    loss_a, _ = model_a.loss(batch)
    torch.manual_seed(1)
    loss_b, _ = model_b.loss(batch)
    torch.testing.assert_close(loss_a, loss_b)


def test_latent_sde_trains_with_a_cnn_or_a_pca_encoder():
    B, N, shape = 4, 10, SHAPE
    D = shape[0] * shape[1]
    torch.manual_seed(0)
    x = torch.randn(B, N, D)
    batch = make_batch(torch.linspace(0, 1, N).repeat(B, 1), x)

    dim_feat, dim_z = 4, 2
    pca, _ = PCA.create(x.reshape(-1, D), percent_cutoff=0.999, max_evecs=dim_feat)
    encoders = {"cnn": ResidualUNetEncoder(shape, dim_feat, channels=(4, 8)), "pca": PCAFrameEncoder(pca)}
    for name, frame_encoder in encoders.items():
        posterior = SequencePosterior(dim_feat, dim_z, hidden=8)
        model = LatentSDE(posterior, DriftNet(dim_z), GaussianDecoder(dim_z, D), dim_z=dim_z,
                          encoder=frame_encoder, n_samples=2)
        torch.manual_seed(0)
        history = fit(model, batch, n_iter=10, lr=3e-3, n_warmup=5, log_every=0)
        assert all(math.isfinite(h) for h in history), (name, history)


if __name__ == "__main__":
    raise SystemExit(run_tests(globals()))
