import math

import torch

from ..data.pca import PCA
from ..decoders.field import PCADecoder, ResidualUNetDecoder, downsampled_shape
from ..encoders.field import PCAFrameEncoder, ResidualUNetEncoder
from ..encoders.sequence import SequencePosterior
from ..latent import DriftNet, LatentSDE, fit, make_batch
from ._reference import run_tests, use_float64

use_float64()
SHAPE = (8, 8)


def test_downsampled_shape_matches_the_encoders_actual_output():
    encoder = ResidualUNetEncoder(SHAPE, dim_feat=3, channels=(4, 8, 16))
    with torch.no_grad():
        h = encoder.stem(torch.randn(1, 1, *SHAPE))
        h = encoder.stages[:-1](h)  # drop the last ResidualBlock, which keeps the spatial size
    assert tuple(h.shape[-2:]) == downsampled_shape(SHAPE, len(encoder.stages) // 2)


def test_residual_unet_decoder_shape_grad_and_exact_resize():
    dim_z, dim_mu = 3, 1
    decoder = ResidualUNetDecoder(dim_z, SHAPE, dim_mu, channels=(4, 8, 16))
    z, mu = torch.randn(5, 7, dim_z), torch.randn(5, 7, dim_mu)
    dist = decoder(z, mu)
    assert dist.mean.shape == (5, 7, SHAPE[0] * SHAPE[1])
    (dist.mean.sum() + dist.scale.sum()).backward()
    assert all(p.grad is not None for p in decoder.parameters())


def test_pca_decoder_reduces_to_pca_decode_when_to_code_is_the_identity():
    y = torch.randn(200, SHAPE[0] * SHAPE[1])
    pca, z = PCA.create(y, percent_cutoff=0.999, max_evecs=5)
    rank = z.shape[1]
    decoder = PCADecoder(PCAFrameEncoder(pca), dim_z=rank)
    with torch.no_grad():
        decoder.to_code.weight.copy_(torch.eye(rank))
        decoder.to_code.bias.zero_()
    mu = torch.zeros(z.shape[0], 0)
    torch.testing.assert_close(decoder(z, mu).mean, pca.decode(z))


def test_pca_decoder_decodes_through_the_encoders_exact_pca():
    """PCADecoder must use encoder.pca itself, not an equal-but-different PCA, even one fit the same way."""
    y = torch.randn(200, SHAPE[0] * SHAPE[1])
    pca_a, _ = PCA.create(y, percent_cutoff=0.999, max_evecs=5, random_state=0)
    pca_b, _ = PCA.create(y, percent_cutoff=0.999, max_evecs=5, random_state=1)
    encoder = PCAFrameEncoder(pca_a)
    decoder = PCADecoder(encoder, dim_z=pca_a.evecs.shape[1])
    assert decoder.pca is pca_a and decoder.pca is not pca_b


def test_field_decoders_train_with_their_matching_encoders():
    """ResidualUNetEncoder/Decoder and PCAFrameEncoder/PCADecoder, each pair trained together end to end."""
    B, N, shape = 4, 10, SHAPE
    D = shape[0] * shape[1]
    torch.manual_seed(0)
    x = torch.randn(B, N, D)
    batch = make_batch(torch.linspace(0, 1, N).repeat(B, 1), x)
    dim_feat, dim_z = 4, 2

    # low-rank, so the automatic PCA rank is dim_feat; a PCA on pure noise could pick any rank <= max_evecs
    x_lowrank = torch.randn(B * N, dim_feat) @ torch.randn(dim_feat, D)
    pca, _ = PCA.create(x_lowrank, percent_cutoff=0.999, max_evecs=dim_feat, rescale=False)
    rank = pca.evecs.shape[1]
    pca_encoder = PCAFrameEncoder(pca)
    pairs = {
        "cnn": (dim_feat, ResidualUNetEncoder(shape, dim_feat, channels=(4, 8)),
               ResidualUNetDecoder(dim_z, shape, channels=(4, 8))),
        "pca": (rank, pca_encoder, PCADecoder(pca_encoder, dim_z)),
    }
    for name, (encoder_dim, frame_encoder, decoder) in pairs.items():
        posterior = SequencePosterior(encoder_dim, dim_z, hidden=8)
        model = LatentSDE(posterior, DriftNet(dim_z), decoder, dim_z=dim_z, encoder=frame_encoder, n_samples=2)
        torch.manual_seed(0)
        history = fit(model, batch, n_iter=10, lr=3e-3, n_warmup=5, log_every=0)
        assert all(math.isfinite(h) for h in history), (name, history)


if __name__ == "__main__":
    raise SystemExit(run_tests(globals()))
