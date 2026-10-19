"""
Encoder-decoder latent SDE with parameters mu and forcings u(t), and its training loop (ROM-VISDE and SDE
matching). The posteriors live in src/encoders (also home to the optional frame encoder, src/encoders/field.py,
for gridded observations), the decoders in src/decoders; this package holds the model (LatentSDE, DriftNet),
the batch dict (make_batch, interp) and the training loop (fit).
"""
from ..decoders.gaussian import GaussianDecoder
from ..encoders.integrated import IntegratedPosterior
from ..encoders.kernel import KernelPosterior
from ..encoders.sequence import SequencePosterior
from .angles import AngleEmbedding, Embedded
from .batch import interp, make_batch
from .diffusion import (AffineCholeskyDiffusion, ConstantDiagonalDiffusion, Diffusion, NetCholeskyDiffusion,
                        NetDiagonalDiffusion)
from .model import DriftNet, LatentSDE, posterior_drift, posterior_drift_full
from .train import CollapseGuard, ReleaseDiffusion, StopTraining, fit

__all__ = ["AffineCholeskyDiffusion", "AngleEmbedding", "CollapseGuard", "ConstantDiagonalDiffusion", "Diffusion",
           "DriftNet", "Embedded", "GaussianDecoder", "IntegratedPosterior", "KernelPosterior", "LatentSDE", "NetCholeskyDiffusion",
           "NetDiagonalDiffusion", "ReleaseDiffusion", "SequencePosterior", "StopTraining", "fit", "interp",
           "make_batch", "posterior_drift", "posterior_drift_full"]
