"""
Posteriors of the encoder-decoder latent SDE (src/latent): q(z(t)) = N(m(t), diag v(t)), amortized over the
observations of a batch. Two families, see each module's docstring:

    kernel    KernelPosterior, the svise type posterior: a Gaussian process, given by kernel interpolation in
              time (ROM-VISDE).
    sequence  SequencePosterior, the SDE-matching type posterior: a network of state and time, reparameterized
              with Gaussian noise.
    integrated  IntegratedPosterior: the mean is the closed-form integral of a rate interpolated between the
              observation times, for sparse, irregular observations of a fast latent state.

field.py's ResidualUNetEncoder (trainable) and PCAFrameEncoder (fixed) are LatentSDE's optional encoder, for
observations that are images (e.g. test_data/data/reaction_diffusion): run once on the raw frame, in front of
whichever posterior above is doing the temporal modelling. The encoder and the posterior are independent,
separately swappable arguments of LatentSDE, not one combined class.
"""
from .base import FrameEncoder, Posterior
from .field import PCAFrameEncoder, ResidualUNetEncoder
from .integrated import IntegratedPosterior
from .kernel import KernelPosterior
from .sequence import SequencePosterior

__all__ = ["FrameEncoder", "IntegratedPosterior", "KernelPosterior", "PCAFrameEncoder", "Posterior",
           "ResidualUNetEncoder", "SequencePosterior"]
