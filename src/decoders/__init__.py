"""Decoders of the encoder-decoder latent SDE (src/latent): p(x | z, mu). field.py's ResidualUNetDecoder and
PCADecoder match encoders/field.py's frame encoders, for gridded data; a future modality-specific decoder goes
in its own module here, with its log-prob summed into the same ELBO."""
from .base import Decoder
from .field import PCADecoder, ResidualUNetDecoder
from .gaussian import GaussianDecoder

__all__ = ["Decoder", "GaussianDecoder", "PCADecoder", "ResidualUNetDecoder"]
