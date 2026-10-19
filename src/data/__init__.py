"""Loading trajectories, and PCA as a fixed linear encoder."""
from .pca import PCA
from .reaction_diffusion import load_batches, load_reaction_diffusion, make_batches, window_trajectories
from .trajectories import load_observations, load_trajectories
