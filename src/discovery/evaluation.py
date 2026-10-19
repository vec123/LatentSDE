"""
Results of equation discovery: the posterior-mean drift coefficients, the equations, and their errors against known
coefficients.

drift_coefficients follows SVISE, reference/svise/experiments/4_symbolic_goveq_discovery/post_process.py (GPL-3.0).
"""
from typing import Dict, List, Sequence

import torch
from torch import Tensor

from .learner import SparsePolynomialSDE


def prune(sde: SparsePolynomialSDE) -> None:
    """Removes the terms that the sparse prior switches off (once, after training) and puts sde into eval mode."""
    sde.sde_prior.update_sparse_index()
    sde.eval()


def drift_coefficients(sde: SparsePolynomialSDE, n_samples: int = 1024, seed: int = 22) -> Tensor:
    """(d, num_features) posterior-mean coefficients in the raw feature basis; call prune first."""
    torch.manual_seed(seed)
    with torch.no_grad():
        sde.resample_sde_params(n_samples)
        return sde.sde_prior._tfm(sde.sde_prior.W.mean(0))


def equations(sde: SparsePolynomialSDE, states: Sequence[str]) -> List[str]:
    """One line "d<state>/dt = ..." for each state, from the feature names after update_sparse_index."""
    return [f"d{s}/dt = {e}" for s, e in zip(states, sde.sde_prior.feature_names)]


def coefficient_errors(W: Tensor, W_true: Tensor) -> Dict[str, float]:
    """Relative error of the coefficients (rer) and the number of terms with the wrong support (mismatched)."""
    return {"rer": ((W - W_true).norm() / W_true.norm()).item(),
            "mismatched": int(((W != 0) != (W_true != 0)).sum().item())}
