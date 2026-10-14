"""
The approximate posterior of SVISE: a Markov Gaussian process given by its marginal mean and covariance
(diagonal, or spectral S = U diag(D) U^T), sampled without simulation; forward() returns the drift residual
between the posterior drift and a prior drift at a batch of times.

Adapted from SVISE, reference/svise/svise/sde_learning/_marginal_sde.py (GPL-3.0). Changes: the posterior drift
comes from src/posterior/gauges.py, SpectralMarginalSDE takes gauge ("sym", SVISE, or "sqrt"), and the drift
squeezes only its last axis (SVISE squeezed all axes of size 1).
"""
from abc import ABC, abstractmethod
from typing import Callable, Iterator, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter

from ..posterior.gauges import GAUGES, dKdt_spectral, sym_drift_diagonal
from .diffusion import DiagonalDiffusionPrior, DiffusionPrior
from .marginals import MeanFCNN, MeanGLM, OrthogonalFCNN, OrthogonalGLM, StrictlyPositiveFCNN, StrictlyPositiveGLM

__all__ = ["MarginalSDE", "DiagonalMarginalSDE", "SpectralMarginalSDE"]


class MarginalSDE(ABC, nn.Module):
    """
    Abstract base class for a model of the marginal statistics of
    a Markov Gaussian process.
    """

    def __init__(self) -> None:
        super().__init__()
        self.device_var = Parameter(torch.empty(0))
        self.low_rank_cov = False  # bool indicating if covariance is low rank

    @property
    def device(self):
        return self.device_var.device

    @abstractmethod
    def mean_parameters(self) -> Iterator[Parameter]:
        """
        Returns an iterator over the mean parameters of the marginal SDE.

        Returns:
            Iterator[Parameter]: An iterator over the mean parameters of the marginal SDE.
        """
        pass

    @abstractmethod
    def generate_samples(self, t: Tensor, num_samples: int, *args, **kwargs) -> Tensor:
        """
        Generates samples from the approximating SDE marginal distribution at time t and optionally
        returns some intermediate quantities.

        Args:
            t (Tensor): (bs, ) time stamps at which to generate samples
            num_samples (int): number of independent samples at each time stamp

        Returns:
            Tensor: samples of latent states
        """
        pass

    @abstractmethod
    def forward(
        self, t: Tensor, f: Callable[[Tensor, Tensor], Tensor], num_samples: int
    ) -> Tensor:
        """
        Computes the (unweighted) residual loss between the approximating and prior SDEs.

        Args:
            t (Tensor): (bs, ) time stamps at which to generate samples
            f (Callable[[Tensor,Tensor],Tensor]): (t, z) -> (num_samples, bs, d) or (bs, d) prior SDE evaluations
            num_samples (int): number of reparameterized samples at each time stamp

        Returns:
            Tensor: (bs,) approximate residual loss
        """
        pass

    @abstractmethod
    def K(self, t: Tensor) -> Tensor:
        """
        Returns the the covariance matrix K(t) evaluated at a batch of times.

        Args:
            t (Tensor): Time stamps at which to compute the covariance matrix

        Returns:
            Tensor: (len(b), d, d) batch of covariance matrices
        """
        pass

    @abstractmethod
    def mean(self, t: Tensor, return_grad=False) -> Tensor:
        """
        Returns the marginal mean at time t

        Args:
            t (Tensor): (bs,) time stamps at which to evaluate mean

        Returns:
            Tensor: (bs, d) mean evaluated at times
        """
        pass


class DiagonalMarginalSDE(MarginalSDE):
    """
    A model for the marginal statistics of a Markov GP whose
    covariance is diagonal.

    Args:
        d (int): dimension of the state
        t_span (Union[Tuple[float, float], Tuple[Tensor, Tensor]]): time span of data
        diffusion_prior (DiagonalDiffusionPrior): Some diagonal diffusion prior
        model_form (str, optional): GLM or FCNN (FCNN not fully tested)
        vmap (bool, optional): DEPRECATED
        **kwargs: arguments passed onto GLM / FCNN
    """

    def __init__(
        self,
        d: int,
        t_span: Union[Tuple[float, float], Tuple[Tensor, Tensor]],
        diffusion_prior: DiagonalDiffusionPrior,
        model_form: str = "GLM",
        vmap: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.vmap = vmap
        t0 = t_span[0]
        tf = t_span[1]
        self.d = d
        self.diffusion_prior = diffusion_prior
        train_x = kwargs.get("train_x", None)
        train_y = kwargs.get("train_y", None)
        train_dydx = kwargs.get("train_dydx", None)
        if model_form == "GLM":
            kernel = kwargs.get("kernel", "matern52")
            n_tau = kwargs.get("n_tau", 100)
            learn_inducing_locations = kwargs.get("learn_inducing_locations", False)
            self.m = MeanGLM(
                d,
                (t0, tf),
                n_tau,
                learn_inducing_locations=learn_inducing_locations,
                kernel=kernel,
                train_x=train_x,
                train_y=train_y,
                train_dydx=train_dydx,
            )
            kernel = "matern52"
            self.K_diag = StrictlyPositiveGLM(
                d,
                (t0, tf),
                n_tau,
                learn_inducing_locations=learn_inducing_locations,
                kernel=kernel,
            )
        elif model_form == "FCNN":
            hidden_size = kwargs.get("hidden_size", 50)
            num_layers = kwargs.get("num_layers", 2)
            self.m = MeanFCNN(
                d, (t0, tf), hidden_size, num_layers, train_t=train_x, train_x=train_y
            )
            self.K_diag = StrictlyPositiveFCNN(d, (t0, tf), hidden_size, num_layers)
        else:
            raise ValueError(f"Invalid model_form: {model_form}")

    def mean(self, t: Tensor, return_grad=False) -> Tensor:
        """
        Returns the marginal mean at time t

        Args:
            t (Tensor): (bs,) time stamps at which to evaluate mean

        Returns:
            Tensor: (bs, d) mean evaluated at times
        """
        if return_grad:
            return self.m(t, return_grad=True)
        else:
            return self.m(t)

    def mean_parameters(self) -> Iterator[Parameter]:
        """
        Returns an iterator over the mean parameters of the marginal SDE.

        Returns:
            Iterator[Parameter]: An iterator over the mean parameters of the marginal SDE.
        """
        return self.m.parameters()

    def K(self, t: Tensor) -> Tensor:
        """
        Compute the marginal covariance matrix at time t.

        Args:
            t (Tensor): Time stamps at which to compute the covariance matrix

        Returns:
            Tensor: (len(b), d, d) batch of covariance matrices
        """
        return self.K_diag(t).diag_embed()

    def drift(self, t: Tensor, z: Tensor) -> Tensor:
        """
        compute the drift function of the equivalent SDE at times t.

        Args:
            t (Tensor): (bs,) time stamps at which to compute the drift
            z (Tensor): (..., bs, d) batch of latent states

        Returns:
            Tensor: (..., bs, d) batch of drift function evaluations
        """
        # computing parameterization and time derivative
        m, dmdt = self.m(t, return_grad=True)
        K_diag, dKdt_diag = self.K_diag(t, return_grad=True)
        p_noise = self.diffusion_prior.process_noise_diag
        # todo: check if this is correct for mle case
        if dKdt_diag.shape[0] != p_noise.shape[0] and p_noise.dim() == 2:
            p_noise = p_noise.unsqueeze(1)
        return sym_drift_diagonal(z, m, dmdt, K_diag, dKdt_diag, p_noise)

    def unweighted_residual_loss(self, drift: Tensor, f: Tensor) -> Tensor:
        """
        Computes the (unweighted) residual loss between the approximating and prior SDEs.

        Args:
            drift (Tensor): (n_samples, bs, d) drift function evaluations
            f (Tensor): (n_samples, bs, d) / (bs, d) prior SDE evaluations

        Returns:
            Tensor: residual loss
        """
        r = drift - f
        return self.diffusion_prior.solve_linear(r.pow(2)).sum(-1)

    def generate_samples(self, t: Tensor, num_samples: int) -> Tensor:
        """
        Generates samples from the approximating SDE marginal distribution at time t and optionally
        returns some intermediate quantities.

        Args:
            t (Tensor): (bs, ) time stamps at which to generate samples
            num_samples (int): number of independent samples at each time stamp
            return_intermediates (bool, optional): indicates whether to return intermediate quanties. Defaults to False.

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]: samples of latent states or samples of latent states and intermediate quantities
        """

        v = torch.randn(num_samples, len(t), self.d, device=self.device)
        # get mean at each time stamp
        return self.mean(t) + (self.K_diag(t).sqrt().mul(v))

    def forward(
        self, t: Tensor, f: Callable[[Tensor, Tensor], Tensor], num_samples: int
    ) -> Tensor:
        """
        Computes the (unweighted) residual loss between the approximating and prior SDEs.

        Args:
            t (Tensor): (bs, ) time stamps at which to generate samples
            f (Callable[[Tensor,Tensor],Tensor]): (t, z) -> (num_samples, bs, d) or (bs, d) prior SDE evaluations
            num_samples (int): number of reparameterized samples at each time stamp

        Returns:
            Tensor: (bs,) approximate residual loss
        """
        # compute residual loss making good use of intermediate quantities
        zs = self.generate_samples(t, num_samples)
        drift = self.drift(t, zs)
        f_samples = f(t, zs)
        return self.unweighted_residual_loss(drift, f_samples).mean(0)


class SpectralMarginalSDE(MarginalSDE):
    """
    A model for the marginal statistics of a Markov GP using the
    spectral parametrization described in the main text.

    Args:
        d (int): dimension of the state
        t_span (Union[Tuple[float, float], Tuple[Tensor, Tensor]]): time span of data
        diffusion_prior (DiffusionPrior): Some diffusion prior
        model_form (str, optional): GLM or FCNN (FCNN not fully tested)
        vmap (bool, optional): DEPRECATED
        gauge (str, optional): drift for the marginals, "sym" (SVISE) or "sqrt" (SDE matching); see
            src/posterior/gauges.py
        **kwargs: arguments passed onto GLM / FCNN
    """

    def __init__(
        self,
        d: int,
        t_span: Union[Tuple[float, float], Tuple[Tensor, Tensor]],
        diffusion_prior: DiffusionPrior,
        model_form: str = "GLM",
        vmap: bool = False,
        gauge: str = "sym",
        **kwargs,
    ) -> None:
        super().__init__()
        if gauge not in GAUGES:
            raise ValueError(f"Unknown gauge '{gauge}', choose from {sorted(GAUGES)}")
        self.gauge = gauge
        t0 = t_span[0]
        tf = t_span[1]
        self.vmap = vmap
        self.d = d
        self.register_buffer("tril_ind", torch.tril_indices(d, d, offset=-1))
        self.diffusion_prior = diffusion_prior
        self.register_buffer("ident", torch.eye(d))
        train_x = kwargs.get("train_x", None)
        train_y = kwargs.get("train_y", None)
        if model_form == "GLM":
            kernel = kwargs.get("kernel", "matern52")
            n_tau = kwargs.get("n_tau", 100)
            learn_inducing_locations = kwargs.get("learn_inducing_locations", False)
            self.m = MeanGLM(
                d,
                (t0, tf),
                n_tau,
                kernel=kernel,
                learn_inducing_locations=learn_inducing_locations,
                train_x=train_x,
                train_y=train_y,
            )
            self.orthogonal = OrthogonalGLM(
                d,
                (t0, tf),
                n_tau,
                kernel=kernel,
                learn_inducing_locations=learn_inducing_locations,
            )
            self.eigenvals = StrictlyPositiveGLM(
                d,
                (t0, tf),
                n_tau,
                kernel=kernel,
                learn_inducing_locations=learn_inducing_locations,
            )
        elif model_form == "FCNN":
            hidden_size = kwargs.get("hidden_size", 50)
            num_layers = kwargs.get("num_layers", 2)
            # notation for train_t vs train_x is confusing
            self.m = MeanFCNN(
                d, (t0, tf), hidden_size, num_layers, train_t=train_x, train_x=train_y
            )
            self.orthogonal = OrthogonalFCNN(d, (t0, tf), hidden_size, num_layers)
            self.eigenvals = StrictlyPositiveFCNN(d, (t0, tf), hidden_size, num_layers)
        else:
            raise ValueError(f"Invalid model_form: {model_form}")

    def mean(self, t: Tensor, return_grad=False) -> Tensor:
        """
        Returns the marginal mean at time t

        Args:
            t (Tensor): (bs,) time stamps at which to evaluate mean

        Returns:
            Tensor: (bs, d) mean evaluated at times
        """
        if return_grad:
            return self.m(t, return_grad=True)
        else:
            return self.m(t)

    def mean_parameters(self) -> Iterator[Parameter]:
        """
        Returns an iterator over the mean parameters of the marginal SDE.

        Returns:
            Iterator[Parameter]: An iterator over the mean parameters of the marginal SDE.
        """
        return self.m.parameters()

    def K(self, t: Tensor) -> Tensor:
        """
        Compute the marginal covariance matrix at time t.

        Args:
            t (Tensor): Time stamps at which to compute the covariance matrix

        Returns:
            Tensor: (len(b), d, d) batch of covariance matrices
        """
        U = self.orthogonal(t)
        D = self.eigenvals(t)
        return (U.mul(D.unsqueeze(-2))) @ U.transpose(-2, -1)

    def _dKdt_fast(self, U: Tensor, dUdt: Tensor, D: Tensor, dDdt: Tensor) -> Tensor:
        """
        Compute the time derivative of the marginal covariance matrix using intermediate quantities.

        Args:
            U (Tensor): (bs, d, d) batch of orthogonal matrices
            dUdt (Tensor): (bs, d, d) batch of time derivatives of orthogonal matrices
            D (Tensor): (bs,d) batch of eigenvalues of the covariance matrix
            dDdt (Tensor): (bs,d) batch of time derivatives of eigenvalues of the covariance matrix

        Returns:
            Tensor: (bs, d, d) batch of time derivatives of the marginal covariance matrix
        """
        return dKdt_spectral(U, dUdt, D, dDdt)

    def drift(self, t: Tensor, z: Tensor) -> Tensor:
        """
        compute the drift function of the equivalent SDE at times t.

        Args:
            t (Tensor): (bs,) time stamps at which to compute the drift
            z (Tensor): (..., bs, d) batch of latent states

        Returns:
            Tensor: (..., bs, d) batch of drift function evaluations
        """
        # computing parameterization and time derivative
        U, dUdt = self.orthogonal(t, return_grad=True)
        D, dDdt = self.eigenvals(t, return_grad=True)
        return self._drift_fast(t, z, U, dUdt, D, dDdt)

    def _drift_fast(
        self, t: Tensor, z: Tensor, U: Tensor, dUdt: Tensor, D: Tensor, dDdt: Tensor
    ) -> Tensor:
        """
        Compute the drift function of the equivalent SDE at times t using intermediate quantities.

        Args:
            t (Tensor): (bs,) time stamps at which to compute the drift
            z (Tensor): (..., bs, d) batch of latent states
            U (Tensor): (bs, d, d) batch of orthogonal matrices
            dUdt (Tensor): (bs, d, d) batch of time derivatives of orthogonal matrices
            D (Tensor): (bs,d) batch of eigenvalues of the covariance matrix
            dDdt (Tensor): (bs,d) batch of time derivatives of eigenvalues of the covariance matrix

        Returns:
            Tensor: (..., bs, d) batch of drift function evaluations
        """
        p_noise = self.diffusion_prior.process_noise
        if U.shape[0] != p_noise.shape[0] and p_noise.dim() == 3:
            p_noise = p_noise.unsqueeze(1)
        # computing mean and time derivitve of mean
        m, dmdt = self.m(t, return_grad=True)
        return GAUGES[self.gauge](z, m, dmdt, U, dUdt, D, dDdt, p_noise)

    def unweighted_residual_loss(self, drift: Tensor, f: Tensor) -> Tensor:
        """
        Computes the (unweighted) residual loss between the approximating and prior SDEs.

        Args:
            drift (Tensor): (n_samples, bs, d) drift function evaluations
            f (Tensor): (n_samples, bs, d) / (bs, d) prior SDE evaluations

        Returns:
            Tensor: residual loss
        """
        r = drift - f
        return (r.mul(self.diffusion_prior.solve_linear(r))).sum(-1)

    def generate_samples(
        self, t: Tensor, num_samples: int, return_intermediates: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]:
        """
        Generates samples from the approximating SDE marginal distribution at time t and optionally
        returns some intermediate quantities.

        Args:
            t (Tensor): (bs, ) time stamps at which to generate samples
            num_samples (int): number of independent samples at each time stamp
            return_intermediates (bool, optional): indicates whether to return intermediate quanties. Defaults to False.

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]: samples of latent states or samples of latent states and intermediate quantities
        """
        # get covariance at each time stamp
        if return_intermediates:
            U, dUdt = self.orthogonal(t, return_grad=True)
            D, dDdt = self.eigenvals(t, return_grad=True)
        else:
            U = self.orthogonal(t, return_grad=False)
            D = self.eigenvals(t, return_grad=False)
        # (len(t), d, d)
        L = U.mul(D.sqrt().unsqueeze(-2))
        # should I generate samples for at each time stamp, or can they be shared between stamps
        # for now we will generate independent samples at each time stamp
        v = torch.randn(num_samples, len(t), self.d, 1, dtype=t.dtype, device=t.device)
        # get mean at each time stamp
        mu = self.mean(t)
        zs = mu + (L @ v).squeeze(-1)
        if return_intermediates:
            return zs, U, dUdt, D, dDdt
        else:
            return zs

    def forward(
        self, t: Tensor, f: Callable[[Tensor, Tensor], Tensor], num_samples: int
    ) -> Tensor:
        """
        Computes the (unweighted) residual loss between the approximating and prior SDEs.

        Args:
            t (Tensor): (bs, ) time stamps at which to generate samples
            f (Callable[[Tensor,Tensor],Tensor]): (t, z) -> (num_samples, bs, d) or (bs, d) prior SDE evaluations
            num_samples (int): number of reparameterized samples at each time stamp

        Returns:
            Tensor: (bs,) approximate residual loss
        """
        # compute residual loss making good use of intermediate quantities
        zs, U, dUdt, D, dDdt = self.generate_samples(
            t, num_samples, return_intermediates=True
        )
        drift = self._drift_fast(t, zs, U, dUdt, D, dDdt)
        f_samples = f(t, zs)
        return self.unweighted_residual_loss(drift, f_samples).mean(0)
