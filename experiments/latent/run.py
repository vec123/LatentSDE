"""
LatentSDE runs from YAML configs, on gridded data (test_data/data/reaction_diffusion, h5) or on vectors (a CSV in
the format of src/data/trajectories.py, with states in the data config and encoder type none, e.g.
test_data/data/lotka_volterra_latent). Run from the repository root:

    python -m experiments.latent.run experiments/latent/configs/reaction_diffusion_pca.yaml [more configs ...]
    python -m experiments.latent.run --evaluate runs/<run_dir> [...]    # outputs again from model.pt, no training

A run writes runs/<name>_<YYYYmmdd-HHMMSS>/: config.yaml, log.txt, checkpoints/model_<step>.pt (training state),
model.pt (final model), history.json, metrics.json, and reconstruction_<split>.png (gridded data) or
prior_simulation_<split>.png (vectors) for every split in data.val / data.test. A config can name a base config in the same folder (base: <file>); its own keys override
the base.

Unlike experiments/discovery/run.py, this does not pin torch to one CPU thread: the frame encoders/decoders
here (ResidualUNetEncoder, ResidualUNetDecoder) are compute-bound by their convolutions, not overhead-bound by
small tensors, so more threads (or a GPU) help rather than hurt. Set output.num_threads in the config to pin it
anyway. output.device: cuda trains on the GPU; it needs a torch with CUDA (environments/torch_gpu_env, with h5py).
On simple_multimodal a step is 7 times faster there (21 ms, against 150 ms on 4 CPU threads), as cuDNN fuses the
GRU of the SequencePosterior. The checkpoints stay on the GPU (read them on a CPU with map_location="cpu");
model.pt and the evaluation are on the CPU.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
import yaml
from matplotlib.colors import LinearSegmentedColormap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.core.callbacks import Checkpoint, History, Log
from experiments.latent.e2e import EndToEndModeSDE, ModalityBank, OffsetDecoder
from experiments.latent.hs_mld import MODALITIES as MLD_MODALITIES, load_hs_mld
from src.data.hs import BLOCK_SIZES, TRAIN_COHORTS, load_hs
from src.data.multimodal import load_multimodal, load_multimodal_reaction
from src.data.pca import PCA
from src.data.reaction_diffusion import load_reaction_diffusion, make_batches, window_trajectories
from src.data.trajectories import load_trajectories
from src.decoders.field import PCADecoder, ResidualUNetDecoder
from src.decoders.gaussian import GaussianDecoder, LatentOnlyDecoder, ResidualGaussianDecoder
from src.encoders.field import PCAFrameEncoder, ResidualUNetEncoder
from src.encoders.kernel import KernelPosterior
from src.encoders.multimodal import MultimodalEncoder
from src.encoders.static import StaticEncoder
from src.encoders.sequence import SequencePosterior
from src.kernels.squared_exponential import DeepGaussianKernel
from src.posterior.kalman import GaussianPotentials, KalmanPosterior
from src.posterior.marginal_kalman import MarginalKalmanPosterior
from src.posterior.mixture_kalman import MixtureKalmanPosterior
from src.latent.modes import ModeDecoder, ModeDrift, ModeHead, ModeLatentSDE
from src.latent.transformer import TransformerDrift
from src.latent.switching import SwitchingLatentSDE
from src.latent import (AffineCholeskyDiffusion, AngleEmbedding, CollapseGuard, ConstantDiagonalDiffusion, DriftNet, Embedded,
                        IntegratedPosterior, LatentSDE, NetCholeskyDiffusion, NetDiagonalDiffusion, ReleaseDiffusion,
                        fit)

SURFACE, INK, MUTED = "#fcfcfb", "#222220", "#6b6b66"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
CMAP = LinearSegmentedColormap.from_list(
    "blue_red", [(0.0, "#0b3a75"), (0.25, "#2a78d6"), (0.5, "#f0efec"), (0.75, "#e34948"), (1.0, "#7d1716")])
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": MUTED, "font.size": 10,
})


# ------------------------------------------------------------------ config and data
def load_config(path: str) -> dict:
    config = yaml.safe_load(open(path))
    base = config.pop("base", None)
    if base is None:
        return config
    merged = load_config(os.path.join(os.path.dirname(path), base))
    for key, value in config.items():
        merged[key] = {**merged[key], **value} if isinstance(value, dict) and key in merged else value
    return merged


def load_split(cfg: dict):
    """
    One split (train, val or test) of the data config: its batch dicts, and the frames' spatial shape. A config
    with states reads vectors from a CSV (src/data/trajectories.py), one with pca_dir or modalities multimodal
    observations (src/data/multimodal.py, whole trajectories, no windows); their shape is None.
    """
    if cfg.get("source") == "hs_mld":  # frozen MLD latents of the HS data, experiments/latent/hs_mld.py
        return load_hs_mld(cfg["split"], cfg.get("batch_size", 25), cfg.get("mld_dir", "runs/mld_ae"),
                           cohorts=cfg.get("cohorts", TRAIN_COHORTS), covariates=cfg.get("covariates", "all"),
                           modalities=cfg.get("modalities", MLD_MODALITIES), raw=cfg.get("raw", False))[0], None
    if cfg.get("source") == "hs":  # test_data/data/multimodal, src/data/hs.py
        return load_hs(cfg["split"], cfg.get("batch_size", 25), cohorts=cfg.get("cohorts", TRAIN_COHORTS),
                        covariates=cfg.get("covariates", "basic"))[0], None
    if "pca_dir" in cfg:
        return load_multimodal_reaction(cfg["path"], cfg["pca_dir"], cfg.get("n_groups", 1))[0], None
    if "modalities" in cfg:
        return load_multimodal(cfg["path"], cfg["modalities"], cfg.get("n_groups", 1))[0], None
    if "states" in cfg:
        t, x = load_trajectories(cfg["path"], cfg["states"], time_column=cfg.get("time_column", "t"))
        mu, shape = None, None
    else:
        t, x, mu = load_reaction_diffusion(cfg["path"], cfg.get("field", "u"), cfg.get("param"))
        shape = tuple(x.shape[-2:])
    t_windows, x_windows, mu_windows = window_trajectories(t, x, cfg["length"], cfg.get("stride"), mu)
    batches = make_batches(t_windows, x_windows, cfg.get("n_groups", 1), mu_windows)
    return batches, shape


# ------------------------------------------------------------------ model
def build_frame_encoder(cfg: dict, shape, D: int):
    """(frame_encoder, dim_feat); type none gives (None, D): the posterior reads the observations directly."""
    kind = cfg["type"]
    if kind == "none":
        return None, D
    if kind == "multimodal":  # one network per modality block of src/data/hs.py, the tokens concatenated
        encoder = MultimodalEncoder(BLOCK_SIZES, cfg.get("d_token", 8), tuple(cfg.get("hidden", (32,))))
        return encoder, encoder.dim_feat
    if kind == "pca":
        pca = PCA()
        pca.load_state_dict(torch.load(cfg["model_path"], weights_only=True))
        return PCAFrameEncoder(pca), pca.evecs.shape[1]
    if kind == "cnn":
        dim_feat, channels = cfg["dim_feat"], tuple(cfg.get("channels", (16, 32, 64)))
        return ResidualUNetEncoder(shape, dim_feat, channels), dim_feat
    raise ValueError(f"Unknown encoder type '{kind}', choose 'none', 'pca' or 'cnn'")


def build_decoder(cfg: dict, frame_encoder, dim_z: int, dim_mu: int, shape, D: int):
    kind, std_init = cfg["type"], cfg.get("std_init", 0.1)
    if kind == "pca":
        return PCADecoder(frame_encoder, dim_z, dim_mu, std_init=std_init)
    if kind == "cnn":
        return ResidualUNetDecoder(dim_z, shape, dim_mu, tuple(cfg.get("channels", (16, 32, 64))), std_init)
    if kind == "residual":  # z plus a correction that reads mu (a mode one-hot); needs dim_x == dim_z
        return ResidualGaussianDecoder(dim_z, D, dim_mu, tuple(cfg.get("hidden", (64,))), cfg.get("std_fixed", 0.3),
                                       cfg.get("sizes"))
    if kind == "gaussian":
        std_fixed = cfg.get("std_fixed")
        if isinstance(std_fixed, (int, float)):  # one value for every column
            std_fixed = [float(std_fixed)] * D
        decoder = GaussianDecoder(dim_z, D, dim_mu, tuple(cfg.get("hidden", ())), std_init, std_fixed)
        if cfg.get("init") == "identity":  # each latent dimension starts on one column, mu has no weight
            if cfg.get("hidden"):
                raise ValueError("decoder init 'identity' needs a linear decoder (no hidden layers)")
            with torch.no_grad():
                decoder.mean[0].weight.copy_(torch.eye(D, dim_z + dim_mu))
                decoder.mean[0].bias.zero_()
        if not cfg.get("learn_mean", True):  # the mean map stays at its start (e.g. the identity): fixed units
            decoder.mean.requires_grad_(False)
        return decoder
    raise ValueError(f"Unknown decoder type '{kind}', choose 'pca', 'cnn' or 'gaussian'")


def build_diffusion(cfg: dict, dim_z: int, dim_mu: int, g_init: float = 0.1):
    """None (LatentSDE's own default, ConstantDiagonalDiffusion) unless cfg asks for a state-dependent one, or for
    a constant one with some dimensions fixed (fixed: one value per latent dimension, null for a learned one;
    release_at: N lets them learn from step N on, see train())."""
    kind = cfg.get("type", "constant")
    if kind == "constant":
        return ConstantDiagonalDiffusion(dim_z, g_init, cfg["fixed"]) if cfg.get("fixed") else None
    if kind == "net":
        return NetDiagonalDiffusion(dim_z, dim_mu, tuple(cfg.get("hidden", (64, 64))))
    if kind == "cholesky":
        return NetCholeskyDiffusion(dim_z, dim_mu, tuple(cfg.get("hidden", (64, 64))))
    if kind == "affine":  # lower triangular, affine in z (needs posterior type kalman_marginal or sequence)
        return AffineCholeskyDiffusion(dim_z, g_init)
    raise ValueError(f"Unknown diffusion type '{kind}', choose 'constant', 'net', 'cholesky' or 'affine'")


def build_posterior(cfg: dict, dim_feat: int, dim_z: int, dim_mu: int, t) -> nn.Module:
    """SequencePosterior (default; gauge: true learns the gauge, Gauge Matching; covariance: full gives full-covariance
    marginals N(m, L L^T), diagonal by default; gap_features: true gives its head the bridge time of each input
    column, so v(t) can be small at the observations and larger between them; neighbor_context: true gives it the
    GRU outputs at each column's previous and next observation; forward_only: true makes it a forward posterior, no future step) or KernelPosterior (cfg type:
    kernel), the svise type posterior: needs t (B, N), one window's observation times, evenly spaced, to size its
    DeepGaussianKernel (the window count N - num_window + 1 nodes, and dt); or IntegratedPosterior (cfg type:
    integrated), the integral of an interpolated rate, for sparse, irregular observations; or KalmanPosterior (cfg
    type: kalman), a Kalman smoother of the model's own prior given Gaussian potentials from a recognition network
    (potentials: hidden, init random or identity, s_init decoder or values; extended: true forces the extended
    filter), see src/posterior/kalman.py; or MarginalKalmanPosterior (cfg type: kalman_marginal, the same
    settings and n_passes), the Kalman marginals for any diffusion, see src/posterior/marginal_kalman.py; or
    MixtureKalmanPosterior (cfg type: kalman_mixture, also offsets: K start shifts of the linearization), a mixture
    of such smoothers, see src/posterior/mixture_kalman.py."""
    kind = cfg.get("type", "sequence")
    if kind == "sequence":
        covariance = cfg.get("covariance", "diagonal")
        if covariance not in ("diagonal", "full"):
            raise ValueError(f"Unknown posterior covariance '{covariance}', choose 'diagonal' or 'full'")
        return SequencePosterior(dim_feat, dim_z, dim_mu, cfg.get("hidden", 64), cfg.get("width", 1.0),
                                  cfg.get("bidirectional", True), cfg.get("gauge", False), cfg.get("learn_width", True),
                                  covariance == "full", cfg.get("gap_features", False),
                                  cfg.get("neighbor_context", False), cfg.get("forward_only", False))
    if kind == "kernel":
        num_window = cfg.get("num_window", 5)
        dt = (t[0, 1] - t[0, 0]).item()
        kernel = DeepGaussianKernel(batch_size=t.shape[1] - num_window + 1, dt=dt,
                                    len_init=cfg.get("len_init", 0.1), var_init=cfg.get("var_init", 1e-2))
        return KernelPosterior(dim_feat, dim_z, num_window, kernel, dim_mu, tuple(cfg.get("hidden", (128, 128))))
    if kind == "integrated":
        return IntegratedPosterior(dim_feat, dim_z, dim_mu, cfg.get("hidden", 64), cfg.get("bidirectional", True))
    if kind in ("kalman", "kalman_marginal", "kalman_mixture"):  # potentials: decoder (exact, linear only) or a net
        pot = cfg.get("potentials", {})
        potentials = None if pot == "decoder" else GaussianPotentials(
            dim_feat, dim_z, dim_mu, tuple(pot.get("hidden", (64, 64))), pot.get("init", "random"), pot.get("s_init"))
        if kind == "kalman_mixture":  # K smoothers, first linearized at m0 + offsets[k]; q is their mixture
            return MixtureKalmanPosterior(dim_z, cfg["offsets"], potentials, cfg.get("n_passes", 3), cfg.get("extended"),
                                          dtype=cfg.get("dtype", "float64"))
        if kind == "kalman_marginal":  # any diffusion: statistical linearization, q with the prior's own G(z)
            return MarginalKalmanPosterior(dim_z, potentials, cfg.get("n_passes", 2), cfg.get("extended"),
                                           dtype=cfg.get("dtype", "float64"))
        return KalmanPosterior(dim_z, potentials, cfg.get("extended"), dtype=cfg.get("dtype", "float64"))
    raise ValueError(f"Unknown posterior type '{kind}', choose 'sequence', 'kernel', 'integrated', 'kalman', "
                     "'kalman_marginal' or 'kalman_mixture'")


def build_model(cfg: dict, shape, D: int, t) -> LatentSDE:
    # n_modes: one dynamical mode per subpopulation, its one-hot appended to mu (src/latent/switching.py)
    n_modes = cfg.get("n_modes", 0)
    dim_z, dim_mu, dim_u = cfg["dim_z"], cfg.get("dim_mu", 0) + n_modes, cfg.get("dim_u", 0)
    frame_encoder, dim_feat = build_frame_encoder(cfg["encoder"], shape, D)
    posterior = build_posterior(cfg.get("posterior", {}), dim_feat, dim_z, dim_mu, t)  # reads mu, not the static w
    # dim_w: a static latent w per trajectory (src/encoders/static.py), appended to mu for drift, diffusion, decoder
    dim_w = cfg.get("dim_w", 0)
    static = StaticEncoder(dim_feat, dim_w, tuple(cfg.get("static_hidden", (64,))),
                           cfg.get("w_components", 1), dim_mu) if dim_w else None
    dim_mu += dim_w
    # angles: latent dimensions the drift and the decoder see as cos and sin (src/latent/angles.py)
    embed = AngleEmbedding(dim_z, cfg["angles"]) if cfg.get("angles") else None
    dim_in = embed.dim_out if embed else dim_z
    drift_cfg = cfg.get("drift", {})
    if drift_cfg.get("type") == "transformer":  # the latent dimensions attend to each other (src/latent/transformer.py)
        drift = TransformerDrift(dim_z, dim_mu, dim_u, drift_cfg.get("d_model", 64), drift_cfg.get("n_layers", 2),
                                 drift_cfg.get("n_heads", 4))
    else:
        drift = DriftNet(dim_z, dim_mu, dim_u=dim_u, hidden=tuple(drift_cfg.get("hidden", (64, 64))), dim_in=dim_in)
    # mode_head: {n: K}: a discrete mode c predicted from mu and the posterior path, read by the decoder alone
    # (src/latent/modes.py); the drift and the posterior read the covariates only
    mode_cfg = cfg.get("mode_head")
    if mode_cfg:
        K, P = mode_cfg["n"], dim_mu
        decoder = ModeDecoder(build_decoder(cfg["decoder"], frame_encoder, dim_in, K, shape, D), K)
        drift = ModeDrift(drift, P)
        diffusion = build_diffusion(cfg.get("diffusion", {}), dim_z, P, cfg.get("g_init", 0.1))
        head = ModeHead(P, dim_z, K, tuple(mode_cfg.get("hidden", (64,))))
        if cfg.get("e2e"):  # the modality autoencoders trained together with the latent SDE (experiments/latent/e2e.py)
            bank = ModalityBank(cfg["e2e"].get("mld_dir", "runs/mld_ae"), cfg["e2e"].get("modalities", MLD_MODALITIES))
            decoder = ModeDecoder(OffsetDecoder(dim_z, K, cfg["e2e"].get("std", 0.3)), K)
            return EndToEndModeSDE(bank, posterior, drift, decoder, dim_z, encoder=frame_encoder, diffusion=diffusion,
                                   g_init=cfg.get("g_init", 0.1), n_samples=cfg.get("n_samples", 8),
                                   n_quad=cfg.get("n_quad", 32), closed_form=False, modes=head,
                                   balance=mode_cfg.get("balance", 0.0))
        return ModeLatentSDE(posterior, drift, decoder, dim_z, encoder=frame_encoder, diffusion=diffusion,
                             g_init=cfg.get("g_init", 0.1), n_samples=cfg.get("n_samples", 8),
                             n_quad=cfg.get("n_quad", 32), closed_form=False, modes=head,
                             balance=mode_cfg.get("balance", 0.0))
    # decoder: reads_mu false: p(x | z) only, the static parameters go to the drift and the posterior
    reads_mu = cfg["decoder"].get("reads_mu", True)
    decoder = build_decoder(cfg["decoder"], frame_encoder, dim_in, dim_mu if reads_mu else 0, shape, D)
    if not reads_mu:
        decoder = LatentOnlyDecoder(decoder)
    diffusion = build_diffusion(cfg.get("diffusion", {}), dim_z, dim_mu, cfg.get("g_init", 0.1))
    if embed:
        if diffusion is not None:
            raise ValueError("angles need the constant diffusion; a state-dependent one would read the raw angle")
        drift, decoder = Embedded(embed, drift), Embedded(embed, decoder)
    # closed_form: false keeps the Monte Carlo ELBO where a closed form exists (kalman posterior, linear drift)
    model = LatentSDE(posterior, drift, decoder, dim_z, encoder=frame_encoder, diffusion=diffusion,
                      g_init=cfg.get("g_init", 0.1), n_samples=cfg.get("n_samples", 8), n_quad=cfg.get("n_quad", 32),
                      closed_form=cfg.get("closed_form", True), static=static)
    if cfg.get("input_dropout"):  # modality dropout of the posterior's input: p, and the block widths (sizes)
        model.input_dropout = {"p": cfg["input_dropout"]["p"], "sizes": list(cfg["input_dropout"]["sizes"])}
    return SwitchingLatentSDE(model, n_modes, cfg.get("learn_pi", False)) if n_modes else model


def _setup(num_threads=None):
    if num_threads:
        torch.set_num_threads(num_threads)


def _to_cpu(model: LatentSDE) -> LatentSDE:
    """The model on the CPU, also the tensors the posterior keeps from its last condition() call (plain
    attributes, e.g. SequencePosterior.h, which Module.cpu does not move)."""
    model.cpu()
    for module in model.modules():
        for name, value in list(vars(module).items()):
            if isinstance(value, torch.Tensor):
                setattr(module, name, value.cpu())
    return model


# ------------------------------------------------------------------ reconstruction (no training)
def reconstruct(model: LatentSDE, batch: dict) -> torch.Tensor:
    """Mean posterior z(t) at the batch's own times, decoded; a point estimate, no reparameterized sampling."""
    model.eval()
    with torch.no_grad():
        if hasattr(model, "conditioned"):  # a switching model: the batch with its most likely mode in mu
            batch = model.conditioned(batch)
        mask = batch["mask"]
        x = torch.where(mask > 0, batch["x"], torch.zeros_like(batch["x"]))
        if model.encoder is None:
            feat, feat_mask = x, mask
        else:
            feat = model.encoder(x, mask) if getattr(model.encoder, "uses_mask", False) else model.encoder(x)
            feat_mask = mask.any(-1, keepdim=True).expand_as(feat).to(feat.dtype)
        covered = model.posterior.condition({**batch, "x": feat, "mask": feat_mask})
        m, _, _, _ = model.posterior(batch["t"])
        mu = batch["mu"]
        if model.static is not None:  # the posterior mean of w
            mu = torch.cat([mu, model.static(feat, covered, mu)[0]], -1)
        if getattr(model, "modes", None) is not None:  # the decoder reads the most likely mode
            mu = torch.cat([mu, torch.nn.functional.one_hot(model.assign(batch), model.modes.n_modes).to(mu.dtype)], -1)
        return model.decoder(m, mu[:, None].expand(-1, batch["t"].shape[1], -1)).mean


def field_axes(ax, data, title=None):
    im = ax.imshow(data, cmap=CMAP, vmin=-1, vmax=1, origin="lower", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, color=INK, fontsize=9)
    return im


def plot_reconstruction(out: str, originals, reconstructions, titles, name: str, error: float):
    n = len(titles)
    side, gap, left, bottom, row_gap, top = 1.6, 0.25, 0.5, 0.4, 0.35, 0.4
    W, H = left + n * (side + gap) + 0.7, top + bottom + 2 * side + row_gap + 0.6
    fig = plt.figure(figsize=(W, H))
    fig.suptitle(f"{name}, relative reconstruction RMS error {error:.4f}", color=INK, fontsize=11,
                y=(H - top / 2) / H)
    for i, (orig, recon, title) in enumerate(zip(originals, reconstructions, titles)):
        x0 = (left + i * (side + gap)) / W
        ax_top = fig.add_axes([x0, (bottom + side + row_gap) / H, side / W, side / H])
        ax_bot = fig.add_axes([x0, bottom / H, side / W, side / H])
        field_axes(ax_top, orig, title=title)
        im = field_axes(ax_bot, recon)
        if i == 0:
            ax_top.set_ylabel("original", color=MUTED, fontsize=9)
            ax_bot.set_ylabel("reconstruction", color=MUTED, fontsize=9)
    cax = fig.add_axes([(left + n * (side + gap)) / W, bottom / H, 0.1 / W, (2 * side + row_gap) / H])
    cb = fig.colorbar(im, cax=cax, ticks=[-1, 0, 1])
    cb.outline.set_visible(False)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_vectors(out: str, model: LatentSDE, batch: dict, states, name: str, error: float, n_windows: int,
                 n_paths: int = 10, substeps: int = 25):
    """Per window: the observations, the posterior-mean reconstruction, and prior simulations from the window's own
    m(t0), with no observations after t0."""
    from .simulate_prior import encode_z0, simulate
    recon = reconstruct(model, batch)
    idx = torch.linspace(0, batch["t"].shape[0] - 1, min(n_windows, batch["t"].shape[0])).round().long().tolist()
    fig, axes = plt.subplots(1, len(idx), figsize=(5 * len(idx), 3.6), squeeze=False)
    torch.manual_seed(0)
    for ax, i in zip(axes[0], idx):
        t, x, mu = batch["t"][i], batch["x"][i], batch["mu"][i:i + 1]
        z0 = encode_z0(model, t, x, mu)
        with torch.no_grad():
            sims = [model.decoder(simulate(model, t, mu, substeps, z0), mu.expand(len(t), -1)).mean
                    for _ in range(n_paths)]
        for d, (state, color) in enumerate(zip(states, SERIES)):
            for k, sim in enumerate(sims):
                ax.plot(t, sim[:, d], color=color, lw=0.6, alpha=0.35,
                        label=f"{state} prior from m(t0)" if k == 0 else None)
            ax.scatter(t, x[:, d], s=8, color=color, alpha=0.6, label=f"{state} observed")
            ax.plot(t, recon[i, :, d], color=color, lw=1.5, label=f"{state} posterior mean")
        ax.set(xlabel="t", title=f"{name} [{i}]")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=8)
    fig.suptitle(f"{name}, relative reconstruction RMS error {error:.4f}, {n_paths} prior simulations per window",
                 color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _style_overlay_axes(fig, axes, title):
    for ax in axes:
        ax.legend(fontsize=8, ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14))
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle(title, color=INK, fontsize=11)
    fig.tight_layout()


def plot_overlay(run_dir: str, model: LatentSDE, batch: dict, states, split: str, n_traj: int = 20,
                 substeps: int = 25):
    """n_traj trajectories, each with one prior rollout from its own true m(t0), truth against rollout:
    prior_rollouts_<split>.png (states against t: x1 and x2 together, then each in its own subplot) and, states
    having exactly two dims, prior_phase_<split>.png (the population and log-population phase portraits)."""
    from .simulate_prior import encode_z0, simulate
    idx = torch.linspace(0, batch["t"].shape[0] - 1, min(n_traj, batch["t"].shape[0])).round().long().tolist()
    torch.manual_seed(0)
    ts, truths, rollouts = [], [], []
    for i in idx:
        t, x, mu = batch["t"][i], batch["x"][i], batch["mu"][i:i + 1]
        z0 = encode_z0(model, t, x, mu)
        with torch.no_grad():
            sim = model.decoder(simulate(model, t, mu, substeps, z0), mu.expand(len(t), -1)).mean
        ts.append(t)
        truths.append(x)
        rollouts.append(sim)

    TRUTH, ROLLOUT = "#2a78d6", "#e34948"
    STATE_COLORS = (("#2a78d6", "#e34948"), ("#1baf7a", "#6250d6"))  # (truth, rollout) per state: x1, x2
    suffix = f"{len(idx)} prior rollouts from each trajectory's own true m(t0)"

    fig, axes = plt.subplots(1, 1 + len(states), figsize=(6 * (1 + len(states)), 4.6), squeeze=False)
    ax = axes[0, 0]
    for d, state in enumerate(states):
        truth_color, rollout_color = STATE_COLORS[d % len(STATE_COLORS)]
        for k, (t, x, sim) in enumerate(zip(ts, truths, rollouts)):
            ax.plot(t, x[:, d], color=truth_color, lw=0.8, alpha=0.5, label=f"{state} truth" if k == 0 else None)
            ax.plot(t, sim[:, d], color=rollout_color, lw=0.8, alpha=0.5,
                    label=f"{state} prior rollout" if k == 0 else None)
    ax.set(xlabel="t", ylabel="state", title=f"{len(idx)} trajectories")
    for d, (ax, state) in enumerate(zip(axes[0, 1:], states)):
        for k, (t, x, sim) in enumerate(zip(ts, truths, rollouts)):
            ax.plot(t, x[:, d], color=TRUTH, lw=0.8, alpha=0.5, label="truth" if k == 0 else None)
            ax.plot(t, sim[:, d], color=ROLLOUT, lw=0.8, alpha=0.5, label="prior rollout" if k == 0 else None)
        ax.set(xlabel="t", ylabel=state, title=state)
    _style_overlay_axes(fig, axes[0], f"{split}, {suffix}")
    fig.savefig(os.path.join(run_dir, f"prior_rollouts_{split}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    if len(states) != 2:
        return

    def safe_log(z):
        z = z.clone()
        z[z <= 0] = float("nan")  # gaps, not spikes, where observation noise took a population below 0
        return z.log()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), squeeze=False)
    for ax, transform, title in ((axes[0, 0], lambda z: z, "Populations"),
                                 (axes[0, 1], safe_log, "Log-populations")):
        for k, (x, sim) in enumerate(zip(truths, rollouts)):
            tx, ts_ = transform(x), transform(sim)
            ax.plot(tx[:, 0], tx[:, 1], color=TRUTH, lw=0.7, alpha=0.4, label="truth" if k == 0 else None)
            ax.plot(ts_[:, 0], ts_[:, 1], color=ROLLOUT, lw=0.7, alpha=0.4,
                    label="prior rollout" if k == 0 else None)
        ax.set(xlabel=states[0], ylabel=states[1], title=title)
    _style_overlay_axes(fig, axes[0], f"{split}, {suffix}")
    fig.savefig(os.path.join(run_dir, f"prior_phase_{split}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def evaluate_split(model: LatentSDE, batches, shape, split: str, run_dir: str, n_snapshots: int,
                   states=None, n_paths: int = 10) -> dict:
    """Relative reconstruction RMS error (of the observed values) and negative ELBO over every window of the split,
    and one figure (none for multimodal observations: shape and states both None)."""
    sq_err, sq_mag, neg_elbo, originals, reconstructions = 0.0, 0.0, 0.0, [], []
    torch.manual_seed(0)
    for batch in batches:
        recon = reconstruct(model, batch)
        with torch.no_grad():
            neg_elbo += model.loss(batch)[0].item() / len(batches)
        sq_err += ((recon - batch["x"]).pow(2) * batch["mask"]).sum().item()
        sq_mag += (batch["x"].pow(2) * batch["mask"]).sum().item()
        originals.append(batch["x"])
        reconstructions.append(recon)
    error = (sq_err / sq_mag) ** 0.5
    if shape is None and states is None:
        return {"relative_rms_error": error, "neg_elbo": neg_elbo}
    if shape is None:
        plot_vectors(os.path.join(run_dir, f"prior_simulation_{split}.png"), model, batches[0], states, split, error,
                     n_snapshots, n_paths)
        return {"relative_rms_error": error, "neg_elbo": neg_elbo}
    originals, reconstructions = torch.cat(originals).reshape(-1, *shape), torch.cat(reconstructions).reshape(
        -1, *shape)
    idx = torch.linspace(0, originals.shape[0] - 1, min(n_snapshots, originals.shape[0])).round().long()
    plot_reconstruction(os.path.join(run_dir, f"reconstruction_{split}.png"), originals[idx], reconstructions[idx],
                        [f"{split} [{i}]" for i in idx.tolist()], split, error)
    return {"relative_rms_error": error, "neg_elbo": neg_elbo}


# ------------------------------------------------------------------ training and evaluation of one run
def input_scaling(batches):
    """Mean and std of each column's observed values over the training batches, and the longest span of
    observation times: the standardization of the posterior's inputs (posterior: standardize_inputs: true)."""
    x = torch.cat([b["x"].flatten(0, 1) for b in batches])
    mask = torch.cat([b["mask"].flatten(0, 1) for b in batches])
    n = mask.sum(0).clamp_min(1)
    mean = (x * mask).sum(0) / n
    std = ((x - mean).pow(2) * mask).sum(0).div(n).sqrt().clamp_min(1e-6)
    span = max((b["t"].amax(1) - b["t"].amin(1)).max().item() for b in batches)
    return mean, std, span


def train(run_dir: str) -> None:
    config = load_config(os.path.join(run_dir, "config.yaml"))
    out = config["output"]
    _setup(out.get("num_threads"))
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    device = torch.device(out.get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"output.device is {device}, but torch {torch.__version__} has no CUDA: run with a CUDA "
                           "torch, e.g. environments/torch_gpu_env")

    train_batches, shape = load_split(config["data"]["train"])
    train_batches = [{k: v.to(device) for k, v in batch.items()} for batch in train_batches]
    D = train_batches[0]["x"].shape[-1]
    # train.init_from: continue a saved model.pt; train.step_offset: its steps so far, so the learning rate goes on
    # from its decayed value, with no second warm-up of the KL weight (Adam's moments start again);
    # train.new_posterior: true keeps its prior and decoder but builds the posterior of model.posterior again
    train_cfg = dict(config["train"])
    init_from, offset = train_cfg.pop("init_from", None), train_cfg.pop("step_offset", 0)
    new_posterior = train_cfg.pop("new_posterior", False)
    if init_from:
        model = torch.load(init_from, weights_only=False, map_location=device)
        model.closed_form = config["model"].get("closed_form", True)
        if getattr(model, "modes", None) is not None:  # a resumed mode model takes the config's balance weight
            model.balance = config["model"].get("mode_head", {}).get("balance", 0.0)
        train_cfg["lr"] *= 0.9 ** (offset / train_cfg.get("lr_decay_every", 1000))
        train_cfg["n_warmup"] = 0
        if new_posterior:
            cfg = config["model"]
            model.posterior = build_posterior(cfg.get("posterior", {}), D, cfg["dim_z"], cfg.get("dim_mu", 0),
                                              train_batches[0]["t"]).to(device)
            if hasattr(model.posterior, "attach"):
                model.posterior.attach(model)
    else:
        model = build_model(config["model"], shape, D, train_batches[0]["t"]).to(device)
    if (not init_from or new_posterior) and config["model"].get("posterior", {}).get("standardize_inputs"):
        if model.encoder is not None:
            raise ValueError("standardize_inputs scales the observed columns; with a frame encoder, the "
                             "posterior reads features")
        mean, std, span = input_scaling(train_batches)
        model.posterior.set_input_scaling(mean, std, span)
        print(f"posterior input scaling: shift {mean.tolist()}, scale {std.tolist()}, time scale {span}",
              flush=True)
    data = train_batches[0] if len(train_batches) == 1 else train_batches
    history = History(out["history_every"])
    callbacks = [Checkpoint(run_dir, out["checkpoint_every"]), history, Log(out["log_every"])]
    guard = None
    if config.get("collapse_guard"):  # checks the first training batch, from the end of the KL warm-up by default
        g = config["collapse_guard"]
        first = g.get("start", config["train"].get("n_warmup", 1000))
        guard = CollapseGuard(train_batches[0], g.get("every", 500), first, g.get("ratio", 0.1), g.get("patience", 3))
        callbacks.append(guard)
    release_at = config["model"].get("diffusion", {}).get("release_at")
    if release_at:  # diffusion: fixed [...], release_at: N lets the fixed dimensions learn from step N
        fixed = getattr(model.diffusion, "is_fixed", None)
        if not isinstance(model.diffusion, ConstantDiagonalDiffusion) or fixed is None:
            raise ValueError("diffusion release_at needs a constant diffusion with fixed dimensions (diffusion: fixed)")
        callbacks.append(ReleaseDiffusion(release_at))
    start = time.time()
    fit(model, data, **train_cfg, callbacks=callbacks)
    seconds = time.time() - start
    torch.save(_to_cpu(model), os.path.join(run_dir, "model.pt"))
    json.dump({"step": history.steps, "loss": history.losses}, open(os.path.join(run_dir, "history.json"), "w"))
    extra = {"seconds": seconds, "device": str(device), "final_loss": history.losses[-1]}
    if guard is not None:
        extra["collapse_guard"] = {"stopped_at": guard.stopped_at, "collapsed_columns": guard.collapsed}
    evaluate(run_dir, extra)


def evaluate(run_dir: str, extra: dict = None) -> dict:
    """
    metrics.json and reconstruction_<split>.png or prior_simulation_<split>.png for every configured split, from
    <run_dir>/model.pt. The values already in metrics.json (e.g. seconds, final_loss) stay.
    """
    config = load_config(os.path.join(run_dir, "config.yaml"))
    _setup(config["output"].get("num_threads"))
    model = torch.load(os.path.join(run_dir, "model.pt"), weights_only=False)
    ev = config.get("evaluate", {})
    metrics_path = os.path.join(run_dir, "metrics.json")
    metrics = {**(json.load(open(metrics_path)) if os.path.exists(metrics_path) else {}), **(extra or {})}
    for split in ("train", "val", "test"):
        if split not in config["data"]:
            continue
        cfg = config["data"][split]
        batches, shape = load_split(cfg)
        metrics[split] = evaluate_split(model, batches, shape, split, run_dir, ev.get("n_snapshots", 6),
                                        cfg.get("states"), ev.get("n_paths", 10))
        if shape is None and "states" in cfg:
            # whole trajectories, not the training windows in `batches`: a fair, consistent comparison across splits
            t, x = load_trajectories(cfg["path"], cfg["states"], time_column=cfg.get("time_column", "t"))
            full_batch = {"t": t.unsqueeze(0).expand(x.shape[0], -1), "x": x, "mu": x.new_zeros(x.shape[0], 0)}
            plot_overlay(run_dir, model, full_batch, cfg["states"], split)
    json.dump(metrics, open(metrics_path, "w"), indent=1)
    print(os.path.basename(run_dir), metrics, flush=True)
    return metrics


# ------------------------------------------------------------------ parallel launch
def launch(config_paths, stamp: str, root: str = "runs") -> int:
    """One process per config, all at the same time; each writes its output to <run_dir>/log.txt."""
    runs = []
    for path in config_paths:
        config = load_config(path)
        run_dir = os.path.join(root, f"{config['name']}_{stamp}")
        os.makedirs(run_dir)
        yaml.safe_dump(config, open(os.path.join(run_dir, "config.yaml"), "w"), sort_keys=False)
        log = open(os.path.join(run_dir, "log.txt"), "w")
        process = subprocess.Popen([sys.executable, "-W", "ignore", "-m", "experiments.latent.run",
                                   "--train", run_dir], stdout=log, stderr=subprocess.STDOUT)
        runs.append((run_dir, process))
        print(f"started {run_dir} (pid {process.pid})", flush=True)
    failed = 0
    for run_dir, process in runs:
        code = process.wait()
        failed += code != 0
        print(f"finished {run_dir}: exit code {code}", flush=True)
    return failed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("configs", nargs="*", help="YAML configs, one parallel run each")
    parser.add_argument("--evaluate", nargs="+", metavar="RUN_DIR", help="outputs again from model.pt")
    parser.add_argument("--runs", default="runs", help="folder for the run folders (default: runs)")
    parser.add_argument("--train", metavar="RUN_DIR", help=argparse.SUPPRESS)  # one run, started by launch
    args = parser.parse_args()
    if args.train:
        return train(args.train)
    if args.evaluate:
        for run_dir in args.evaluate:
            evaluate(run_dir)
        return 0
    names = [load_config(path)["name"] for path in args.configs]
    if not names or len(set(names)) < len(names):
        parser.error("give at least one config, and each config needs its own name")
    return launch(args.configs, time.strftime("%Y%m%d-%H%M%S"), args.runs)


if __name__ == "__main__":
    raise SystemExit(main())
