"""
Training loop of LatentSDE: Adam with a decaying learning rate and a beta warmup (ROM-VISDE); CollapseGuard, a
callback that ends training (StopTraining) when the model starts to explain an observed column as pure noise; and
ReleaseDiffusion, a callback that lets fixed diffusion dimensions learn after a number of steps.
"""
import itertools
from typing import Callable, Iterable, Union

import torch
from torch import Tensor

from .model import LatentSDE


class StopTraining(Exception):
    """Raised by a callback to end fit early; fit then returns the losses so far."""


class CollapseGuard:
    """
    Ends training when an observed column collapses: the decoder mean at the posterior mean, over the steps of
    `batch` where the column is observed, varies by less than `ratio` times the decoder's noise std for it, so the
    model explains the column as noise (as y2 in runs/simple_multimodal_20261016-154052). Checked every `every` steps
    from step `start`; a column must be collapsed in `patience` checks in a row, so that one which is still being
    learned does not count. After a stop, self.stopped_at is the step and self.collapsed the column indices.
    """

    def __init__(self, batch: dict, every: int = 500, start: int = 0, ratio: float = 0.1, patience: int = 3):
        self.batch, self.every, self.start, self.ratio, self.patience = batch, every, start, ratio, patience
        self.strikes, self.stopped_at, self.collapsed = {}, None, []

    def spread(self, model: LatentSDE) -> Tensor:
        """Per column: std of the decoded value over its observed steps / its mean noise std (nan below 2 steps)."""
        batch, mask = self.batch, self.batch["mask"]
        with torch.no_grad():
            x = torch.where(mask > 0, batch["x"], torch.zeros_like(batch["x"]))
            if model.encoder is None:
                feat, feat_mask = x, mask
            else:
                feat = model.encoder(x, mask) if getattr(model.encoder, "uses_mask", False) else model.encoder(x)
                feat_mask = mask.any(-1, keepdim=True).expand_as(feat).to(feat.dtype)
            model.posterior.condition({**batch, "x": feat, "mask": feat_mask})
            m = model.posterior(batch["t"])[0]
            dist = model.decoder(m, batch["mu"][:, None].expand(-1, m.shape[1], -1))
            out = torch.full((mask.shape[-1],), float("nan"), device=mask.device)
            for c in range(mask.shape[-1]):
                seen = mask[..., c] > 0
                if seen.sum() >= 2:
                    out[c] = dist.mean[..., c][seen].std() / dist.stddev[..., c][seen].mean()
        return out.cpu()

    def __call__(self, step: int, model: LatentSDE, loss: float) -> None:
        if step < self.start or step % self.every:
            return
        spread = self.spread(model)
        low = [c for c, r in enumerate(spread.tolist()) if r < self.ratio]  # nan (unobserved) never counts
        self.strikes = {c: self.strikes.get(c, 0) + 1 for c in low}
        print(f"collapse guard, step {step}: decoded std / noise std per column "
              f"{[round(r, 3) for r in spread.tolist()]}", flush=True)
        done = sorted(c for c, n in self.strikes.items() if n >= self.patience)
        if done:
            self.stopped_at, self.collapsed = step, done
            raise StopTraining(f"columns {done} collapsed: their decoded value varies by less than {self.ratio} "
                               f"times their noise std, in {self.patience} checks in a row")


class ReleaseDiffusion:
    """
    At step `at`, lets the fixed dimensions of the model's ConstantDiagonalDiffusion learn, from their fixed values:
    a schedule for the diffusion of a state that is observed rarely. Learned from the start, it trades off with the
    drift: the model explains the state's motion between its observations partly as noise of its own, and the
    trajectories that do not observe it make that cheap (runs/simple_multimodal_linear_d10_continued_20261017-184055:
    diffusion of x 0.51 for 0, drive 0.48 for 1; fixed at 0.05, runs/simple_multimodal_linear_fixed_x_20261017-190012:
    drive 0.86). Fixed first, the posterior and the drift are fitted before the diffusion can absorb the motion.
    """

    def __init__(self, at: int):
        self.at = at

    def __call__(self, step: int, model: LatentSDE, loss: float) -> None:
        if step == self.at:
            model.diffusion.release()
            print(f"step {step}: fixed diffusion dimensions released", flush=True)


def fit(model: LatentSDE, data: Union[dict, Iterable[dict]], n_iter: int = 5000, lr: float = 1e-3,
        n_warmup: int = 1000, lr_decay_every: int = 1000, log_every: int = 500,
        callbacks: Iterable[Callable[[int, LatentSDE, float], None]] = (), diffusion_penalty: float = 0.0,
        beta_max: float = 1.0, clip_grad: float = None) -> list:
    """
    Adam with the learning rate lr * 0.9^(i / lr_decay_every), and beta from 0 to 1 over the first n_warmup
    iterations (ROM-VISDE), to beta_max (0: no dynamics prior, an ablation). data is one batch, or an iterable of batches. Returns the loss of each iteration.
    diffusion_penalty goes to LatentSDE.loss.
    callbacks (step, model, loss), e.g. src/core/callbacks.py's Checkpoint, History and Log, or CollapseGuard; steps
    are counted from 1, as in discovery/train.py's fit. A callback that raises StopTraining ends the loop early.
    """
    batches = itertools.repeat(data) if isinstance(data, dict) else itertools.cycle(data)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9 ** (1 / lr_decay_every))
    history = []
    try:
        for i, batch in zip(range(n_iter), batches):
            loss, terms = model.loss(batch, beta=beta_max * (min(1.0, i / n_warmup) if n_warmup else 1.0),
                                     diffusion_penalty=diffusion_penalty)
            optimizer.zero_grad()
            loss.backward()
            if clip_grad:  # the norm of the gradient of all the parameters, e.g. for a sharp fixed observation noise
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            scheduler.step()
            history.append(loss.item())
            if log_every and i % log_every == 0:
                print(i, f"loss {loss.item():.4g}", *(f"{k} {v.item():.4g}" for k, v in terms.items()))
            for callback in callbacks:
                callback(i + 1, model, loss.item())
    except StopTraining as stop:
        print(f"stopped after step {len(history)}: {stop}", flush=True)
    return history
