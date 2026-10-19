"""Training callbacks shared by discovery/train.py and latent/train.py: (step, model, loss) -> None."""
import os
import time

import torch


class Checkpoint:
    """Saves the whole model to <run_dir>/checkpoints/model_<step>.pt every `every` steps."""

    def __init__(self, run_dir: str, every: int):
        self.dir, self.every = os.path.join(run_dir, "checkpoints"), every
        os.makedirs(self.dir, exist_ok=True)

    def __call__(self, step: int, model, loss: float) -> None:
        if step % self.every == 0:
            torch.save(model, os.path.join(self.dir, f"model_{step:06d}.pt"))


class History:
    """Keeps the loss every `every` steps in self.steps and self.losses."""

    def __init__(self, every: int):
        self.every, self.steps, self.losses = every, [], []

    def __call__(self, step: int, model, loss: float) -> None:
        if step % self.every == 0:
            self.steps.append(step)
            self.losses.append(loss)


class Log:
    """Prints the loss and the time since the start every `every` steps."""

    def __init__(self, every: int, prefix: str = ""):
        self.every, self.prefix, self.start = every, prefix, time.time()

    def __call__(self, step: int, model, loss: float) -> None:
        if step % self.every == 0:
            print(f"{self.prefix}step {step} loss {loss:.3f} {time.time() - self.start:.0f}s", flush=True)
