"""
Trains the MLD autoencoder of each modality, each alone (MLD, section 3: "individual training of the different
autoencoders"). Adam, lr 1e-3, no weight decay, as the MLD autoencoders (table of appendix D). The batches are
shuffled train rows (drop_last). Each eval_every steps the loss of the val rows is computed; the model keeps the
state with the lowest val loss.

    python -m experiments.mld_ae.train [--modalities exam photo ...] [--out runs/mld_ae] [--seed 0]
                                       [--steps N] [--latent exam=8 records=16 ...]

Writes <out>/split.csv (the patient split, made once and then used again), and for each modality
<out>/<modality>/model.pt (columns with their transforms, the model, z_mean, z_std) and history.csv.
"""
import argparse
import copy
import os
import time

import numpy as np
import pandas as pd
import torch

from . import data
from .model import ModalityAE, save

# latent size, batch size and number of steps of each modality. MLD wants an autoencoder with almost no loss of
# information (section 3). The columns of serum and photo have much independent noise: with latent 4 (serum) and 6
# (photo), the test R^2 of the cytokines and the image embeddings was only 0.61 to 0.95; with 8 and 10, 0.96 to 1.00.
CONFIG = {
    "exam": dict(latent=8, batch=128, steps=3000),
    "photo": dict(latent=10, batch=128, steps=3000),
    "ultrasound": dict(latent=4, batch=128, steps=3000),
    "serum": dict(latent=8, batch=128, steps=3000),
    "self_report": dict(latent=2, batch=64, steps=1500),
    "records": dict(latent=12, batch=512, steps=4000),
}
HIDDEN, LR, EVAL_EVERY = (256, 256), 1e-3, 100


def get_split(out: str, seed: int) -> pd.DataFrame:
    path = os.path.join(out, "split.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    split = data.split_patients(seed)
    os.makedirs(out, exist_ok=True)
    split.to_csv(path, index=False)
    return split


def tensors(table: pd.DataFrame, cols, split: str):
    x, blq = data.arrays(table[table["split"] == split], cols)
    return torch.as_tensor(x), torch.as_tensor(blq)


def mean_loss(model: ModalityAE, x, blq, chunk: int = 8192) -> float:
    model.eval()
    with torch.no_grad():
        total = sum(model.loss(model(x[i:i + chunk], blq[i:i + chunk])[0], x[i:i + chunk], blq[i:i + chunk]).sum()
                    for i in range(0, len(x), chunk))
    model.train()
    return float(total) / len(x)


def train(modality: str, split: pd.DataFrame, latent: int, batch: int, steps: int, seed: int):
    """The trained model (best val state, z_mean and z_std set) and the history."""
    torch.manual_seed(seed)
    table, cols = data.load(modality, split)
    cols = data.fit(cols, table)
    x, blq = tensors(table, cols, "train")
    x_val, blq_val = tensors(table, cols, "val")
    model = ModalityAE(cols, latent, HIDDEN)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(seed)
    order, at, running, history = torch.randperm(len(x), generator=gen), 0, [], []
    best = (float("inf"), None, 0)
    for step in range(1, steps + 1):
        if at + batch > len(x):
            order, at = torch.randperm(len(x), generator=gen), 0
        idx = order[at:at + batch]
        at += batch
        out, _ = model(x[idx], blq[idx])
        loss = model.loss(out, x[idx], blq[idx]).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        running.append(loss.item())
        if step % EVAL_EVERY == 0:
            val = mean_loss(model, x_val, blq_val)
            history.append(dict(step=step, train=float(np.mean(running)), val=val))
            running = []
            if val < best[0]:
                best = (val, copy.deepcopy(model.state_dict()), step)
    model.load_state_dict(best[1])
    model.eval()
    with torch.no_grad():
        z = torch.cat([model.encode(x[i:i + 8192], blq[i:i + 8192]) for i in range(0, len(x), 8192)])
    model.z_mean.copy_(z.mean(0))
    model.z_std.copy_(z.std(0).clamp_min(1e-6))
    return model, pd.DataFrame(history), best[2], len(x), len(x_val)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modalities", nargs="+", default=list(data.MODALITIES), choices=data.MODALITIES)
    parser.add_argument("--out", default=os.path.join("runs", "mld_ae"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, help="number of steps of every modality (default: CONFIG)")
    parser.add_argument("--latent", nargs="*", default=[], help="latent sizes, as modality=size")
    args = parser.parse_args()
    latent = {k: int(v) for k, v in (item.split("=") for item in args.latent)}
    split = get_split(args.out, args.seed)
    for modality in args.modalities:
        cfg = {**CONFIG[modality], **({"steps": args.steps} if args.steps else {})}
        cfg["latent"] = latent.get(modality, cfg["latent"])
        start = time.time()
        model, history, best_step, n_train, n_val = train(modality, split, cfg["latent"], cfg["batch"],
                                                          cfg["steps"], args.seed)
        folder = os.path.join(args.out, modality)
        os.makedirs(folder, exist_ok=True)
        save(os.path.join(folder, "model.pt"), model, config={**cfg, "seed": args.seed, "lr": LR},
             best_step=best_step)
        history.to_csv(os.path.join(folder, "history.csv"), index=False)
        best = history.loc[history["step"] == best_step].iloc[0]
        print(f"{modality:12s} latent {cfg['latent']:2d}  train {n_train:6d} rows, val {n_val:5d}  "
              f"best step {best_step:5d}: train loss {best['train']:.4f}, val loss {best['val']:.4f}  "
              f"({time.time() - start:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
