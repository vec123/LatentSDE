"""
Fits one PCA (src/data/pca.py) per modality on the train observations of multimodal_reaction_test
(observations_train.h5, from sample_modalities.py), and writes <dir>/PCA/<modality>/model.pt (its state_dict):

    imaging    the noisy u images, flattened to (n_images, n * n); reconstructions.png shows held-out test images
    biomarker  the scalar gamma samples, (n_samples, 1). A PCA of one dimension is a standardization,
               (y - mean) / std up to sign; it is kept as a PCA so that every modality has the same encoder type

For each, prints the rank and the relative reconstruction RMS error on the train, val and test observations; for
the images also against the clean frames (truth_frames), since the imaging noise sets a floor on the first.

    python -m test_data.scripts.multimodal_reaction_test.fit_pca
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from src.data.pca import PCA
from test_data.scripts.reaction_diffusion.fit_pca import plot_reconstructions
from test_data.src.multimodal_reaction import truth_frames

DATA = Path(__file__).resolve().parents[2] / "data" / "multimodal_reaction_test"
SPLITS = ("train", "val", "test")


def load(dir: Path, split: str):
    """Noisy images (n_images, n * n), the clean frames at the same visits, and biomarker values (n_samples, 1)."""
    images, clean, bio = [], [], []
    with h5py.File(dir / f"observations_{split}.h5", "r") as g, h5py.File(dir / f"truth_{split}.h5", "r") as f:
        for i, (n_img, n_bio) in enumerate(zip(g["imaging/count"][:], g["biomarker/count"][:])):
            images.append(g["imaging/u"][i, :n_img])
            clean.append(truth_frames(f, i, g["imaging/frame"][i, :n_img]))
            bio.append(g["biomarker/y"][i, :n_bio])
    dtype = torch.get_default_dtype()
    flat = lambda a: torch.as_tensor(np.concatenate(a), dtype=dtype).flatten(1)
    return flat(images), flat(clean), torch.as_tensor(np.concatenate(bio), dtype=dtype)[:, None]


def relative_error(pca: PCA, y: torch.Tensor, target: torch.Tensor) -> float:
    return ((pca.decode(pca.encode(y)) - target).pow(2).mean().sqrt() / target.pow(2).mean().sqrt()).item()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, default=DATA)
    p.add_argument("--percent-cutoff", type=float, default=0.999, help="as in reaction_diffusion/fit_pca.py")
    p.add_argument("--max-evecs", type=int, default=30, help="the images' rank: without this cap the rank "
                   "criterion picks 157 (it fits the noise of 200 images), but the test error against the clean "
                   "frames stops falling at 30 (3.2%%; 3.5%% at 20, 8.7%% at 10)")
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--n-snapshots", type=int, default=6, help="test images shown in reconstructions.png")
    args = p.parse_args()

    data = {split: load(args.dir, split) for split in SPLITS}
    images, _, bio = data["train"]
    shape = int(round(images.shape[1] ** 0.5)), int(round(images.shape[1] ** 0.5))
    for name, y in [("imaging", images), ("biomarker", bio)]:
        print(f"{name}: fitting on {y.shape[0]} train observations of dimension {y.shape[1]}")
        pca, _ = PCA.create(y, percent_cutoff=args.percent_cutoff, max_evecs=args.max_evecs,
                            random_state=args.random_state)
        out = args.dir / "PCA" / name
        out.mkdir(parents=True, exist_ok=True)
        torch.save(pca.state_dict(), out / "model.pt")
        column = 0 if name == "imaging" else 2
        errors = {split: relative_error(pca, d[column], d[column]) for split, d in data.items()}
        line = ", ".join(f"{split} {e:.4f}" for split, e in errors.items())
        print(f"  rank {int(pca.n_components)}, relative reconstruction RMS error: {line}")
        if name == "imaging":
            clean = {split: relative_error(pca, d[0], d[1]) for split, d in data.items()}
            print("  against the clean frames: " + ", ".join(f"{split} {e:.4f}" for split, e in clean.items()))
            test = data["test"][0]
            idx = torch.linspace(0, test.shape[0] - 1, args.n_snapshots).round().long()
            plot_reconstructions(out / "reconstructions.png", test[idx].reshape(-1, *shape),
                                 pca.decode(pca.encode(test[idx])).reshape(-1, *shape),
                                 [f"test image {i}" for i in idx.tolist()], int(pca.n_components), errors["test"])
        else:
            print(f"  mean {pca.mean.item():.4f}, scale {(pca.scale * pca.evecs.abs()).item():.4f}")
        print(f"  wrote {out / 'model.pt'}")


if __name__ == "__main__":
    main()
