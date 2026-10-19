"""
Tests of the MLD autoencoder parts that do not need the data files.

    python -m experiments.mld_ae.test_mld_ae
"""
import numpy as np
import torch

from .data import Column
from .model import ModalityAE

COLS = [Column("count", "count"), Column("value", "normal", log=True), Column("marker", "censored"),
        Column("flag", "binary"), Column("stage", "categorical", levels=(1, 2, 3))]


def _fitted():
    raw = {"count": [0, 2, 5, 9], "value": [1.0, 2.0, 4.0, 8.0], "marker": [0.5, 1.0, 3.0, 9.0]}
    return [c.fit(raw[c.name]) if c.name in raw else c for c in COLS]


def test_transform_round_trip():
    for c in _fitted():
        raw = {"categorical": np.array([1.0, 3.0, np.nan]), "binary": np.array([0.0, 1.0, np.nan])}.get(
            c.kind, np.array([0.5, 2.0, np.nan]) if c.kind != "count" else np.array([0.0, 7.0, np.nan]))
        back = c.inverse(c.forward(raw))
        assert np.allclose(back[:2], raw[:2]) and np.isnan(back[2]), c.name


def test_missing_values_do_not_count():
    torch.manual_seed(0)
    model = ModalityAE(_fitted(), latent_dim=3, hidden=(16,))
    x = torch.tensor([[0.1, -0.3, 0.2, 1.0, 2.0], [0.5, 0.4, -0.1, 0.0, 0.0]])
    blq = torch.zeros_like(x, dtype=torch.bool)
    out, _ = model(x, blq)
    full = model.loss(out, x, blq)
    # a missing value: the same loss as with the value, minus the term of that value
    x_missing = x.clone()
    x_missing[0, 1] = float("nan")
    reduced = model.loss(out, x_missing, blq)
    term = (out[0, 1] - x[0, 1]) ** 2
    assert torch.allclose(reduced[0], full[0] - term) and torch.allclose(reduced[1], full[1])
    # the encoder input has no NaN
    assert torch.isfinite(model.inputs(x_missing, blq)).all()


def test_censored_loss_below_the_lod():
    model = ModalityAE(_fitted(), latent_dim=2, hidden=(8,))
    lod = 0.3
    x = torch.full((2, 5), float("nan"))
    x[:, 2] = lod
    blq = torch.zeros_like(x, dtype=torch.bool)
    blq[:, 2] = True
    out = torch.zeros(2, 4 + 3)  # 4 scalar outputs (count, value, marker, flag), 3 stage logits
    out[0, 2], out[1, 2] = lod - 1.0, lod + 0.5  # the marker output: below, then above the LOD
    loss = model.loss(out, x, blq)
    assert torch.allclose(loss, torch.tensor([0.0, 0.25]))


def test_categorical_input_and_prediction():
    model = ModalityAE(_fitted(), latent_dim=2, hidden=(8,))
    x = torch.full((2, 5), float("nan"))
    x[0, 4] = 2.0  # level index 2 (stage 3)
    inp = model.inputs(x, torch.zeros_like(x, dtype=torch.bool))
    # 4 scalar values, 4 masks, 1 BLQ flag, then the one-hot of the 3 levels and the mask
    assert inp.shape == (2, 4 + 4 + 1 + 3 + 1)
    assert inp[0, -4:].tolist() == [0.0, 0.0, 1.0, 1.0] and inp[1, -4:].tolist() == [0.0, 0.0, 0.0, 0.0]
    out = torch.zeros(1, 7)
    out[0, 4:] = torch.tensor([0.1, 3.0, -1.0])
    assert model.predict(out)[0, 4] == 1.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: ok")
