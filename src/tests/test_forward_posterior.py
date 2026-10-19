"""SequencePosterior(forward_only=True) is a forward posterior (a filter): the posterior at t does not change when an observation after t changes."""
import torch

from src.latent import SequencePosterior  # src.latent first: it settles the import order of src.encoders


def _posterior(forward_only: bool, seed: int = 0) -> SequencePosterior:
    torch.manual_seed(seed)
    return SequencePosterior(dim_x=5, dim_z=3, dim_mu=2, hidden=16, bidirectional=False, forward_only=forward_only)


def _batch(seed: int = 1, B: int = 4, N: int = 7):
    g = torch.Generator().manual_seed(seed)
    t = torch.sort(torch.rand(B, N, generator=g) * 5, dim=1).values
    x = torch.randn(B, N, 5, generator=g)
    mask = (torch.rand(B, N, 5, generator=g) > 0.3).float()
    mask[:, :, 0] = 1.0  # every step has a value
    return {"t": t, "x": x, "mask": mask, "mu": torch.randn(B, 2, generator=g)}


def _mean_at_steps(posterior: SequencePosterior, batch: dict, keep: int):
    """The posterior mean at the first `keep` observation times, with the whole batch as input."""
    posterior.condition(batch)
    with torch.no_grad():
        return posterior(batch["t"][:, :keep])[0]


def test_forward_posterior_ignores_the_future():
    posterior, batch = _posterior(True), _batch()
    before = _mean_at_steps(posterior, batch, 4)
    changed = {**batch, "x": batch["x"].clone()}
    changed["x"][:, 4:] += 10.0  # the observations from step 4 on
    after = _mean_at_steps(posterior, changed, 4)
    assert torch.allclose(before, after, atol=1e-6), (before - after).abs().max()


def test_forward_posterior_sees_the_past_and_the_present():
    posterior, batch = _posterior(True), _batch()
    before = _mean_at_steps(posterior, batch, 4)
    changed = {**batch, "x": batch["x"].clone()}
    changed["x"][:, 2] += 10.0
    after = _mean_at_steps(posterior, changed, 4)
    assert not torch.allclose(before[:, 2:], after[:, 2:], atol=1e-3)  # step 2 reaches times 2 and 3
    assert torch.allclose(before[:, :2], after[:, :2], atol=1e-6)  # but not times 0 and 1


def test_non_forward_posterior_does_see_the_future():
    posterior, batch = _posterior(False), _batch()
    before = _mean_at_steps(posterior, batch, 4)
    changed = {**batch, "x": batch["x"].clone()}
    changed["x"][:, 4:] += 10.0
    after = _mean_at_steps(posterior, changed, 4)
    assert not torch.allclose(before, after, atol=1e-3)  # the attention averages over the later steps


def test_forward_posterior_is_a_filter_of_each_prefix():
    """The mean at the first 4 times equals that of a batch cut to the first 4 steps."""
    posterior, batch = _posterior(True), _batch()
    full = _mean_at_steps(posterior, batch, 4)
    cut = {k: (v[:, :4] if v.dim() > 1 and v.shape[1] == 7 else v) for k, v in batch.items()}
    assert torch.allclose(full, _mean_at_steps(posterior, cut, 4), atol=1e-6)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
