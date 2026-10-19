# Extend the latent SDE to ROM-VISDE and SDE matching

## Context

The project has two paths. The `discovery/` path does equation discovery. It
has no encoder. It must stay exactly as it works now. The `latent/` path has
an encoder, a decoder, and a choice of posterior: `KernelPosterior` (ROM-VISDE)
or `SequencePosterior` (SDE matching). Both paths share the drift math in
`posterior/gauges.py`.

The user asked for three things:

1. A flexible choice of posterior: the svise type (a GP, given by kernel
   interpolation in time) or the SDE-matching type (a network of state and
   time, with Gaussian reparameterization noise).
2. Diffusion that can depend on the state `z`, as SDE matching allows.
3. Encoders and decoders split into their own, reusable packages, so a future
   per-modality encoder (for the multimodal hidradenitis-suppurativa data)
   can be added without touching the model.

The user confirmed: build the encoder and decoder interfaces now, for later
multimodal use. Do not build per-modality masks yet, since no multimodal data
feeds the model yet.

Two facts from research back this plan:

- `posterior/gauges.py` states that for diagonal covariance, the svise gauge
  (`sym_drift_diagonal`) and the SDE-matching gauge give the same drift. This
  was checked by a review and confirmed by hand. So no new gauge function is
  needed for state-dependent diffusion in the diagonal case.
- `z` in the latent model has four axes, `(samples, batch, time, dim)`.
  `core/linalg.py`'s `solve_lyapunov_diag` only has special cases for two or
  three axes. A state-dependent diffusion tensor, computed from this
  four-axis `z`, will not broadcast the way that function expects. The fix
  must flatten `(samples, batch, time)` to one axis before the gauge call,
  and restore the shape after. This is a note for step 4 below, not a change
  to `core/linalg.py` itself.

## Status

### Done

1. Moved the training callbacks `Checkpoint`, `History`, `Log` out of
   `discovery/train.py` into `core/callbacks.py`. They take `(step, model,
   loss)` and never depended on `SDELearner`. `discovery/train.py` now
   imports them; its own call sites (`experiments/discovery/run.py`) are
   unchanged, since `discovery/train.py` re-exports the same names.

2. Split `latent.py` into a package, parallel to `discovery/`:
   - `latent/model.py` — `LatentSDE` and `DriftNet`.
   - `latent/train.py` — `fit`.
   - `latent/batch.py` — `make_batch` and `interp`.
   - `latent/__init__.py` re-exports `DriftNet, GaussianDecoder,
     KernelPosterior, LatentSDE, SequencePosterior, fit, interp, make_batch`,
     so `src/tests/test_latent.py` needed no changes to its imports.

3. Moved the posteriors into a new `encoders/` package:
   - `encoders/base.py` — the shared shape, `condition(batch) -> Tensor` and
     `forward(t) -> (m, logv, dm, dlogv)`, as a `Protocol`.
   - `encoders/kernel.py` — `KernelPosterior`, the svise type posterior: a GP,
     given by kernel interpolation in time. Kept as ROM-VISDE parity only:
     its kernel fixes one `dt` and one `batch_size` at construction, so it
     cannot take irregular times without a much larger rewrite.
   - `encoders/sequence.py` — `SequencePosterior`, the SDE-matching type
     posterior: a network of state (the GRU context) and time, with Gaussian
     reparameterization noise. This is the irregular-time path already, via
     its `mask` input.
   - Class names are unchanged, by the user's choice, to keep the diff small.

   Moved the decoder into a new `decoders/` package the same way:
   `decoders/base.py` (the `Decoder` protocol) and `decoders/gaussian.py`
   (`GaussianDecoder`). A future per-modality encoder or decoder is a new
   file in each package; `latent/model.py` does not need to change for it.

   Shared helpers moved to `core/`, since two or more packages now use them:
   `mlp` to `core/nn.py`, and the scalar inverse of softplus to
   `core/transforms.py` as `inverse_softplus_scalar`.

   One limit stays on record: today one `mask` tensor covers all channels at
   each time step. Real multimodal data, where each modality has its own
   missing times, will need a per-modality mask inside
   `encoders/sequence.py`. Out of scope for now, by the user's choice above.

   Verified: the full test suite passes, zero failures, same as before the
   split (`python -m src.tests.run_all`).

4. Added a state-dependent diffusion to the encoder path, in a new
   `latent/diffusion.py`:
   - `Diffusion`, a `Protocol`: `__call__(z, mu, t) -> Tensor`, a positive
     diagonal std, the same shape as `z`.
   - `ConstantDiagonalDiffusion` — the old `raw_g` behavior, moved here. It
     is the default, state-independent and learnable, built from `g_init`
     when `LatentSDE` gets no `diffusion` argument.
   - `NetDiagonalDiffusion` — `softplus(MLP([z, mu, t]))`. SDE matching's
     own volatility module takes `(z, t, *args)`, so `t` is an input here
     too.

   `LatentSDE` takes `diffusion` as a fourth swappable module, next to
   `posterior`, `drift`, `decoder`. `latent/model.py`'s `loss` calls it at
   the quadrature-time samples `z[:, :, q]`, squares it, and passes it to
   `sym_drift_diagonal` in place of the old `self.g.pow(2)`.

   The four-axis broadcast problem was real, but its fix turned out smaller
   than expected: `core/linalg.py`'s `solve_lyapunov_diag` special-cased
   `RHS_diag.dim() == 3` and left everything else to a `shape[0]` check that
   assumed rank 2. Widening that one line to `>= 3` — matching the
   function's own docstring, `(..., n)`, which already promised any number
   of leading dims — makes a four-axis `Q_diag` broadcast correctly, with no
   change at the `latent/model.py` call site and no flatten/reshape needed.
   Confirmed by hand and by the new shape test below; every existing caller
   only ever used rank 2 or 3, so this is additive, not a behavior change.

   New tests, in `src/tests/test_latent.py`:
   - `test_explicit_constant_diffusion_matches_the_default` — an explicit
     `ConstantDiagonalDiffusion(dim_z, g_init)` gives the same loss as the
     default, bit for bit (regression check).
   - `test_net_diagonal_diffusion_reduces_to_the_constant_case` — a
     `NetDiagonalDiffusion` forced (zero weight, constant bias) to output a
     fixed `g` matches the exact-OU closed form, the same way
     `test_residual_and_kl_vanish_for_exact_ou_posterior` does for the
     default diffusion.
   - `test_net_diagonal_diffusion_trains_without_shape_errors` — fits a
     model with `NetDiagonalDiffusion` at three different `(n_samples, B)`
     pairs and checks the loss stays finite, to catch the four-axis bug
     class directly rather than by luck.

   Verified: `python -m src.tests.run_all`, zero failures, including the
   three new tests above.

## Files touched

- New: `core/callbacks.py`, `encoders/__init__.py`, `encoders/base.py`,
  `encoders/kernel.py`, `encoders/sequence.py`, `decoders/__init__.py`,
  `decoders/base.py`, `decoders/gaussian.py`, `latent/__init__.py`,
  `latent/model.py`, `latent/train.py`, `latent/batch.py`,
  `latent/diffusion.py`.
- Removed: `latent.py` (split into the package above).
- Edited: `discovery/train.py` (import callbacks, no behavior change),
  `src/__init__.py` (layout docstring), `core/nn.py` (added `mlp`),
  `core/transforms.py` (added `inverse_softplus_scalar`), `core/linalg.py`
  (one-line `solve_lyapunov_diag` widening, see above),
  `src/tests/test_latent.py` (new imports and the three new tests).
- Not touched: `discovery/` beyond the import above, `posterior/`,
  `bayes/`, `data/`.

## Suggested commit order

Small commits, as already agreed for this project:

1. Move callbacks to `core/callbacks.py`.
2. Split `latent.py` into the `latent/`, `encoders/`, `decoders/` packages,
   no behavior change, tests green.
3. Add `latent/diffusion.py`, wire it into `LatentSDE`, widen
   `solve_lyapunov_diag`, and add the three new tests.

All three steps are implemented and tests pass; none are committed yet.

## Verification

- Run the full test suite: `python -m src.tests.run_all`. Currently: zero
  failures, a handful of pre-existing skips where optional reference
  packages (`gpytorch`, `jaxtyping`) are not installed in this environment.
- `src/tests/test_latent.py` alone: `python -m src.tests.test_latent`.
