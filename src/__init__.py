"""
Latent SDEs in PyTorch.

Layout, from the bottom up:

    core/        linear algebra, transforms, small networks, interpolation, Euler-Maruyama
    kernels/     kernels with analytic time derivatives, input warpings (SVISE, ROM-VISDE)
    quadrature/  quadrature rules for the time integral of the residual (SVISE)
    bayes/       variational families and sparsity priors (SVISE, old JAX src)
    data/        trajectory loading, PCA (SVISE)
    posterior/   drift of the approximate posterior from its marginals (gauges)
    discovery/   equation discovery without an encoder: the SVISE stack, its training loop and evaluation
    encoders/    posteriors of the latent SDE: KernelPosterior (svise type, kernel interpolation for a GP,
                 ROM-VISDE) and SequencePosterior (SDE-matching type, a network of state and time)
    decoders/    p(x | z, mu) of the latent SDE: GaussianDecoder, one modality today
    latent/      encoder-decoder latent SDE with parameters mu and forcings u(t), and its training loop
                 (ROM-VISDE and SDE matching)
    plotting.py  figures of the approximate variational posterior and of the posterior SDE paths

Reused code: SVISE (Course & Nair) and ROM-VISDE (Ilersich, Course & Nair) are GPL-3.0; Helmholtz-SDE (Smith,
Trippe & Linderman) is MIT. The sde_matching repository has no license, so its method is written from the paper. Each reused module names its origin.
Imports inside the package are relative, so the package works under any folder name.
"""
