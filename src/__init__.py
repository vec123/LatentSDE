"""
Latent SDEs in PyTorch.

Layout, from the bottom up:

    core/        linear algebra, transforms, small networks, interpolation, Euler-Maruyama
    kernels/     kernels with analytic time derivatives, input warpings (SVISE, ROM-VISDE)
    quadrature/  quadrature rules for the time integral of the residual (SVISE)
    bayes/       variational families and sparsity priors (SVISE, old JAX src)
    data/        trajectory loading, PCA (SVISE)
    posterior/   drift of the approximate posterior from its marginals (gauges)
    discovery/   equation discovery without an encoder: the SVISE stack
    latent.py    encoder-decoder latent SDE with parameters mu and forcings u(t), and its training loop
                 (ROM-VISDE and SDE matching)

Reused code: SVISE (Course & Nair) and ROM-VISDE (Ilersich, Course & Nair) are GPL-3.0. The sde_matching
repository has no license, so its method is written from the paper. Each reused module names its origin.
Imports inside the package are relative, so the package works under any folder name.
"""
