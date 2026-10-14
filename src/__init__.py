"""
Latent SDEs in PyTorch.

Layout, from the bottom up:

    core/        linear algebra, transforms, small networks, interpolation, Euler-Maruyama
    kernels/     kernels with analytic time derivatives, input warpings (SVISE, ROM-VISDE)
    quadrature/  quadrature rules for the time integral of the residual (SVISE)
    bayes/       variational families and sparsity priors (SVISE, old JAX src)
    data/        trajectory loading, windows over trajectories (ROM-VISDE), PCA (SVISE)
    posterior/   drift of the approximate posterior from its marginals (gauges), Helmholtz correction
    discovery/   equation discovery without an encoder: the SVISE stack
    latent/      encoder-decoder models with parameters mu and forcings f(t): the ROM-VISDE stack
    training/    training loop and callbacks

Reused code: SVISE (Course & Nair) and ROM-VISDE (Ilersich, Course & Nair) are GPL-3.0, Helmholtz-SDE
(Smith, Trippe & Linderman) is MIT. Each reused module names its origin in its docstring.
Imports inside the package are relative, so the package works under any folder name.
"""
