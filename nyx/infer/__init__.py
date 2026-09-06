"""Estimating a nyx model's parameters from data.

Everything that takes a built :class:`~nyx.core.scene.Scene` and asks
*what values fit these counts* lives here.  :mod:`nyx.core` stays the
forward model alone: it renders, it does not solve.

Two ways to answer the question, over the same model and the same
:class:`~nyx.core.parameter.Parameter` tree:

**Optimisation** -- a point estimate plus its covariance.

- :mod:`~nyx.infer.optimizer` drives a solver and reports the fit.
- :mod:`~nyx.infer.uncertainty` turns a converged fit into error bars
  and a correlation matrix.
- :mod:`~nyx.infer.profile` holds a profile-likelihood scan.
- :mod:`~nyx.infer.multitarget` links parameters across targets.
- :mod:`~nyx.infer.convergence` records the path a fit took, step by
  step, for a convergence plot or animation.

**Sampling** -- the full posterior, via NumPyro.

- :mod:`~nyx.infer.numpyro_bridge` reads the sample sites off the model:
  the parameters left unfrozen, named by the paths
  :func:`~nyx.core.parameter.parameters_table` prints and shaped as they
  already are.  Requires the ``nyx[infer]`` extra.

Both read the same switch -- a frozen parameter is held, an unfrozen one
is solved for -- so a scene set up for one is set up for the other::

    from nyx.infer import Optimizer, scene_model

    scene = nyx.freeze_all(scene)
    scene = nyx.unfreeze(scene, 'atmosphere.Mie.aod_500')

    fit = Optimizer(loss).run(scene)          # point estimate
    kernel = numpyro.infer.NUTS(scene_model)  # posterior
"""

from nyx.infer.convergence import FitTrace, record_fit
from nyx.infer.multitarget import MultiTargetFit
from nyx.infer.numpyro_bridge import free_parameters, init_values, scene_model
from nyx.infer.optimizer import FitSummary, Optimizer
from nyx.infer.profile import ProfileGrid
from nyx.infer.uncertainty import (
    ParameterCorrelation,
    parameter_correlation,
    parameter_errors,
    rescale_from_errors,
)

__all__ = [
    # optimisation
    "FitSummary",
    "FitTrace",
    "MultiTargetFit",
    "Optimizer",
    "ParameterCorrelation",
    "ProfileGrid",
    "parameter_correlation",
    "parameter_errors",
    "record_fit",
    "rescale_from_errors",
    # sampling
    "free_parameters",
    "init_values",
    "scene_model",
]
