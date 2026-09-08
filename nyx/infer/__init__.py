"""Estimating a nyx model's parameters from data.

Optimisation (:mod:`~nyx.infer.optimizer`, :mod:`~nyx.infer.uncertainty`,
:mod:`~nyx.infer.profile`, :mod:`~nyx.infer.multitarget`,
:mod:`~nyx.infer.convergence`) gives a point estimate and its covariance;
:mod:`~nyx.infer.numpyro_bridge` gives the posterior, and needs the
``nyx[infer]`` extra.  Both solve for the unfrozen parameters and hold the
frozen ones.
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
