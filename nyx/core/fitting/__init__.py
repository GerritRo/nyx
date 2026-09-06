"""Fitting a nyx model to data.

- :mod:`~nyx.core.fitting.optimizer` drives a solver and reports the fit.
- :mod:`~nyx.core.fitting.uncertainty` turns a converged fit into error
  bars and a correlation matrix.
- :mod:`~nyx.core.fitting.profile` holds a profile-likelihood scan.
- :mod:`~nyx.core.fitting.multitarget` links parameters across targets.
"""

from nyx.core.fitting.multitarget import MultiTargetFit
from nyx.core.fitting.optimizer import FitSummary, Optimizer
from nyx.core.fitting.profile import ProfileGrid
from nyx.core.fitting.uncertainty import (
    ParameterCorrelation,
    parameter_correlation,
    parameter_errors,
    rescale_from_errors,
)

__all__ = [
    "FitSummary",
    "MultiTargetFit",
    "Optimizer",
    "ParameterCorrelation",
    "ProfileGrid",
    "parameter_correlation",
    "parameter_errors",
    "rescale_from_errors",
]
