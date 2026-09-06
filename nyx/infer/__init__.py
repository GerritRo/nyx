"""Bayesian inference over a nyx model, via NumPyro.

:func:`scene_model` reads the sample sites off the model: the parameters
left unfrozen, named by the paths
:func:`~nyx.core.parameter.parameters_table` prints and shaped as they
already are.  Requires the ``nyx[infer]`` extra.
"""

from nyx.infer.numpyro_bridge import free_parameters, init_values, scene_model

__all__ = ["scene_model", "free_parameters", "init_values"]
