from __future__ import annotations

import dataclasses
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from nyx.core.parameter import is_parameter
from nyx.core.records import PerObs

__all__ = [
    "per_obs_filter",
    "select_obs",
    "tile_per_obs",
]


def _is_marked(x: Any) -> bool:
    return is_parameter(x) or isinstance(x, PerObs)


def per_obs_filter[T](tree: T) -> T:
    """Boolean mask for every per-observation leaf in *tree*.

    True for the ``factor`` of a :class:`~nyx.core.parameter.Parameter` with
    ``per_obs=True``, and for everything below a
    :class:`~nyx.core.records.PerObs`.

    Parameters
    ----------
    tree : pytree

    Returns
    -------
    pytree
        A filter spec: the same shape as *tree*, with a bool at every leaf and
        a single ``True`` standing in for a whole ``PerObs`` subtree.
    """

    def at_leaf(x: Any) -> Any:
        if is_parameter(x):
            return dataclasses.replace(x, factor=bool(x.per_obs))
        return isinstance(x, PerObs)

    return jax.tree.map(at_leaf, tree, is_leaf=_is_marked)


def select_obs[T](tree: T, index: int) -> T:
    """Slice observation *index* out of a stacked tree; inverse of :func:`tile_per_obs`.

    Parameters
    ----------
    tree : pytree
    index : int

    Returns
    -------
    pytree
        The same tree with the leading observation axis dropped from every
        per-observation leaf, and everything else untouched.
    """
    per_obs, shared = eqx.partition(tree, per_obs_filter(tree))
    return eqx.combine(shared, jax.tree.map(lambda x: x[index], per_obs))


def tile_per_obs[T](tree: T, nobs: int) -> T:
    """Give every ``per_obs`` Parameter in *tree* a leading ``(nobs, ...)`` axis.

    Parameters
    ----------
    tree : pytree
    nobs : int

    Returns
    -------
    pytree
        The same tree; data wrapped in :class:`~nyx.core.records.PerObs`,
        which stacks itself at construction, is untouched.
    """

    def at_leaf(x: Any) -> Any:
        if is_parameter(x) and x.per_obs:
            return dataclasses.replace(
                x,
                factor=jnp.stack([x.factor] * nobs),
            )
        return x

    return jax.tree.map(at_leaf, tree, is_leaf=is_parameter)
