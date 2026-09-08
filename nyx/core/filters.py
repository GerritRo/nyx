from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from nyx.core.parameter import is_parameter
from nyx.core.paramtree import navigate, walk


def _find_declared_paths(root: Any, declaration: str) -> Iterator[tuple[Any, ...]]:
    """Yield path tuples to every eqx.Module declaring *declaration* non-empty.

    Parameters
    ----------
    root : pytree
    declaration : str
        Name of the class-level tuple, e.g. ``"per_obs_fields"``.

    Yields
    ------
    tuple
        Path from *root* to each declaring module.
    """

    def declares(node: Any) -> bool:
        return isinstance(node, eqx.Module) and bool(getattr(node, declaration, None))

    # Parameters are pruned rather than descended into: they hold no
    # declarations, and their leaves are handled by the flag path above.
    for path, _node in walk(root, declares, prune=is_parameter, descend_into_match=True):
        yield path


def per_obs_filter[T](tree: T) -> T:
    """Boolean mask for every per-observation leaf in *tree*.

    True for the ``factor`` of a :class:`~nyx.core.parameter.Parameter` with
    ``per_obs=True``, and for every leaf under a field named in a containing
    eqx.Module's ``per_obs_fields``.

    Parameters
    ----------
    tree : pytree

    Returns
    -------
    pytree
        The same shape, with a bool at every leaf.
    """

    def at_leaf(x: Any) -> Any:
        if is_parameter(x):
            return dataclasses.replace(x, factor=bool(x.per_obs))
        return False

    filt = jax.tree.map(at_leaf, tree, is_leaf=is_parameter)

    for path in _find_declared_paths(tree, "per_obs_fields"):
        component = navigate(tree, path)
        for field_name in component.per_obs_fields:
            full_path = path + (field_name,)
            subtree = navigate(filt, full_path)

            def _const_true(_: object) -> bool:
                return True

            replacement = jax.tree.map(
                _const_true,
                subtree,
                is_leaf=is_parameter,
            )

            def _at_path(m: object, _p: tuple[Any, ...] = full_path) -> object:
                return navigate(m, _p)

            filt = eqx.tree_at(_at_path, filt, replacement)
    return filt


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
        The same tree; data containers, which stack themselves at
        construction, are untouched.
    """

    def at_leaf(x: Any) -> Any:
        if is_parameter(x) and x.per_obs:
            return dataclasses.replace(
                x,
                factor=jnp.stack([x.factor] * nobs),
            )
        return x

    return jax.tree.map(at_leaf, tree, is_leaf=is_parameter)
