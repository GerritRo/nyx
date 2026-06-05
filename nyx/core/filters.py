from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from nyx.core.parameter import Parameter, _is_param

__all__ = ["per_obs_filter", "tile_per_obs", "_navigate"]


def _navigate(obj: Any, path: tuple[Any, ...]) -> Any:
    """Follow a path tuple through a mix of eqx.Modules, dicts, lists, tuples.

    Shared utility for :meth:`nyx.core.scene.Scene.set` / ``set_params``.
    """
    for step in path:
        if isinstance(step, int) or isinstance(obj, dict):
            obj = obj[step]
        else:
            obj = getattr(obj, step)
    return obj


def _find_declared_paths(
    root: Any, declaration: str, _prefix: tuple[Any, ...] = ()
) -> Iterator[tuple[Any, ...]]:
    """Yield path tuples to every eqx.Module whose ``declaration`` attribute
    is a non-empty tuple.  Does not descend into :class:`Parameter`
    instances (they carry their own metadata)."""
    if isinstance(root, Parameter):
        return
    if isinstance(root, eqx.Module) and getattr(root, declaration, None):
        yield _prefix
    if isinstance(root, eqx.Module):
        for name in root.__dataclass_fields__:
            yield from _find_declared_paths(
                getattr(root, name),
                declaration,
                _prefix + (name,),
            )
    elif isinstance(root, dict):
        for key, child in root.items():
            yield from _find_declared_paths(
                child,
                declaration,
                _prefix + (key,),
            )
    elif isinstance(root, (list, tuple)):
        for i, child in enumerate(root):
            yield from _find_declared_paths(
                child,
                declaration,
                _prefix + (i,),
            )


def per_obs_filter[T](tree: T) -> T:
    """Boolean mask for every per-observation leaf in *tree*.

    Marks True in two cases:

    - The ``factor`` leaf of any :class:`Parameter` with ``per_obs=True``.
    - Every leaf under a field listed in a containing eqx.Module's
      ``_per_obs`` class tuple.
    """

    def at_leaf(x: Any) -> Any:
        if _is_param(x):
            return dataclasses.replace(x, factor=bool(x.per_obs))
        return False

    filt = jax.tree.map(at_leaf, tree, is_leaf=_is_param)

    for path in _find_declared_paths(tree, "_per_obs"):
        component = _navigate(tree, path)
        for field_name in component._per_obs:
            full_path = path + (field_name,)
            subtree = _navigate(filt, full_path)

            def _const_true(_: object) -> bool:
                return True

            replacement = jax.tree.map(
                _const_true,
                subtree,
                is_leaf=_is_param,
            )

            def _at_path(m: object, _p: tuple[Any, ...] = full_path) -> object:
                return _navigate(m, _p)

            filt = eqx.tree_at(_at_path, filt, replacement)
    return filt


def tile_per_obs[T](tree: T, nobs: int) -> T:
    """Tile the ``factor`` of every :class:`Parameter` with ``per_obs=True``
    to leading shape ``(nobs, ...)``.

    Data containers (:class:`SourceObsData`, :class:`RenderGeometry`) do
    their own stacking at construction time, so this function only handles
    the Parameter path.
    """

    def at_leaf(x: Any) -> Any:
        if _is_param(x) and x.per_obs:
            return dataclasses.replace(
                x,
                factor=jnp.stack([x.factor] * nobs),
            )
        return x

    return jax.tree.map(at_leaf, tree, is_leaf=_is_param)
