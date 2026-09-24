from __future__ import annotations

import warnings
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from nyx import NyxWarning
from nyx.core.parameter import Parameter
from nyx.core.paramtree import friendly_keypath, friendly_path, iter_parameters


def _is_trainable(x: Any) -> bool:
    return isinstance(x, Parameter) and not x.frozen


def _sum_of_squares(residuals: Any) -> jax.Array:
    """Reduce a residual array or pytree of arrays to ``sum(r ** 2)``."""
    return jax.tree.reduce(
        lambda a, b: a + b,
        jax.tree.map(lambda x: jnp.sum(x**2), residuals),
    )


def _flat_parameter_names(diff: Any) -> list[str]:
    """For reporting by index."""
    names: list[str] = []
    for path, leaf in jax.tree_util.tree_leaves_with_path(diff):
        names += [friendly_keypath(path)] * int(np.size(leaf))
    return names


def _non_finite_parameters(tree: Any) -> list[str]:
    """Friendly paths of every Parameter in *tree* that went non-finite."""
    return [
        friendly_path(path)
        for path, p in iter_parameters(tree)
        if not bool(jnp.all(jnp.isfinite(p.factor)))
    ]


def _warn_non_finite(names: list[str], what: str, stacklevel: int = 3) -> None:
    """Report non-finite parameters."""
    if not names:
        return
    shown = ", ".join(names[:5]) + (f" and {len(names) - 5} more" if len(names) > 5 else "")
    warnings.warn(
        f"{what} is non-finite for {shown}. The loss diverged rather than "
        f"converging; the usual causes are a prediction that reached zero or "
        f"negative under a log, a residual divided by a zero uncertainty, or a "
        f"parameter driven outside the domain of its transform. Inspect "
        f"`parameters_table()`, freeze what is degenerate, or start closer.",
        NyxWarning,
        stacklevel=stacklevel,
    )
