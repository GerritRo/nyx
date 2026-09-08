"""Finding, naming and editing the Parameters inside a pytree."""

from __future__ import annotations

import fnmatch
from collections.abc import Callable, Iterable, Iterator
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from nyx.core.parameter import Parameter, is_parameter

__all__ = [
    "ParametersTable",
    "autoscale",
    "dump_params",
    "freeze",
    "freeze_all",
    "friendly_keypath",
    "friendly_path",
    "hide_path_segments",
    "iter_parameters",
    "matching_paths",
    "n_trainable",
    "navigate",
    "parameters_table",
    "set_parameters",
    "unfreeze",
    "unfreeze_all",
    "walk",
]


def navigate(obj: Any, path: tuple[Any, ...]) -> Any:
    """Follow a path tuple through a mix of eqx.Modules, dicts, lists, tuples."""
    for step in path:
        if isinstance(step, int) or isinstance(obj, dict):
            obj = obj[step]
        else:
            obj = getattr(obj, step)
    return obj


def walk(
    tree: Any,
    match: Callable[[Any], bool],
    *,
    prune: Callable[[Any], bool] | None = None,
    descend_into_match: bool = False,
    _prefix: tuple[Any, ...] = (),
) -> Iterator[tuple[tuple[Any, ...], Any]]:
    """Yield ``(path, node)`` for the nodes of *tree* satisfying *match*.

    Depth-first over eqx.Module fields, dict items and list/tuple entries,
    with the path spelled the way :func:`navigate` reads it back.

    Parameters
    ----------
    tree : pytree
        Root to walk; itself a candidate.
    match : callable
        ``node -> bool``; only matching nodes are yielded.
    prune : callable, optional
        ``node -> bool``; descent stops at a node satisfying it, which is
        still yielded first if it matches.
    descend_into_match : bool
        Whether to keep walking below a matching node.

    Yields
    ------
    tuple
        Path from *tree* to the node.
    object
        The node.
    """
    if match(tree):
        yield _prefix, tree
        if not descend_into_match:
            return
    if prune is not None and prune(tree):
        return

    if isinstance(tree, eqx.Module):
        for name in tree.__dataclass_fields__:
            yield from walk(getattr(tree, name), match, prune=prune, _prefix=_prefix + (name,))
    elif isinstance(tree, dict):
        for key, child in tree.items():
            yield from walk(child, match, prune=prune, _prefix=_prefix + (key,))
    elif isinstance(tree, (list, tuple)):
        for i, child in enumerate(tree):
            yield from walk(child, match, prune=prune, _prefix=_prefix + (i,))


def iter_parameters(tree: Any) -> Iterator[tuple[tuple[Any, ...], Parameter]]:
    """Yield ``(path_tuple, Parameter)`` for every Parameter in *tree*."""
    return walk(tree, is_parameter)


def friendly_keypath(keypath: Iterable[Any]) -> str:
    """A jax KeyPath as the dotted name :func:`friendly_path` gives.

    Parameters
    ----------
    keypath : iterable

    Returns
    -------
    str
    """
    parts = []
    for entry in keypath:
        for attr in ("name", "key", "idx"):
            if hasattr(entry, attr):
                parts.append(getattr(entry, attr))
                break
    return friendly_path(p for p in parts if p != "factor")


def friendly_path(parts: Iterable[Any]) -> str:
    """Join path *parts* into a dotted name, dropping hidden container fields.

    Parameters
    ----------
    parts : iterable

    Returns
    -------
    str
    """
    return ".".join(str(p) for p in parts if p not in _HIDDEN_PATH_SEGMENTS)


def _format_value(param: Parameter) -> str:
    v = np.asarray(param.value)
    if v.ndim == 0:
        return f"{float(v):+.4e}"
    lo, hi = float(np.min(v)), float(np.max(v))
    if lo == hi:
        return f"{lo:+.4e}"
    return f"[{lo:+.4e}, {hi:+.4e}]"


class ParametersTable:
    """Pretty-printable listing of every :class:`Parameter` in a pytree.

    Instances render as a fixed-width text table via ``str``/``repr``
    and as an HTML table in Jupyter via ``_repr_html_``.
    """

    _COLS = ("name", "value", "scale", "shape", "per_obs", "frozen", "transform")

    def __init__(self, rows: Iterable[dict[str, str]]) -> None:
        self._rows = list(rows)

    def __repr__(self) -> str:
        if not self._rows:
            return "Parameters: (none)"
        widths = {c: max(len(c), *(len(r[c]) for r in self._rows)) for c in self._COLS}
        sep = "  "
        header = sep.join(c.ljust(widths[c]) for c in self._COLS)
        rule = sep.join("-" * widths[c] for c in self._COLS)
        body = "\n".join(sep.join(r[c].ljust(widths[c]) for c in self._COLS) for r in self._rows)
        return f"{header}\n{rule}\n{body}"

    __str__ = __repr__

    def _repr_html_(self) -> str:
        if not self._rows:
            return "<p>Parameters: (none)</p>"
        head = "".join(f"<th>{c}</th>" for c in self._COLS)
        body = "".join(
            "<tr>" + "".join(f"<td>{r[c]}</td>" for c in self._COLS) + "</tr>" for r in self._rows
        )
        return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def parameters_table(tree: Any) -> ParametersTable:
    """Build a :class:`ParametersTable` listing every Parameter in *tree*.

    Parameters
    ----------
    tree : pytree

    Returns
    -------
    ParametersTable
    """
    rows = []
    for path, p in iter_parameters(tree):
        rows.append(
            {
                "name": friendly_path(path),
                "value": _format_value(p),
                "scale": f"{p.scale:.2e}",
                "shape": str(tuple(np.asarray(p.factor).shape)),
                "per_obs": "yes" if p.per_obs else "-",
                "frozen": "yes" if p.frozen else "-",
                "transform": p.transform or "-",
            }
        )
    rows.sort(key=lambda r: r["name"])
    return ParametersTable(rows)


def set_parameters[T](tree: T, params: dict[str, Any]) -> T:
    """Set Parameters by the names :func:`parameters_table` prints.

    Parameters
    ----------
    tree : pytree
    params : dict of str to array-like
        Name or glob pattern to value.  A raw value is wrapped preserving
        the target's metadata, and must match its shape or be a scalar.

    Returns
    -------
    pytree
    """
    # Resolve every pattern first, then rebuild once.
    resolved: dict[tuple[Any, ...], tuple[str, Any]] = {}
    for pattern, value in params.items():
        for path in matching_paths(tree, pattern):
            resolved[path] = (pattern, value)
    if not resolved:
        return tree
    paths = list(resolved)
    wrapped = tuple(
        _wrap_value(pattern, navigate(tree, path), value)
        for path, (pattern, value) in resolved.items()
    )
    return eqx.tree_at(lambda t: tuple(navigate(t, p) for p in paths), tree, wrapped)


def n_trainable(tree: Any, *, include_frozen: bool = False) -> int:
    """Number of free scalar values: the ``n`` of a reduced chi-squared.

    Parameters
    ----------
    tree : pytree
    include_frozen : bool
        Whether to count frozen values too.

    Returns
    -------
    int
    """
    return sum(
        int(np.size(np.asarray(p.factor)))
        for _, p in iter_parameters(tree)
        if include_frozen or not p.frozen
    )


def dump_params(tree: Any) -> dict[str, np.ndarray]:
    """Physical value of every Parameter, frozen ones included.

    Parameters
    ----------
    tree : pytree

    Returns
    -------
    dict of str to numpy.ndarray
        Keyed by :func:`friendly_path`, so the result round-trips through
        :func:`set_parameters`.
    """
    return {friendly_path(path): np.asarray(p.value) for path, p in iter_parameters(tree)}


def autoscale[T](tree: T) -> T:
    """Rescale every :class:`Parameter` in *tree*, preserving value and metadata.

    ``factor`` is brought into the ``[0.1, 10)`` band up to sign; transformed
    parameters keep ``scale = 1.0``.

    Parameters
    ----------
    tree : pytree

    Returns
    -------
    pytree
    """

    def at_leaf(x: Any) -> Any:
        if is_parameter(x):
            return Parameter.from_value(
                x.value,
                per_obs=x.per_obs,
                frozen=x.frozen,
                transform=x.transform,
            )
        return x

    return jax.tree.map(at_leaf, tree, is_leaf=is_parameter)


def freeze_all[T](tree: T) -> T:
    """Return *tree* with every :class:`Parameter` frozen."""
    return jax.tree.map(
        lambda x: x.freeze() if is_parameter(x) else x,
        tree,
        is_leaf=is_parameter,
    )


def unfreeze_all[T](tree: T) -> T:
    """Return *tree* with every :class:`Parameter` unfrozen."""
    return jax.tree.map(
        lambda x: x.unfreeze() if is_parameter(x) else x,
        tree,
        is_leaf=is_parameter,
    )


def _apply_selector[T](
    tree: T, selector: Callable[[Any], Any], op: Callable[[Parameter], Parameter]
) -> T:
    """Apply *op* to the Parameter that the ``eqx.tree_at`` *selector* resolves to.

    Raises
    ------
    TypeError
        If *selector* does not resolve to a Parameter.
    """
    target = selector(tree)
    if not is_parameter(target):
        raise TypeError(f"Selector must resolve to a Parameter, got {type(target).__name__}.")
    return eqx.tree_at(selector, tree, op(target))


def matching_paths(tree: Any, pattern: str) -> list[tuple[Any, ...]]:
    """Internal paths of every Parameter whose displayed name matches *pattern*.

    Parameters
    ----------
    tree : pytree
    pattern : str
        Glob against the names :func:`parameters_table` prints.

    Returns
    -------
    list of tuple

    Raises
    ------
    KeyError
        If *pattern* matches no parameter.
    """
    found = [
        path for path, _ in iter_parameters(tree) if fnmatch.fnmatch(friendly_path(path), pattern)
    ]
    if not found:
        known = sorted(friendly_path(path) for path, _ in iter_parameters(tree))
        raise KeyError(
            f"{pattern!r} matches no parameter. A pattern that matches nothing is "
            f"almost always a typo, so it is refused rather than silently ignored. "
            f"Available: {', '.join(known) if known else '(none)'}"
        )
    return found


def _apply_at_paths[T](
    tree: T, paths: list[tuple[Any, ...]], op: Callable[[Parameter], Parameter]
) -> T:
    """Apply *op* to the Parameters at *paths*."""
    return eqx.tree_at(
        lambda t: tuple(navigate(t, p) for p in paths),
        tree,
        tuple(op(navigate(tree, p)) for p in paths),
    )


def _apply[T](tree: T, selectors: tuple[Any, ...], op: Callable[[Parameter], Parameter]) -> T:
    for sel in selectors:
        if isinstance(sel, str):
            tree = _apply_at_paths(tree, matching_paths(tree, sel), op)
        else:
            tree = _apply_selector(tree, sel, op)
    return tree


def freeze[T](tree: T, *selectors: str | Callable[[Any], Any]) -> T:
    """Return *tree* with the Parameters at *selectors* frozen.

    Parameters
    ----------
    tree : pytree
    *selectors : str or callable
        A parameter name as :func:`parameters_table` prints it, optionally
        globbed, or an ``eqx.tree_at`` callable.

    Returns
    -------
    pytree

    Raises
    ------
    KeyError
        If a pattern matches no parameter.
    """
    return _apply(tree, selectors, Parameter.freeze)


def unfreeze[T](tree: T, *selectors: str | Callable[[Any], Any]) -> T:
    """Return *tree* with the Parameters at *selectors* unfrozen.

    Parameters
    ----------
    tree : pytree
    *selectors : str or callable
        As :func:`freeze`.

    Returns
    -------
    pytree
    """
    return _apply(tree, selectors, Parameter.unfreeze)


def _wrap_like(target: Any, value: Any) -> Any:
    """Wrap *value* into a Parameter carrying *target*'s metadata.

    Returns *value* unchanged unless *target* is a Parameter and *value* is
    not.
    """
    if is_parameter(target) and not is_parameter(value):
        return Parameter.from_value(
            value,
            scale=target.scale,
            per_obs=target.per_obs,
            frozen=target.frozen,
            transform=target.transform,
        )
    return value


def _wrap_value(path: str, target: Any, value: Any) -> Any:
    """Wrap *value* for the Parameter at *path*, checking its shape fits.

    Raises
    ------
    ValueError
        If *value* is neither a scalar nor the target's own shape.
    """
    wrapped = _wrap_like(target, value)
    if is_parameter(target) and is_parameter(wrapped):
        want = jnp.shape(target.factor)
        got = jnp.shape(wrapped.factor)
        if got != () and got != want:
            per_obs = " (leading axis is the observation count)" if target.per_obs else ""
            raise ValueError(
                f"{path!r} has shape {want}{per_obs}, but the value given has shape "
                f"{got}; pass a matching array or a scalar to fill it"
            )
    return wrapped


# Intermediate container names to hide from displayed paths:
# ``instruments.CT1.shift`` reads better as ``CT1.shift``.
_HIDDEN_PATH_SEGMENTS: set[str] = set()


def hide_path_segments(*names: str) -> None:
    """Declare container fields that should not appear in parameter names.

    Call at import time from the module owning the field.

    Parameters
    ----------
    *names : str
        Field names to hide.
    """
    _HIDDEN_PATH_SEGMENTS.update(names)
