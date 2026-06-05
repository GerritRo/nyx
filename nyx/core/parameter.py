from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Iterator
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "Parameter",
    "autoscale",
    "freeze",
    "unfreeze",
    "freeze_all",
    "unfreeze_all",
    "parameters_table",
    "dump_params",
]


def _scale10(value: ArrayLike) -> float:
    """Return ``10 ** floor(log10(max |value|))``, or 1.0 for zero input.

    Used by :meth:`Parameter.from_value` to pick a default scale when the
    caller does not supply one.
    """
    v = float(jnp.max(jnp.abs(jnp.asarray(value))))
    if v == 0.0 or not np.isfinite(v):
        return 1.0
    return float(10.0 ** np.floor(np.log10(v)))


class Parameter(eqx.Module):
    """A trainable physical parameter with an explicit characteristic scale.

    ``value = factor * scale``.  The optimizer differentiates through
    ``factor`` only, so a well-chosen ``scale`` makes a single global
    learning rate work across parameters of very different physical
    magnitudes (e.g. ~1e-4 rad shifts vs ~1e0 aerosol optical depth).

    Attributes
    ----------
    factor : jax.Array
        The JAX leaf.  O(1) by construction when ``scale`` is chosen
        correctly.  This is the only pytree leaf; ``scale``, ``per_obs``
        and ``frozen`` are static metadata.
    scale : float
        Characteristic physical scale.  Set once (typically at
        construction) and static for JIT; change it via
        :func:`autoscale` or by building a new Parameter.
    per_obs : bool
        If True, this parameter carries a leading ``(nobs, ...)`` axis
        after :func:`tile_per_obs`; typical for instrument pointing
        (``shift``, ``rotation``) that varies observation-to-observation.
    frozen : bool
        If True, the parameter is excluded from optimization even when
        selected by default.
    """

    factor: jax.Array
    scale: float = eqx.field(static=True, default=1.0)
    per_obs: bool = eqx.field(static=True, default=False)
    frozen: bool = eqx.field(static=True, default=False)

    @property
    def value(self) -> jax.Array:
        """Physical value ``factor * scale``."""
        return self.factor * self.scale

    def freeze(self) -> Parameter:
        """Return a copy with ``frozen=True``."""
        return dataclasses.replace(self, frozen=True)

    def unfreeze(self) -> Parameter:
        """Return a copy with ``frozen=False``."""
        return dataclasses.replace(self, frozen=False)

    @classmethod
    def from_value(
        cls,
        value: ArrayLike,
        scale: float | None = None,
        per_obs: bool = False,
        frozen: bool = False,
    ) -> Parameter:
        """Construct from a physical value.

        Parameters
        ----------
        value : array-like
            Physical value (in the original physical units).
        scale : float or None
            Characteristic scale.  When ``None``, chosen as
            ``10**floor(log10(max|value|))``; 1.0 for zero.
        per_obs, frozen : bool
            Metadata flags; see class docstring.
        """
        v = jnp.asarray(value)
        if not jnp.issubdtype(v.dtype, jnp.floating):
            v = v.astype(jnp.float32)
        if scale is None:
            scale = _scale10(v)
        scale = float(scale)
        return cls(factor=v / scale, scale=scale, per_obs=per_obs, frozen=frozen)


# Tree-level utilities


def _is_param(x: Any) -> bool:
    return isinstance(x, Parameter)


def autoscale[T](tree: T) -> T:
    """Rescale every :class:`Parameter` in *tree* using ``scale10``.

    ``value`` is preserved; ``factor`` is brought into the [0.1, 10) band
    (up to sign) for non-zero parameters.  Flags (``per_obs``, ``frozen``)
    are preserved.
    """

    def at_leaf(x: Any) -> Any:
        if _is_param(x):
            return Parameter.from_value(
                x.value,
                per_obs=x.per_obs,
                frozen=x.frozen,
            )
        return x

    return jax.tree.map(at_leaf, tree, is_leaf=_is_param)


def freeze_all[T](tree: T) -> T:
    """Return *tree* with every :class:`Parameter` frozen."""
    return jax.tree.map(
        lambda x: x.freeze() if _is_param(x) else x,
        tree,
        is_leaf=_is_param,
    )


def unfreeze_all[T](tree: T) -> T:
    """Return *tree* with every :class:`Parameter` unfrozen."""
    return jax.tree.map(
        lambda x: x.unfreeze() if _is_param(x) else x,
        tree,
        is_leaf=_is_param,
    )


def _apply_selector[T](
    tree: T, selector: Callable[[Any], Any], op: Callable[[Parameter], Parameter]
) -> T:
    """Apply *op* (a Parameter → Parameter function) at the position pointed
    to by *selector* (a ``eqx.tree_at`` callable).
    """
    target = selector(tree)
    if not _is_param(target):
        raise TypeError(f"Selector must resolve to a Parameter, got {type(target).__name__}.")
    return eqx.tree_at(selector, tree, op(target))


def freeze[T](tree: T, *selectors: Callable[[Any], Any]) -> T:
    """Return *tree* with the Parameters addressed by *selectors* frozen.

    Each selector is a callable ``tree -> Parameter`` of the form used with
    :func:`equinox.tree_at`, e.g.::

        scene = freeze(
            scene,
            lambda s: s.instruments['CT1'].efficiency,
            lambda s: s.atmosphere.components['Mie'].hg_asymmetry,
        )
    """
    for sel in selectors:
        tree = _apply_selector(tree, sel, Parameter.freeze)
    return tree


def unfreeze[T](tree: T, *selectors: Callable[[Any], Any]) -> T:
    """Return *tree* with the Parameters addressed by *selectors* unfrozen.

    See :func:`freeze` for the selector convention.
    """
    for sel in selectors:
        tree = _apply_selector(tree, sel, Parameter.unfreeze)
    return tree


# Auto-wrap helper used by Scene.set / set_params


def _wrap_like(target: Any, value: Any) -> Any:
    """If *target* is a :class:`Parameter` and *value* is not, wrap
    *value* into a new Parameter that preserves *target*'s ``scale``,
    ``per_obs`` and ``frozen`` metadata.  Otherwise return *value*
    unchanged.
    """
    if _is_param(target) and not _is_param(value):
        return Parameter.from_value(
            value,
            scale=target.scale,
            per_obs=target.per_obs,
            frozen=target.frozen,
        )
    return value


# Parameters table (pretty-print)

# Intermediate container names to hide from displayed paths:
# ``instruments.CT1.shift`` reads better as ``CT1.shift``.
_HIDDEN_PATH_SEGMENTS = {
    "instruments",
    "sources",
    "components",
    "target_scenes",
    "canonical_instruments",
}


def _iter_parameters(
    tree: Any, _prefix: tuple[Any, ...] = ()
) -> Iterator[tuple[tuple[Any, ...], Parameter]]:
    """Yield ``(path_tuple, Parameter)`` for every Parameter in *tree*."""
    if _is_param(tree):
        yield _prefix, tree
        return
    if isinstance(tree, eqx.Module):
        for name in tree.__dataclass_fields__:
            yield from _iter_parameters(getattr(tree, name), _prefix + (name,))
    elif isinstance(tree, dict):
        for key, child in tree.items():
            yield from _iter_parameters(child, _prefix + (key,))
    elif isinstance(tree, (list, tuple)):
        for i, child in enumerate(tree):
            yield from _iter_parameters(child, _prefix + (i,))


def _friendly_path(parts: Iterable[Any]) -> str:
    return ".".join(str(p) for p in parts if p not in _HIDDEN_PATH_SEGMENTS)


def _format_value(param: Parameter) -> str:
    v = np.asarray(param.value)
    if v.ndim == 0:
        return f"{float(v):+.4e}"
    lo, hi = float(np.min(v)), float(np.max(v))
    if lo == hi:
        return f"{lo:+.4e}"
    return f"[{lo:+.4e}, {hi:+.4e}]"


class _ParametersTable:
    """Pretty-printable listing of every :class:`Parameter` in a pytree.

    Instances render as a fixed-width text table via ``str``/``repr``
    and as an HTML table in Jupyter via ``_repr_html_``.
    """

    _COLS = ("name", "value", "scale", "shape", "per_obs", "frozen")

    def __init__(self, rows: Iterable[dict[str, str]]) -> None:
        # rows: list of dicts with the keys in ``_COLS``.
        self._rows = list(rows)

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> Iterator[dict[str, str]]:
        return iter(self._rows)

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


def parameters_table(tree: Any) -> _ParametersTable:
    """Build a :class:`_ParametersTable` listing every Parameter in *tree*.

    Paths are displayed in the same short form used by
    :meth:`nyx.core.scene.Scene.set` (internal container names like
    ``instruments`` or ``components`` are stripped)."""
    rows = []
    for path, p in _iter_parameters(tree):
        rows.append(
            {
                "name": _friendly_path(path),
                "value": _format_value(p),
                "scale": f"{p.scale:.2e}",
                "shape": str(tuple(np.asarray(p.factor).shape)),
                "per_obs": "yes" if p.per_obs else "-",
                "frozen": "yes" if p.frozen else "-",
            }
        )
    rows.sort(key=lambda r: r["name"])
    return _ParametersTable(rows)


def dump_params(tree: Any) -> dict[str, np.ndarray]:
    """Return ``{friendly_path: ndarray}`` of every Parameter's physical value.

    Paths match those used by :meth:`nyx.core.scene.Scene.set_params`, so
    the dict round-trips: ``scene.set_params(dump_params(scene))`` is a
    no-op.  Frozen parameters are included.
    """
    return {_friendly_path(path): np.asarray(p.value) for path, p in _iter_parameters(tree)}
