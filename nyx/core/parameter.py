"""The trainable parameter: a value, a scale, and how to constrain it."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

__all__ = ["Parameter", "is_parameter"]


_TRANSFORMS: dict[str | None, tuple[Callable, Callable, Callable]] = {
    None: (lambda u: u, lambda v: v, lambda u: jnp.ones_like(u)),
    "log": (jnp.exp, jnp.log, jnp.exp),
    "softplus": (
        lambda u: jnp.logaddexp(u, 0.0),
        lambda v: v + jnp.log(-jnp.expm1(-v)),
        jax.nn.sigmoid,
    ),
    "tanh": (jnp.tanh, jnp.arctanh, lambda u: 1.0 - jnp.tanh(u) ** 2),
}

# Physical domain of each transform.
_TRANSFORM_DOMAINS: dict[str | None, str] = {
    None: "(-inf, inf)",
    "log": "(0, inf)",
    "softplus": "(0, inf)",
    "tanh": "(-1, 1)",
}


def _scale10(value: ArrayLike) -> float:
    """Return ``10 ** floor(log10(max |value|))``, or 1.0 for zero input.

    Parameters
    ----------
    value : array-like

    Returns
    -------
    float
    """
    v = float(jnp.max(jnp.abs(jnp.asarray(value))))
    if v == 0.0 or not np.isfinite(v):
        return 1.0
    return float(10.0 ** np.floor(np.log10(v)))

class Parameter(eqx.Module):
    """A trainable physical parameter with an explicit characteristic scale.

    Attributes
    ----------
    factor : jax.Array
        The only pytree leaf; O(1) for a well-chosen *scale*.
    scale : float
        Characteristic scale in the unconstrained space, static for JIT.
        Change it via :func:`~nyx.core.paramtree.autoscale`.
    per_obs : bool
        Whether the parameter carries a leading ``(nobs, ...)`` axis after
        :func:`~nyx.core.filters.tile_per_obs`.
    frozen : bool
        Whether the parameter is excluded from optimization.
    transform : str or None
        ``None`` (unconstrained), ``'log'`` or ``'softplus'`` for a positive
        quantity, or ``'tanh'`` for one confined to ``(-1, 1)``.
    """

    factor: jax.Array
    scale: float = eqx.field(static=True, default=1.0)
    per_obs: bool = eqx.field(static=True, default=False)
    frozen: bool = eqx.field(static=True, default=False)
    transform: str | None = eqx.field(static=True, default=None)

    @property
    def value(self) -> jax.Array:
        """Physical value ``transform(factor * scale)``."""
        return _TRANSFORMS[self.transform][0](self.factor * self.scale)

    @property
    def dvalue_dfactor(self) -> jax.Array:
        """``d value / d factor``, elementwise; ``scale`` when untransformed."""
        return self.scale * _TRANSFORMS[self.transform][2](self.factor * self.scale)

    def __repr__(self) -> str:
        # equinox would show ``factor``, the leaf the optimizer steps in --
        # for a log-transformed AOD of 0.1 that reads -2.3.
        value = np.asarray(self.value)
        shown = (
            f"{float(value):+.4g}"
            if value.ndim == 0
            else (f"[{float(value.min()):+.4g}, {float(value.max()):+.4g}] {value.shape}")
        )
        flags = "".join(
            f", {name}" for name, on in (("per_obs", self.per_obs), ("frozen", self.frozen)) if on
        )
        transform = f", {self.transform}" if self.transform else ""
        return f"Parameter({shown}{transform}{flags})"

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
        transform: str | None = None,
    ) -> Parameter:
        """Construct from a physical value.

        Parameters
        ----------
        value : array-like
            Physical value; must lie in the domain of *transform*.
        scale : float or None
            Characteristic scale.  When ``None``, ``10**floor(log10(max|value|))``
            of the unconstrained value; 1.0 if that is zero or *transform* is set.
        per_obs, frozen : bool
        transform : str or None

        Returns
        -------
        Parameter

        Raises
        ------
        ValueError
            If *transform* is unknown, or *value* lies outside its domain.
        """
        if transform not in _TRANSFORMS:
            raise ValueError(
                f"Unknown transform {transform!r}; "
                f"choices are None, {', '.join(repr(k) for k in _TRANSFORMS if k)}."
            )
        v = jnp.asarray(value)
        if not jnp.issubdtype(v.dtype, jnp.floating):
            v = v.astype(jnp.float32)
        unconstrained = _TRANSFORMS[transform][1](v)
        if not isinstance(unconstrained, jax.core.Tracer) and bool(
            jnp.any(jnp.isnan(unconstrained))
        ):
            raise ValueError(
                f"value {np.asarray(value)} lies outside {_TRANSFORM_DOMAINS[transform]}, "
                f"the domain of the {transform!r} transform."
            )
        if scale is None:
            scale = 1.0 if transform is not None else _scale10(unconstrained)
        scale = float(scale)
        return cls(
            factor=unconstrained / scale,
            scale=scale,
            per_obs=per_obs,
            frozen=frozen,
            transform=transform,
        )


def is_parameter(x: Any) -> bool:
    """Whether *x* is a :class:`Parameter`; the ``is_leaf`` predicate throughout.

    Parameters
    ----------
    x : object

    Returns
    -------
    bool
    """
    return isinstance(x, Parameter)
