"""Turn a nyx model into a NumPyro model."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from nyx.core.parameter import (
    _friendly_path,
    _iter_parameters,
    _matching_paths,
    set_parameters,
)

__all__ = ["scene_model", "free_parameters", "init_values"]

_MISSING = (
    "nyx.infer needs NumPyro, which nyx does not require by default. "
    'Install it with `pip install "nyx[infer]"`.'
)


def _numpyro() -> Any:
    try:
        import numpyro
    except ImportError as exc:  # pragma: no cover - exercised only without numpyro
        raise ImportError(_MISSING) from exc
    return numpyro


def free_parameters(model: Any) -> dict[str, tuple[int, ...]]:
    """``{name: shape}`` of every unfrozen parameter.

    The sites :func:`scene_model` samples, and what the optimizer would
    fit: a model says once which parameters are free.  Shapes are of the
    physical value, which is what a prior is over.
    """
    return {
        _friendly_path(path): tuple(jnp.shape(p.value))
        for path, p in _iter_parameters(model)
        if not p.frozen
    }


def init_values(model: Any) -> dict[str, jax.Array]:
    """Free parameter values, keyed to :func:`scene_model`'s sites.

    For warm-starting a chain at a fit rather than at the prior::

        kernel = NUTS(scene_model(fitted, ...),
                      init_strategy=init_to_value(values=init_values(fitted)))
    """
    return {_friendly_path(path): p.value for path, p in _iter_parameters(model) if not p.frozen}


def _resolve_priors(
    model: Any, priors: Mapping[str, Any], free: dict[str, tuple[int, ...]]
) -> dict[str, Any]:
    """Expand globbed prior keys onto the free parameter names."""
    resolved: dict[str, Any] = {}
    for pattern, prior in priors.items():
        names = [_friendly_path(path) for path in _matching_paths(model, pattern)]
        matched = [n for n in names if n in free]
        if not matched:
            raise KeyError(
                f"the prior {pattern!r} matches no free parameter. It matches "
                f"{names or 'nothing'}, and the free parameters are "
                f"{sorted(free)}. Unfreeze it, or drop the prior."
            )
        for name in matched:
            resolved[name] = prior

    missing = sorted(set(free) - set(resolved))
    if missing:
        raise KeyError(
            f"no prior given for free parameter(s) {', '.join(missing)}. Every "
            f"free parameter needs one, or it would be sampled from nothing; "
            f"freeze it instead if it should be held fixed."
        )
    return resolved


def scene_model(
    model: Any,
    data: Mapping[str, Any],
    priors: Mapping[str, Any],
    likelihood: Callable[[jax.Array], Any],
) -> Callable[[], None]:
    """A NumPyro model sampling *model*'s unfrozen parameters.

    One site per free parameter, named and shaped as the parameter already
    is, so a ``(nobs, 2)`` pointing offset is one site rather than two to
    re-stack.  Sites carry physical values, so priors are in physical
    units -- and the sampler sees the problem's own conditioning.  Give
    NUTS a few hundred warmup steps and start it from a fit with
    :func:`init_values`.

    Parameters
    ----------
    model : Scene or MultiTargetFit
        Free parameters unfrozen, as :class:`~nyx.infer.Optimizer`
        reads it.
    data : mapping of {instrument: array}
        Observed pixel rates, ``(nobs, n_pixels)``.
    priors : mapping of {name: distribution}
        One per free parameter; keys may glob. A scalar distribution is
        broadcast to the parameter's shape.
    likelihood : callable
        ``(predicted) -> distribution`` for one instrument's rates.

    Examples
    --------
    ::

        scene = unfreeze(freeze_all(scene), 'atmosphere.Mie.aod_500', '*.shift')
        model = scene_model(
            scene, {'CT1': images},
            {'atmosphere.Mie.aod_500': dist.LogNormal(np.log(0.1), 0.5),
             '*.shift': dist.Normal(0.0, 1e-3)},
            lambda p: dist.LogNormal(jnp.log(p), 0.05),
        )
    """
    numpyro = _numpyro()
    free = free_parameters(model)
    if not free:
        raise ValueError(
            "every parameter in this model is frozen, so there is nothing to "
            "sample; unfreeze what you want to infer."
        )
    resolved = _resolve_priors(model, priors, free)

    rendered = model.render()
    unknown = set(data) - set(rendered)
    if unknown:
        raise KeyError(f"data has no instrument {sorted(unknown)}; model has {sorted(rendered)}")

    def numpyro_model() -> None:
        sampled = {
            name: numpyro.sample(name, resolved[name].expand(shape or ()).to_event(len(shape)))
            for name, shape in free.items()
        }
        predicted = set_parameters(model, sampled).render()
        for instrument, observed in data.items():
            observed = jnp.asarray(observed)
            numpyro.sample(
                f"rates/{instrument}",
                likelihood(predicted[instrument]).to_event(np.ndim(observed)),
                obs=observed,
            )

    return numpyro_model
