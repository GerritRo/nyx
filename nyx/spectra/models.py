from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax

from nyx.core.parameter import Parameter

__all__ = [
    "ParametricSpectrum",
    "PassThroughSpectrum",
    "SpectralModel",
    "StoredSpectrum",
]


class SpectralModel(eqx.Module):
    """Base class for spectral models: source conditions to spectra."""

    def __call__(self, conditions: jax.Array | None = None) -> jax.Array:
        raise NotImplementedError


class StoredSpectrum(SpectralModel):
    """Pre-computed spectrum array."""

    spectra: jax.Array

    def __call__(self, conditions: jax.Array | None = None) -> jax.Array:
        return self.spectra


class PassThroughSpectrum(SpectralModel):
    """Pass conditions through unchanged as the spectrum."""

    def __call__(self, conditions: jax.Array | None = None) -> jax.Array:
        if conditions is None:
            raise ValueError("PassThroughSpectrum requires `conditions`; got None.")
        return conditions


class ParametricSpectrum(SpectralModel):
    """Parametric spectral model: ``model_fn(params, conditions) -> spectra``.

    ``params`` may be a raw array or pytree (non-trainable), or a
    :class:`~nyx.core.parameter.Parameter` (trainable).
    """

    params: object  # raw array/pytree, or Parameter
    _model_fn: Callable[..., jax.Array] = eqx.field(static=True)

    def __call__(self, conditions: jax.Array | None = None) -> jax.Array:
        p = self.params.value if isinstance(self.params, Parameter) else self.params
        return self._model_fn(p, conditions)
