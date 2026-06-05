from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from nyx.core.parameter import Parameter
from nyx.core.units import to_wavelength_nm

if TYPE_CHECKING:
    from nyx.utils.spectra import SpectralGrid


def _linear_interp(wl_in: ArrayLike, flux_in: jax.Array, wl_out: ArrayLike) -> jax.Array:
    if flux_in.ndim == 1:
        return jnp.interp(wl_out, wl_in, flux_in)
    return jnp.stack([jnp.interp(wl_out, wl_in, f) for f in flux_in], axis=0)


def _conserve_interp(wl_in: jax.Array, flux_in: jax.Array, wl_out: jax.Array) -> jax.Array:
    if len(wl_in) < 2 or len(wl_out) < 2:
        return _linear_interp(wl_in, flux_in, wl_out)

    def get_edges(wl: jax.Array) -> jax.Array:
        edges = jnp.zeros(len(wl) + 1)
        edges = edges.at[1:-1].set((wl[1:] + wl[:-1]) / 2)
        edges = edges.at[0].set(wl[0] - (wl[1] - wl[0]) / 2)
        edges = edges.at[-1].set(wl[-1] + (wl[-1] - wl[-2]) / 2)
        return edges

    edges_in = get_edges(wl_in)
    edges_out = get_edges(wl_out)
    delta_in = edges_in[1:] - edges_in[:-1]
    delta_out = edges_out[1:] - edges_out[:-1]

    overlap_lo = jnp.maximum(edges_out[:-1, None], edges_in[None, :-1])
    overlap_hi = jnp.minimum(edges_out[1:, None], edges_in[None, 1:])
    overlap = jnp.clip(overlap_hi - overlap_lo, 0, None)
    safe_delta_in = jnp.maximum(delta_in, jnp.finfo(delta_in.dtype).eps)
    frac = overlap / safe_delta_in[None, :]

    was_1d = flux_in.ndim == 1
    if was_1d:
        flux_in = flux_in[None, :]

    weighted = flux_in[:, None, :] * frac[None, :, :] * delta_in[None, None, :]
    flux_out = jnp.sum(weighted, axis=-1) / delta_out[None, :]

    if was_1d:
        flux_out = flux_out[0]
    return flux_out


def resample_flux(
    wvl_in: jax.Array, flux_in: jax.Array, wvl_out: jax.Array, method: str = "conserve"
) -> jax.Array:
    """Resample flux array(s) from one wavelength grid to another.

    Parameters
    ----------
    wvl_in : array
        Source wavelengths in nm.
    flux_in : array
        Source flux, 1D or 2D (batch, wavelength).
    wvl_out : array
        Target wavelengths in nm.
    method : str
        'conserve' (default) or 'linear'.
    """
    if method == "conserve":
        return _conserve_interp(wvl_in, flux_in, wvl_out)
    elif method == "linear":
        return _linear_interp(wvl_in, flux_in, wvl_out)
    else:
        raise ValueError(f"Unknown method: {method}")


# SpectralModel hierarchy


class SpectralModel(eqx.Module):
    """Base class for spectral models.

    A SpectralModel maps source conditions to spectra.

    Any :class:`~nyx.core.parameter.Parameter` fields on a subclass are
    automatically discovered and trained by :class:`Optimizer`.
    """

    def __call__(self, conditions: jax.Array | None = None) -> jax.Array:
        raise NotImplementedError


class StoredSpectrum(SpectralModel):
    """Return a fixed, pre-computed spectrum array."""

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

    General-purpose class for any deterministic mapping from source
    conditions to spectra.  The ``model_fn`` defines the physics;
    ``params`` holds the data or weights it needs.

    ``params`` may be either a raw array / pytree (non-trainable) or a
    :class:`~nyx.core.parameter.Parameter` (trainable).  When it is a
    Parameter, ``__call__`` passes ``params.value`` to ``_model_fn`` so the
    model function itself sees a raw array either way.

    Examples:

    * **Grid interpolation** (Pickles/Gaia stars):
      ``params`` = spectral grid, ``conditions`` = raw photometry
      (e.g. ``[G, BP, RP]``), ``model_fn`` computes color internally
      and does grid interpolation + magnitude scaling.
      Use :meth:`from_color_grid` factory.

    * **Neural emulator** (e.g. PHOENIX/MARCS):
      ``params`` = network weights, ``conditions`` = ``[teff, logg, mag]``,
      ``model_fn`` runs the forward pass.
    """

    params: object  # raw array/pytree, or Parameter
    _model_fn: Callable[..., jax.Array] = eqx.field(static=True)

    def __call__(self, conditions: jax.Array | None = None) -> jax.Array:
        p = self.params.value if isinstance(self.params, Parameter) else self.params
        return self._model_fn(p, conditions)

    @classmethod
    def from_color_grid(
        cls,
        spec_grid: SpectralGrid,
        wavelengths: jax.Array,
        color_fn: Callable[[Any], Any],
        mag_fn: Callable[[Any], Any],
    ) -> ParametricSpectrum:
        """Create from a colour-indexed :class:`SpectralGrid`.

        Builds a model that interpolates the Pickles (or similar) atlas
        by colour index and scales by magnitude::

            spectrum = 10^(-0.4 * mag_fn(conditions)) * grid_interp(color_fn(conditions))

        Parameters
        ----------
        spec_grid : SpectralGrid
            Colour-indexed spectral grid (e.g. from :func:`create_color_grid`).
        wavelengths : array
            Target wavelength grid to resample onto.
        color_fn : Callable
            ``conditions -> color_array``.  Extracts the colour index from
            the conditions array, e.g. ``lambda c: c[..., 2] - c[..., 1]``
            for Gaia RP - BP.
        mag_fn : Callable
            ``conditions -> mag_array``.  Extracts the magnitude from the
            conditions array, e.g. ``lambda c: c[..., 0]`` for Gaia G.
        """
        wvl_native = to_wavelength_nm(spec_grid.wvl)
        flx_native = jnp.asarray(np.nanmedian(spec_grid.flx, axis=-1))
        flux_resampled = resample_flux(wvl_native, flx_native, wavelengths)

        grid_points = (jnp.asarray(np.asarray(spec_grid.points[0])),)
        clip_max = float(np.asarray(spec_grid.points[0])[-1]) - 0.01

        def _interp_model(params: jax.Array, conditions: jax.Array) -> jax.Array:
            from jax.scipy.interpolate import RegularGridInterpolator

            interpol = RegularGridInterpolator(
                grid_points,
                params,
                method="linear",
                fill_value=0,
            )
            color = jnp.clip(color_fn(conditions), max=clip_max)
            mag = mag_fn(conditions)
            shapes = interpol(color)
            return 10 ** (-0.4 * mag)[..., None] * shapes

        return cls(params=flux_resampled, _model_fn=_interp_model)
