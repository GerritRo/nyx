from __future__ import annotations

import warnings
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


def _check_coverage(covered: jax.Array, wl_in: jax.Array, wl_out: jax.Array) -> None:
    """
    Raise if any output bin lies entirely outside the input grid.
    """
    if isinstance(covered, jax.core.Tracer):
        return
    gaps = np.flatnonzero(np.asarray(covered) <= 0)
    if gaps.size == 0:
        return
    wl_out_np, wl_in_np = np.asarray(wl_out), np.asarray(wl_in)
    raise ValueError(
        f"resample_flux: {gaps.size} of {wl_out_np.size} output bins lie outside "
        f"the input wavelength grid ({wl_in_np[0]:.3f}-{wl_in_np[-1]:.3f} nm). "
        f"Uncovered output wavelengths span {wl_out_np[gaps[0]]:.3f}-"
        f"{wl_out_np[gaps[-1]]:.3f} nm. Narrow the output grid, extend the "
        f"input table, or pass method='linear' to hold the edge value instead."
    )


def bin_edges(wvl: ArrayLike) -> jax.Array:
    """
    Bin edges of a wavelength grid: midpoints, half-extrapolated at the ends.

    Parameters
    ----------
    wvl : array
        Strictly increasing wavelength grid, at least two points.

    Returns
    -------
    jax.Array, shape ``(len(wvl) + 1,)``
    """
    wl = jnp.asarray(wvl)
    edges = jnp.zeros(len(wl) + 1)
    edges = edges.at[1:-1].set((wl[1:] + wl[:-1]) / 2)
    edges = edges.at[0].set(wl[0] - (wl[1] - wl[0]) / 2)
    edges = edges.at[-1].set(wl[-1] + (wl[-1] - wl[-2]) / 2)
    return edges


def bin_widths(wvl: ArrayLike) -> jax.Array:
    """
    Width of each bin of a wavelength grid, from :func:`bin_edges`.

    The quadrature weight for integrating a per-nm quantity over the
    grid.

    Parameters
    ----------
    wvl : array
        Strictly increasing wavelength grid, at least two points.

    Returns
    -------
    jax.Array, shape ``(len(wvl),)``
    """
    return jnp.diff(bin_edges(wvl))


def _conserve_interp(wl_in: jax.Array, flux_in: jax.Array, wl_out: jax.Array) -> jax.Array:
    if len(wl_in) < 2 or len(wl_out) < 2:
        return _linear_interp(wl_in, flux_in, wl_out)

    edges_in = bin_edges(wl_in)
    edges_out = bin_edges(wl_out)

    overlap_lo = jnp.maximum(edges_out[:-1, None], edges_in[None, :-1])
    overlap_hi = jnp.minimum(edges_out[1:, None], edges_in[None, 1:])
    overlap = jnp.clip(overlap_hi - overlap_lo, 0, None)

    was_1d = flux_in.ndim == 1
    if was_1d:
        flux_in = flux_in[None, :]

    weighted = flux_in[:, None, :] * overlap[None, :, :]
    covered = jnp.sum(overlap, axis=-1)
    _check_coverage(covered, wl_in, wl_out)
    flux_out = jnp.sum(weighted, axis=-1) / jnp.where(covered > 0, covered, jnp.nan)

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
    conditions to spectra.

    ``params`` may be either a raw array / pytree (non-trainable) or a
    :class:`~nyx.core.parameter.Parameter` (trainable). When it is a
    Parameter, ``__call__`` passes ``params.value`` to ``_model_fn`` so the
    model function itself sees a raw array either way.

    Examples:

    * **Grid interpolation** (Pickles/Gaia stars):
      ``params`` = spectral grid, ``conditions`` = raw photometry
      (e.g. ``[G, BP, RP]``), ``model_fn`` computes color internally
      and does grid interpolation + magnitude scaling.
      Use :meth:`from_color_grid` factory.
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
        active_fn: Callable[[Any], Any] | None = None,
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
            ``conditions -> color_array``. Extracts the colour index from
            the conditions array, e.g. ``lambda c: c[..., 2] - c[..., 1]``
            for Gaia RP - BP.
        mag_fn : Callable
            ``conditions -> mag_array``. Extracts the magnitude from the
            conditions array, e.g. ``lambda c: c[..., 0]`` for Gaia G.
        active_fn : Callable or None
            ``conditions -> active_array``.  Optional 0/1 mask switching
            individual sources off, e.g. ``lambda c: c[..., 2]`` for a
            below-horizon flag.
        """
        axis = np.asarray(spec_grid.points[0], dtype=float)
        wvl_native = to_wavelength_nm(spec_grid.wvl)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices
            flx_native = np.nanmedian(spec_grid.flx, axis=-1)
        unreachable = ~np.isfinite(flx_native).all(axis=-1)
        if unreachable.any():
            bad = axis[unreachable]
            raise ValueError(
                f"from_color_grid: {unreachable.sum()} of {axis.size} colour grid "
                f"nodes are not reachable by any template in the library, spanning "
                f"colour {bad.min():.3f} to {bad.max():.3f}. Narrow the colour range "
                f"or widen the reddening range the grid was built over."
            )
        flux_resampled = resample_flux(wvl_native, jnp.asarray(flx_native), wavelengths)

        grid_axis = jnp.asarray(axis)
        grid_points = (grid_axis,)
        clip_min, clip_max = grid_axis[0], grid_axis[-1]

        def _interp_model(params: jax.Array, conditions: jax.Array) -> jax.Array:
            from jax.scipy.interpolate import RegularGridInterpolator

            interpol = RegularGridInterpolator(
                grid_points,
                params,
                method="linear",
                fill_value=0,
            )
            color = jnp.clip(color_fn(conditions), clip_min, clip_max)
            mag = mag_fn(conditions)
            shapes = interpol(color)
            spectra = 10 ** (-0.4 * mag)[..., None] * shapes
            if active_fn is not None:
                spectra = active_fn(conditions)[..., None] * spectra
            return spectra

        return cls(params=flux_resampled, _model_fn=_interp_model)
