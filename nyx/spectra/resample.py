from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

__all__ = ["bin_edges", "bin_widths", "resample_flux"]


def bin_edges(wvls: ArrayLike) -> jax.Array:
    """Bin edges of a wavelength grid: midpoints, half-extrapolated at the ends.

    Parameters
    ----------
    wvls : array-like
        Strictly increasing wavelength grid, at least two points.

    Returns
    -------
    jax.Array, shape (len(wvls) + 1,)
    """
    wl = jnp.asarray(wvls)
    edges = jnp.zeros(len(wl) + 1)
    edges = edges.at[1:-1].set((wl[1:] + wl[:-1]) / 2)
    edges = edges.at[0].set(wl[0] - (wl[1] - wl[0]) / 2)
    edges = edges.at[-1].set(wl[-1] + (wl[-1] - wl[-2]) / 2)
    return edges


def bin_widths(wvls: ArrayLike) -> jax.Array:
    """Width of each bin of a wavelength grid: the per-nm quadrature weight.

    Parameters
    ----------
    wvls : array-like
        Strictly increasing wavelength grid, at least two points.

    Returns
    -------
    jax.Array, shape (len(wvls),)
    """
    return jnp.diff(bin_edges(wvls))


def _linear_interp(wl_in: ArrayLike, flux_in: jax.Array, wl_out: ArrayLike) -> jax.Array:
    if flux_in.ndim == 1:
        return jnp.interp(wl_out, wl_in, flux_in)
    return jnp.stack([jnp.interp(wl_out, wl_in, f) for f in flux_in], axis=0)


def _check_coverage(covered: jax.Array, wl_in: jax.Array, wl_out: jax.Array) -> None:
    """Raise if any output bin lies entirely outside the input grid."""
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
    wvls_in: jax.Array, flux_in: jax.Array, wvls_out: jax.Array, method: str = "conserve"
) -> jax.Array:
    """Resample flux from one wavelength grid to another.

    Parameters
    ----------
    wvls_in : array-like
        Source wavelengths in nm.
    flux_in : array-like
        Source flux, 1-D or ``(batch, wavelength)``.
    wvls_out : array-like
        Target wavelengths in nm.
    method : str
        ``'conserve'`` or ``'linear'``.

    Returns
    -------
    jax.Array
    """
    if method == "conserve":
        return _conserve_interp(wvls_in, flux_in, wvls_out)
    elif method == "linear":
        return _linear_interp(wvls_in, flux_in, wvls_out)
    else:
        raise ValueError(f"Unknown method: {method}")
