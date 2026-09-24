from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

__all__ = [
    "AtmosphereResult",
    "PerObs",
    "PointSourceData",
    "RenderGeometry",
    "SkyGeometry",
    "SourceObsData",
    "unwrap",
]


class PerObs(eqx.Module):
    """Marks a value whose leading axis is the observation axis.

    Attributes
    ----------
    value : pytree
        The wrapped value, carrying a leading ``(nobs, ...)`` axis.
    """

    value: Any


def unwrap(x: Any) -> Any:
    """The value inside a :class:`PerObs`, or *x* unchanged if it is not one.

    Parameters
    ----------
    x : object

    Returns
    -------
    object
    """
    return x.value if isinstance(x, PerObs) else x


class SkyGeometry(eqx.Module):
    """Hemisphere positions and FOV grid for one observation, in every frame."""

    altaz_coord: jnp.ndarray  # (nsky, 2)
    icrs_coord: jnp.ndarray  # (nsky, 2)
    sref_coord: jnp.ndarray  # (nsky, 2)
    fov_altaz_grid: jnp.ndarray  # (n_lon, n_lat, 2) lon-major; pairs are (az, alt)
    height_km: jnp.ndarray  # scalar - observer height above sea level [km]
    hemisphere_mask: jnp.ndarray  # (npix,) bool - upper hemisphere pixels


class RenderGeometry(eqx.Module):
    """Sky geometry and pointing for one observation.

    Both fields vary per observation, so both are wrapped in
    :class:`PerObs`; read them through :func:`unwrap`.
    """

    sky: PerObs  # wraps SkyGeometry
    pointing_matrix: PerObs  # wraps (3, 3)


class PointSourceData(eqx.Module):
    """Point source data for the render loop (single observation)."""

    spectra: jax.Array  # (n_sources, n_wvl)
    coords: jax.Array  # (n_sources, 2) AltAz (az, alt) in radians


class SourceObsData(eqx.Module):
    """Per-instrument observation data for any sky source.

    Returned by every emitter builder's ``prepare(obs)``. All condition
    fields are optional: a diffuse-only emitter sets ``diffuse_conditions``,
    a point-only one ``source_conditions`` and ``source_coords``.

    Wrap a field in :class:`PerObs` when it carries a leading observation
    axis, and leave it bare when one value is shared by every observation.

    Parameters
    ----------
    diffuse_conditions : PerObs, jax.Array or None
        Per-pixel conditions for the diffuse path.
    diffuse_norm : jax.Array
        Factor applied after spectral evaluation, e.g. ``1 / pixel_area``.
    source_conditions : PerObs, jax.Array or None
        Per-source conditions for the point path.
    source_coords : PerObs, jax.Array or None
        Point-source AltAz positions, shape ``(n_src, 2)`` per observation.
    direct : bool
        Whether diffuse radiance takes line-of-sight extinction.  Map
        scattering applies either way.
    inscatter : bool
        Whether point sources are in-scattered individually via
        ``scatter_sources``.
    """

    diffuse_conditions: PerObs | jax.Array | None = None
    diffuse_norm: jax.Array = eqx.field(default_factory=lambda: jnp.array(1.0))
    source_conditions: PerObs | jax.Array | None = None
    source_coords: PerObs | jax.Array | None = None
    direct: bool = eqx.field(static=True, default=True)
    inscatter: bool = eqx.field(static=True, default=False)


class AtmosphereResult(eqx.Module):
    """Output of atmosphere evaluation, consumed by the render loop.

    ``extinction_hp`` and ``scattering_map`` cover the upper hemisphere only
    (``nsky`` pixels, HEALPix RING ordering); ``npix`` is the full-sphere
    count, so that :meth:`apply_extinction` can return a full map.
    """

    extinction_hp: jax.Array  # [nsky, n_wvl]   exp(-tau * sec_z)
    scattering_map: jax.Array  # [grid_lon, grid_lat, nsky, n_wvl]
    npix: int = eqx.field(static=True)  # full-sphere HEALPix pixel count

    @property
    def nsky(self) -> int:
        """Number of hemisphere sky pixels."""
        return self.extinction_hp.shape[0]

    @property
    def n_wvl(self) -> int:
        """Number of wavelength bins."""
        return self.extinction_hp.shape[1]

    def apply_extinction(self, sky_radiance: jax.Array, bandpass: jax.Array) -> jax.Array:
        """Band-integrate radiance through line-of-sight extinction.

        Parameters
        ----------
        sky_radiance : jax.Array, shape (nsky, n_wvl)
        bandpass : jax.Array, shape (n_wvl,)

        Returns
        -------
        jax.Array, shape (npix,)
            Full-sphere HEALPix array, zero below the horizon; hemisphere
            pixels lead it in RING ordering.
        """
        hp_values = jnp.sum(bandpass * sky_radiance * self.extinction_hp, axis=-1)
        return jnp.zeros(self.npix).at[: hp_values.shape[0]].set(hp_values)

    def apply_scattering(self, sky_radiance_obs: jax.Array, bandpass: jax.Array) -> jax.Array:
        """Scatter radiance into the FOV grid, one observation at a time.

        Parameters
        ----------
        sky_radiance_obs : jax.Array, shape (nsky, n_wvl)
        bandpass : jax.Array, shape (n_wvl,)

        Returns
        -------
        jax.Array, shape (n_lon, n_lat)
        """
        return jnp.sum(bandpass * sky_radiance_obs * self.scattering_map, axis=(-2, -1))
