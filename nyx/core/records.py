from __future__ import annotations

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp

__all__ = [
    "AtmosphereResult",
    "PointSourceData",
    "RenderGeometry",
    "SkyGeometry",
    "SourceObsData",
]


class SkyGeometry(eqx.Module):
    """Hemisphere positions and FOV grid for one observation, in every frame."""

    altaz_coord: jnp.ndarray  # (nsky, 2)
    icrs_coord: jnp.ndarray  # (nsky, 2)
    sref_coord: jnp.ndarray  # (nsky, 2)
    fov_altaz_grid: jnp.ndarray  # (n_lon, n_lat, 2) lon-major; pairs are (az, alt)
    height_km: jnp.ndarray  # scalar - observer height above sea level [km]
    hemisphere_mask: jnp.ndarray  # (npix,) bool - upper hemisphere pixels


class RenderGeometry(eqx.Module):
    """Sky geometry and pointing for one observation."""

    sky: SkyGeometry
    pointing_matrix: jnp.ndarray  # (3, 3)
    per_obs_fields: tuple[str, ...] = eqx.field(
        static=True, default=("sky", "pointing_matrix")
    )


class PointSourceData(eqx.Module):
    """Point source data for the render loop (single observation)."""

    spectra: jax.Array  # (n_sources, n_wvl)
    coords: jax.Array  # (n_sources, 2) AltAz (az, alt) in radians


class SourceObsData(eqx.Module):
    """Per-instrument observation data for any sky source.

    Returned by every emitter builder's ``prepare(obs)``. All condition
    fields are optional: a diffuse-only emitter sets ``diffuse_conditions``,
    a point-only one ``source_conditions`` and ``source_coords``.

    Parameters
    ----------
    diffuse_conditions : jax.Array or None
        Per-pixel conditions for the diffuse path.
    diffuse_norm : jax.Array
        Factor applied after spectral evaluation, e.g. ``1 / pixel_area``.
    source_conditions : jax.Array or None
        Per-source conditions for the point path.
    source_coords : jax.Array or None
        Point-source AltAz positions, shape ``(nobs, n_src, 2)``.
    direct : bool
        Whether diffuse radiance takes line-of-sight extinction.  Map
        scattering applies either way.
    inscatter : bool
        Whether point sources are in-scattered individually via
        ``scatter_sources``.
    per_obs_fields : tuple of str
        Which of the above carry a leading observation axis.
    """

    diffuse_conditions: jax.Array | None = None
    diffuse_norm: jax.Array = eqx.field(default_factory=lambda: jnp.array(1.0))
    source_conditions: jax.Array | None = None
    source_coords: jax.Array | None = None
    direct: bool = eqx.field(static=True, default=True)
    inscatter: bool = eqx.field(static=True, default=False)
    per_obs_fields: tuple[str, ...] = eqx.field(static=True, default=())

    def __check_init__(self) -> None:
        valid = {f.name for f in dataclasses.fields(self) if f.name != "per_obs_fields"}
        bad = set(self.per_obs_fields) - valid
        if bad:
            raise ValueError(
                f"SourceObsData.per_obs_fields references unknown fields: {bad!r}. Valid fields: {valid!r}"
            )


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
