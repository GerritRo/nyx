from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.coordinates.angles import angular_separation
from astropy.time import Time

from nyx.core.coordinates import (
    SunRelativeEclipticFrame,
    offset_to_altaz,
    rotation_matrix_from_altaz,
)
from nyx.core.geometry import Geometry


class SkyGeometry(eqx.Module):
    """Hemisphere sky positions and FOV grid in all coordinate systems
    (single observation).
    """

    altaz_coord: jnp.ndarray  # (nsky, 2)
    icrs_coord: jnp.ndarray  # (nsky, 2)
    sref_coord: jnp.ndarray  # (nsky, 2)
    fov_altaz_grid: jnp.ndarray  # (ngrid, ngrid, 2)
    scattering_angle: jnp.ndarray  # (ngrid, ngrid, nsky)
    height_km: jnp.ndarray  # scalar - observer height above sea level [km]
    hemisphere_mask: jnp.ndarray  # (npix,) bool - upper hemisphere pixels


class RenderGeometry(eqx.Module):
    """Combined sky geometry and pointing for the render pipeline
    (single observation)."""

    sky: SkyGeometry
    pointing_matrix: jnp.ndarray  # (3, 3)
    _per_obs: tuple[str, ...] = eqx.field(static=True, default=("sky", "pointing_matrix"))


def _extract_icrs(skycoord: SkyCoord) -> tuple[np.ndarray, np.ndarray]:
    """Extract (lon, lat) = (ra, dec) in radians from ICRS."""
    return skycoord.ra.rad, skycoord.dec.rad


def _extract_sref(skycoord: SkyCoord) -> tuple[np.ndarray, np.ndarray]:
    """Extract (lon, lat) = (alpha, beta) in radians from SunRelativeEcliptic."""
    return skycoord.alpha.rad, skycoord.beta.rad


# Registry of known frames: key -> (astropy_frame, extractor)
_BUILTIN_FRAMES = {
    "icrs": ("icrs", _extract_icrs),
    "sref": (SunRelativeEclipticFrame(), _extract_sref),
}


class Observation:
    """Multi-observation context with precomputed coordinates and ephemeris.

    Coordinate transforms are lazy: computed on first access via
    get_sky_coords() and cached. Built-in frames (ICRS, SunRelativeEcliptic)
    are pre-registered. Additional frames can be registered via
    register_frame().

    Each Observation has a single target and single location, observed at
    one or more times (nobs = len(times)).

    Parameters
    ----------
    location : EarthLocation
        Single observer location.
    times : Time
        Observation times (array or list).
    target : SkyCoord
        Single scalar target (shared across all times).
    geom : Geometry
        Resolution configuration.
    refract_pointing : bool, optional
        If True, the target ICRS is transformed to AltAz using the same
        refracting frame as the rest of the pipeline (i.e. the pointing
        direction is the apparent position). If False (default), the
        pointing direction is the geometric AltAz (no refraction), while
        the simulation pipeline still applies refraction via ``**kwargs``.
    **kwargs
        Additional AltAz frame parameters (pressure, temperature, etc.).
    """

    def __init__(
        self,
        location: EarthLocation,
        times: Time,
        target: SkyCoord,
        geom: Geometry,
        refract_pointing: bool = False,
        **kwargs: Any,
    ) -> None:
        if isinstance(target, list):
            raise TypeError("target must be a single SkyCoord, not a list.")
        if not target.isscalar:
            raise TypeError("target must be a scalar SkyCoord.")

        self.geom = geom
        self.times = times
        self.target_icrs = target
        self.nobs = len(times)

        self.location = location
        self._altaz_kwargs = kwargs

        # Build per-observation AltAz frames and pointing geometry
        self.altaz_frames = [AltAz(location=self.location, obstime=t, **kwargs) for t in times]
        ## Check if refraction should be applied for altaz pointing
        pointing_kwargs = kwargs if refract_pointing else {}
        pointing_frames = [
            AltAz(location=self.location, obstime=t, **pointing_kwargs) for t in times
        ]
        self.targets_altaz, self.pointing_matrices = self._build_pointing(
            self.target_icrs,
            pointing_frames,
        )

        # Precompute sky grids and scattering angles
        self.hp_coords_altaz = SkyCoord(
            geom.lon,
            geom.lat,
            unit="rad",
            frame=self.altaz_frames[0],
        )
        self.fov_coords = self._build_fov_coords(geom, self.pointing_matrices, self.altaz_frames)
        self.scattering_angle = self._compute_scattering_angles(
            self.fov_coords,
            self.hp_coords_altaz,
        )

        # Frame registry and coordinate cache
        self._frames = dict(_BUILTIN_FRAMES)
        self._sky_cache: dict[str, list[SkyCoord]] = {}
        self._pixel_cache: dict[str, Any] = {}

    @staticmethod
    def _build_pointing(
        target_icrs: SkyCoord, altaz_frames: list[AltAz]
    ) -> tuple[list[SkyCoord], list[np.ndarray]]:
        """Compute per-observation AltAz targets and pointing matrices."""
        targets_altaz = [target_icrs.transform_to(af) for af in altaz_frames]
        pointing_matrices = [rotation_matrix_from_altaz(t.az.rad, t.alt.rad) for t in targets_altaz]
        return targets_altaz, pointing_matrices

    @staticmethod
    def _build_fov_coords(
        geom: Geometry, pointing_matrices: list[np.ndarray], altaz_frames: list[AltAz]
    ) -> list[SkyCoord]:
        """Build FOV grid coordinates in AltAz for each observation."""
        fov_coords = []
        for R, af in zip(pointing_matrices, altaz_frames, strict=True):
            az, alt = offset_to_altaz(geom.X, geom.Y, R)
            fov_coords.append(SkyCoord(az=az, alt=alt, unit="rad", frame=af))
        return fov_coords

    @staticmethod
    def _compute_scattering_angles(
        fov_coords: list[SkyCoord], hp_coords_altaz: SkyCoord
    ) -> np.ndarray:
        """Compute scattering angles between FOV grid and hemisphere pixels."""
        return np.stack(
            [
                angular_separation(
                    fc.az.rad[:, :, np.newaxis],
                    fc.alt.rad[:, :, np.newaxis],
                    hp_coords_altaz.az.rad[np.newaxis, np.newaxis, :],
                    hp_coords_altaz.alt.rad[np.newaxis, np.newaxis, :],
                )
                for fc in fov_coords
            ],
            axis=0,
        )

    @property
    def height_km(self) -> float:
        """Observatory height in km."""
        return self.location.height.to("km").value

    def register_frame(self, key: str, frame: Any, extractor: Callable[..., Any]) -> None:
        """Register a new coordinate frame for lazy computation.

        Parameters
        ----------
        key : str
            Name for this coordinate system (e.g., 'galactic').
        frame : astropy frame or str
            Target frame for coordinate transforms.
        extractor : callable
            Function (skycoord_in_frame) -> (lon_rad, lat_rad).
        """
        self._frames[key] = (frame, extractor)

    def get_sky_coords(self, key: str) -> list[SkyCoord]:
        """Get sky hemisphere coordinates in the given frame (lazy, cached).

        Parameters
        ----------
        key : str
            Frame key (e.g., 'icrs', 'sref', or any registered frame).

        Returns
        -------
        list of SkyCoord
            One per observation, each with nsky points.
        """
        if key not in self._sky_cache:
            frame, _ = self._frames[key]
            self._sky_cache[key] = [
                SkyCoord(
                    self.geom.lon,
                    self.geom.lat,
                    unit="rad",
                    frame=af,
                ).transform_to(frame)
                for af in self.altaz_frames
            ]
        return self._sky_cache[key]

    def _sky_coords_jax(self, key: str, obs_idx: int) -> jax.Array:
        """Get sky coords as a jax array (nsky, 2) for a single observation."""
        coords = self.get_sky_coords(key)
        _, extractor = self._frames[key]
        lon, lat = extractor(coords[obs_idx])
        return jnp.stack([jnp.array(lon), jnp.array(lat)], axis=-1)

    def _pixel_coords_jax(self, key: str, result_altaz: SkyCoord) -> jax.Array:
        """Get pixel coords as a jax array (npix, 2) for a single observation."""
        frame, extractor = self._frames[key]
        result = result_altaz.transform_to(frame)
        lon, lat = extractor(result)
        return jnp.stack([jnp.array(lon), jnp.array(lat)], axis=-1)

    def get_render_geometry(self) -> list[RenderGeometry]:
        """Build a list of single-observation RenderGeometry.

        Returns
        -------
        list[RenderGeometry]
            One per observation. Each contains single-obs geometry with
            no leading nobs axis. Suitable for direct use in Scene.render()
            or for stacking/vmapping by the caller.
        """
        # AltAz sky coords (hemisphere grid)
        altaz_sky = jnp.stack(
            [jnp.array(self.geom.lon), jnp.array(self.geom.lat)],
            axis=-1,
        )

        geometries = []
        for i in range(self.nobs):
            sky = SkyGeometry(
                altaz_coord=altaz_sky,
                icrs_coord=self._sky_coords_jax("icrs", i),
                sref_coord=self._sky_coords_jax("sref", i),
                fov_altaz_grid=jnp.stack(
                    [jnp.array(self.fov_coords[i].az.rad), jnp.array(self.fov_coords[i].alt.rad)],
                    axis=-1,
                ),
                scattering_angle=jnp.array(self.scattering_angle[i]),
                height_km=jnp.array(self.height_km),
                hemisphere_mask=jnp.array(self.geom.mask),
            )

            geometries.append(
                RenderGeometry(
                    sky=sky,
                    pointing_matrix=jnp.array(self.pointing_matrices[i]),
                )
            )

        return geometries
