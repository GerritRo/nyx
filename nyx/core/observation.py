from __future__ import annotations

from typing import Any

import astropy.units as u
import jax
import jax.numpy as jnp
import numpy as np
from astropy.coordinates import (
    AltAz,
    BaseCoordinateFrame,
    EarthLocation,
    FunctionTransform,
    GeocentricTrueEcliptic,
    RepresentationMapping,
    SkyCoord,
    SphericalRepresentation,
    TimeAttribute,
    frame_transform_graph,
    get_body,
)
from astropy.time import Time

from nyx.core.coordinates import offset_to_altaz, rotation_matrix_from_altaz
from nyx.core.geometry import Geometry
from nyx.core.records import PerObs, RenderGeometry, SkyGeometry

__all__ = [
    "Observation",
    "SunRelativeEclipticFrame",
]


class SunRelativeEclipticFrame(BaseCoordinateFrame):
    """Ecliptic frame with longitude measured from the Sun.

    Coordinates are ``alpha`` (longitude, wrapped to +-180 deg) and ``beta``
    (latitude).
    """

    default_representation = SphericalRepresentation
    obstime = TimeAttribute(default=None)

    frame_specific_representation_info = {
        SphericalRepresentation: [
            RepresentationMapping("lon", "alpha"),
            RepresentationMapping("lat", "beta"),
            RepresentationMapping("distance", "distance"),
        ]
    }


@frame_transform_graph.transform(
    FunctionTransform, GeocentricTrueEcliptic, SunRelativeEclipticFrame
)
def _gte_to_sunrel(
    gte_coords: GeocentricTrueEcliptic, sunrel_frame: SunRelativeEclipticFrame
) -> SunRelativeEclipticFrame:
    obstime = gte_coords.obstime
    if obstime is None:
        raise ValueError("GeocentricTrueEcliptic coords must have obstime")
    sun = get_body("sun", obstime)
    sun_ecl = sun.transform_to(GeocentricTrueEcliptic(obstime=obstime))
    alpha = (gte_coords.lon - sun_ecl.lon).wrap_at(180 * u.deg)
    beta = gte_coords.lat
    distance = gte_coords.distance if gte_coords.distance.unit != u.one else None
    return SunRelativeEclipticFrame(alpha=alpha, beta=beta, distance=distance, obstime=obstime)


@frame_transform_graph.transform(
    FunctionTransform, SunRelativeEclipticFrame, GeocentricTrueEcliptic
)
def _sunrel_to_gte(
    sunrel_coords: SunRelativeEclipticFrame, gte_frame: GeocentricTrueEcliptic
) -> GeocentricTrueEcliptic:
    obstime = sunrel_coords.obstime
    if obstime is None:
        raise ValueError("SunRelativeEclipticFrame must have obstime")
    sun_ecl = get_body("sun", obstime).transform_to(GeocentricTrueEcliptic(obstime=obstime))
    lon = (sun_ecl.lon + sunrel_coords.alpha).wrap_at(360 * u.deg)
    lat = sunrel_coords.beta
    distance = sunrel_coords.distance if sunrel_coords.distance.unit != u.one else None
    return GeocentricTrueEcliptic(lon=lon, lat=lat, distance=distance, obstime=obstime)


def _format_span(seconds: float) -> str:
    """A duration in the largest unit that still reads under 120.

    Parameters
    ----------
    seconds : float

    Returns
    -------
    str
    """
    for value, unit in ((seconds, "s"), (seconds / 60.0, "min")):
        if float(f"{value:.4g}") < 120.0:
            return f"{value:.4g} {unit}"
    return f"{seconds / 3600.0:.4g} h"


def _extract_icrs(skycoord: SkyCoord) -> tuple[np.ndarray, np.ndarray]:
    """Extract (lon, lat) = (ra, dec) in radians from ICRS."""
    return skycoord.ra.rad, skycoord.dec.rad


def _extract_sref(skycoord: SkyCoord) -> tuple[np.ndarray, np.ndarray]:
    """Extract (lon, lat) = (alpha, beta) in radians from SunRelativeEcliptic."""
    return skycoord.alpha.rad, skycoord.beta.rad


# Registry of known frames
_BUILTIN_FRAMES = {
    "icrs": ("icrs", _extract_icrs),
    "sref": (SunRelativeEclipticFrame(), _extract_sref),
}


class Observation:
    """One target and one location, observed at ``nobs = len(times)`` times.

    Parameters
    ----------
    location : EarthLocation
    times : Time
        Observation times.
    target : SkyCoord
        Scalar target, shared across all times.
    geo : Geometry
        Resolution configuration.
    refract_pointing : bool, optional
        Whether the pointing direction is the apparent position rather than
        the geometric AltAz.
    **kwargs
        Further AltAz frame parameters, e.g. pressure and temperature.

    Raises
    ------
    TypeError
        If *target* is not a scalar SkyCoord.
    """

    def __init__(
        self,
        location: EarthLocation,
        times: Time,
        target: SkyCoord,
        geo: Geometry,
        refract_pointing: bool = False,
        **kwargs: Any,
    ) -> None:
        if isinstance(target, list):
            raise TypeError("target must be a single SkyCoord, not a list.")
        if not target.isscalar:
            raise TypeError("target must be a scalar SkyCoord.")

        self.geo = geo
        self.times = times
        self.target_icrs = target
        self.nobs = len(times)

        self.location = location
        self._altaz_kwargs = kwargs
        self._refract_pointing = refract_pointing

        self.altaz_frames = [AltAz(location=self.location, obstime=t, **kwargs) for t in times]
        pointing_kwargs = kwargs if refract_pointing else {}
        pointing_frames = [
            AltAz(location=self.location, obstime=t, **pointing_kwargs) for t in times
        ]
        self.pointing_matrices, self._target_alt_deg = self._build_pointing(
            self.target_icrs, pointing_frames
        )

        self.fov_coords = self._build_fov_coords(geo, self.pointing_matrices, self.altaz_frames)

        self._sky_cache: dict[str, list[SkyCoord]] = {}

    def _repr_when(self) -> str:
        """How many frames, from when, over how long."""
        start = self.times[0].utc.isot.removesuffix(".000")
        if self.nobs < 2:
            return f"1 time, {start}"
        seconds = float((self.times[-1] - self.times[0]).to_value("s"))
        return f"{self.nobs} times, {start} +{_format_span(seconds)}"

    def _repr_target(self) -> str:
        """Where the target is on the sky, and whether it is actually up."""
        icrs = self.target_icrs.icrs
        alt = self._target_alt_deg
        lo, hi = float(alt.min()), float(alt.max())
        where = f"alt={lo:.2f} deg" if lo == hi else f"alt {lo:.2f} to {hi:.2f} deg"
        if hi < 0.0:
            where += " (below horizon)"
        elif lo < 0.0:
            where += " (partly below horizon)"
        return f"ra={icrs.ra.deg:.4f} dec={icrs.dec.deg:.4f} deg, {where}"

    def _repr_site(self) -> str:
        """Where the observer is."""
        parts = [
            f"lon={self.location.lon.deg:.4f} lat={self.location.lat.deg:.4f} deg",
            f"height={self.height_km:.3g} km",
        ]
        if self._refract_pointing:
            parts.append("refracted pointing")
        parts += [f"{key}={value}" for key, value in self._altaz_kwargs.items()]
        return ", ".join(parts)

    def __repr__(self) -> str:
        """When, where and at what geometry, one per line."""
        return "\n".join(
            [
                f"Observation({self._repr_when()})",
                f"  target  {self._repr_target()}",
                f"  site    {self._repr_site()}",
                f"  geo     {self.geo!r}",
            ]
        )

    @staticmethod
    def _build_pointing(
        target_icrs: SkyCoord, altaz_frames: list[AltAz]
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Per-observation pointing matrices and the target altitudes behind them.

        Parameters
        ----------
        target_icrs : SkyCoord
            Scalar target.
        altaz_frames : list of AltAz
            One frame per observation.

        Returns
        -------
        matrices : list of ndarray, each shape (3, 3)
        alt_deg : ndarray, shape (nobs,)
            Target altitude in each frame, kept for :meth:`__repr__`.
        """
        targets = [target_icrs.transform_to(af) for af in altaz_frames]
        matrices = [rotation_matrix_from_altaz(t.az.rad, t.alt.rad) for t in targets]
        return matrices, np.array([t.alt.deg for t in targets])

    @staticmethod
    def _build_fov_coords(
        geo: Geometry, pointing_matrices: list[np.ndarray], altaz_frames: list[AltAz]
    ) -> list[SkyCoord]:
        """FOV grid coordinates in AltAz, one SkyCoord per observation."""
        fov_coords = []
        for R, af in zip(pointing_matrices, altaz_frames, strict=True):
            az, alt = offset_to_altaz(geo.X, geo.Y, R)
            fov_coords.append(SkyCoord(az=az, alt=alt, unit="rad", frame=af))
        return fov_coords

    @property
    def height_km(self) -> float:
        """Observatory height in km."""
        return self.location.height.to("km").value

    def get_sky_coords(self, key: str) -> list[SkyCoord]:
        """Hemisphere coordinates in the given frame, computed once and cached.

        Parameters
        ----------
        key : str
            Frame key: ``'icrs'`` or ``'sref'``.

        Returns
        -------
        list of SkyCoord
            One per observation, each with nsky points.
        """
        if key not in self._sky_cache:
            frame, _ = _BUILTIN_FRAMES[key]
            self._sky_cache[key] = [
                SkyCoord(
                    self.geo.lon,
                    self.geo.lat,
                    unit="rad",
                    frame=af,
                ).transform_to(frame)
                for af in self.altaz_frames
            ]
        return self._sky_cache[key]

    def _sky_coords_jax(self, key: str, obs_idx: int) -> jax.Array:
        """Sky coordinates of one observation as a ``(nsky, 2)`` jax array."""
        coords = self.get_sky_coords(key)
        _, extractor = _BUILTIN_FRAMES[key]
        lon, lat = extractor(coords[obs_idx])
        return jnp.stack([jnp.array(lon), jnp.array(lat)], axis=-1)

    def get_render_geometry(self) -> list[RenderGeometry]:
        """Build the per-observation render geometries.

        Returns
        -------
        list of RenderGeometry
            One per observation, with no leading ``nobs`` axis.
        """
        altaz_sky = jnp.stack(
            [jnp.array(self.geo.lon), jnp.array(self.geo.lat)],
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
                height_km=jnp.array(self.height_km),
                hemisphere_mask=jnp.array(self.geo.mask),
            )

            geometries.append(
                RenderGeometry(
                    sky=PerObs(sky),
                    pointing_matrix=PerObs(jnp.array(self.pointing_matrices[i])),
                )
            )

        return geometries
