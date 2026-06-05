from __future__ import annotations

import astropy.units as u
import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
from astropy.coordinates import (
    BaseCoordinateFrame,
    FunctionTransform,
    GeocentricTrueEcliptic,
    RepresentationMapping,
    SkyCoord,
    SphericalRepresentation,
    TimeAttribute,
    frame_transform_graph,
    get_body,
)
from jax.typing import ArrayLike

# HEALPix spatial index


class HEALPixCatalog:
    """Fast ra/dec cone search using HEALPix (nested ordering) spatial index."""

    def __init__(self, ra: np.ndarray, dec: np.ndarray, nside: int = 256):
        """
        Parameters
        ----------
        ra, dec : array-like, degrees
        nside : HEALPix nside (power of 2).
        """
        self.ra = np.asarray(ra, dtype=np.float64)
        self.dec = np.asarray(dec, dtype=np.float64)
        self.nside = nside

        # Convert to theta/phi (colatitude/longitude in radians)
        theta = np.radians(90.0 - self.dec)
        phi = np.radians(self.ra)

        # Assign each source to a nested pixel
        pix = hp.ang2pix(nside, theta, phi, nest=True)

        # Sort by pixel for contiguous slicing via searchsorted
        order = np.argsort(pix)
        self._sorted_pix = pix[order]
        self._sorted_idx = order

    def query(self, ra: float, dec: float, radius: float) -> np.ndarray:
        """
        Cone search.

        Parameters
        ----------
        ra, dec : center in degrees
        radius : search radius in degrees

        Returns
        -------
        indices : array of integer indices into the original catalog
        """
        theta_c = np.radians(90.0 - dec)
        phi_c = np.radians(ra)
        vec = hp.ang2vec(theta_c, phi_c)
        rad = np.radians(radius)

        # Get all HEALPix pixels overlapping the disc
        candidate_pix = hp.query_disc(self.nside, vec, rad, nest=True, inclusive=True)

        # Gather candidate source indices (includes edge-pixel extras)
        lefts = np.searchsorted(self._sorted_pix, candidate_pix, side="left")
        rights = np.searchsorted(self._sorted_pix, candidate_pix, side="right")
        idx_lists = [
            self._sorted_idx[left:right]
            for left, right in zip(lefts, rights, strict=False)
            if left < right
        ]
        if not idx_lists:
            return np.array([], dtype=np.int64)
        return np.concatenate(idx_lists)


# Healpix utility


def rotate_healpix(
    map_in: np.ndarray,
    frame_in: BaseCoordinateFrame,
    frame_out: BaseCoordinateFrame,
    nside_out: int | None = None,
) -> np.ndarray:
    """Rotate a HEALPix map between any two astropy coordinate frames."""
    nside = hp.get_nside(map_in)
    nside_out = nside_out or nside
    npix = hp.nside2npix(nside_out)

    theta, phi = hp.pix2ang(nside_out, np.arange(npix))
    lat = 90 - np.degrees(theta)
    lon = np.degrees(phi)

    coords_out = SkyCoord(lon * u.deg, lat * u.deg, frame=frame_out)
    coords_in = coords_out.transform_to(frame_in)

    theta_in = np.pi / 2 - coords_in.spherical.lat.rad
    phi_in = coords_in.spherical.lon.rad

    return hp.get_interp_val(map_in, theta_in, phi_in)


# Handrolled implementation of SkyOffsetFrame compatible with jax


def rotation_matrix_from_altaz(az_rad: float, alt_rad: float) -> np.ndarray:
    """Build rotation matrix R from an AltAz pointing direction.

    Parameters
    ----------
    az_rad, alt_rad : float
        Pointing direction in radians.

    Returns
    -------
    R : ndarray, shape (3, 3)
    """
    ca, sa = np.cos(az_rad), np.sin(az_rad)
    cd, sd = np.cos(alt_rad), np.sin(alt_rad)
    return np.array(
        [
            [cd * sa, cd * ca, sd],
            [ca, -sa, 0.0],
            [-sd * sa, -sd * ca, cd],
        ]
    )


def altaz_to_offset(az: ArrayLike, alt: ArrayLike, R: ArrayLike) -> tuple[jax.Array, jax.Array]:
    """Transform AltAz (az, alt) to offset frame (lon, lat) using R.

    Parameters
    ----------
    az, alt : array_like
        AltAz coordinates in radians.
    R : array_like, shape (3, 3)
        Rotation matrix from rotation_matrix_from_altaz.

    Returns
    -------
    lon, lat : jax.Array
        Offset frame coordinates in radians.
    """
    p = jnp.stack(
        [
            jnp.cos(alt) * jnp.sin(az),
            jnp.cos(alt) * jnp.cos(az),
            jnp.sin(alt),
        ],
        axis=-1,
    )
    R = jnp.asarray(R)
    p_local = jnp.einsum("ij,...j->...i", R, p)
    lon = jnp.arctan2(p_local[..., 1], p_local[..., 0])
    lat = jnp.arcsin(jnp.clip(p_local[..., 2], -1, 1))
    return lon, lat


def offset_to_altaz(lon: ArrayLike, lat: ArrayLike, R: ArrayLike) -> tuple[jax.Array, jax.Array]:
    """Transform offset frame (lon, lat) to AltAz (az, alt) using R.

    Inverse of altaz_to_offset.

    Parameters
    ----------
    lon, lat : array_like
        Offset frame coordinates in radians.
    R : array_like, shape (3, 3)
        Rotation matrix from rotation_matrix_from_altaz.

    Returns
    -------
    az, alt : jax.Array
        AltAz coordinates in radians.
    """
    p_local = jnp.stack(
        [
            jnp.cos(lat) * jnp.cos(lon),
            jnp.cos(lat) * jnp.sin(lon),
            jnp.sin(lat),
        ],
        axis=-1,
    )
    R = jnp.asarray(R)
    p = jnp.einsum("ij,...j->...i", R.T, p_local)

    eps = jnp.finfo(p.dtype).eps
    horiz_sq = p[..., 0] ** 2 + p[..., 1] ** 2
    at_pole = horiz_sq < 4 * eps

    safe_pz = jnp.where(at_pole, 0.0, p[..., 2])
    alt = jnp.where(
        at_pole,
        jnp.sign(p[..., 2]) * (jnp.pi / 2),
        jnp.arcsin(safe_pz),
    )

    safe_px = jnp.where(at_pole, 0.0, p[..., 0])
    safe_py = jnp.where(at_pole, 1.0, p[..., 1])
    az = jnp.where(at_pole, 0.0, jnp.arctan2(safe_px, safe_py))

    return az, alt


def cos_angular_separation_jax(
    az1: jax.Array, alt1: jax.Array, az2: jax.Array, alt2: jax.Array
) -> jax.Array:
    """JAX-compatible cosine of great-circle angular separation.

    Parameters
    ----------
    az1, alt1, az2, alt2 : jax.Array
        Coordinates in radians.

    Returns
    -------
    jax.Array
    """
    return jnp.sin(alt1) * jnp.sin(alt2) + jnp.cos(alt1) * jnp.cos(alt2) * jnp.cos(az1 - az2)


# Custom astropy coordinates


class SunRelativeEclipticFrame(BaseCoordinateFrame):
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
def gte_to_sunrel(
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
def sunrel_to_gte(
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
