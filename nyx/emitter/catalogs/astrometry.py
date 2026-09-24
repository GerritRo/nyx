from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import astropy.units as u
import healpy as hp
import numpy as np
from astropy.coordinates import BaseCoordinateFrame, SkyCoord
from astropy.time import Time
from erfa import ErfaWarning

__all__ = [
    "altaz_array",
    "altaz_track",
    "angular_separation_deg",
    "propagate_space_motion",
    "rotate_healpix",
]


def angular_separation_deg(
    ra: float, dec: float, ra_arr: np.ndarray, dec_arr: np.ndarray
) -> np.ndarray:
    """Great-circle separation in degrees.

    Parameters
    ----------
    ra, dec : float
        Query position in degrees.
    ra_arr, dec_arr : numpy.ndarray
        Catalog positions in degrees.

    Returns
    -------
    numpy.ndarray
    """
    lon1, lat1 = np.radians(ra), np.radians(dec)
    lon2 = np.radians(np.asarray(ra_arr, dtype=np.float64))
    lat2 = np.radians(np.asarray(dec_arr, dtype=np.float64))

    sdlon, cdlon = np.sin(lon2 - lon1), np.cos(lon2 - lon1)
    slat1, clat1 = np.sin(lat1), np.cos(lat1)
    slat2, clat2 = np.sin(lat2), np.cos(lat2)

    num1 = clat2 * sdlon
    num2 = clat1 * slat2 - slat1 * clat2 * cdlon
    denom = slat1 * slat2 + clat1 * clat2 * cdlon
    return np.degrees(np.arctan2(np.hypot(num1, num2), denom))


def propagate_space_motion(coord: SkyCoord, time: Time) -> SkyCoord:
    """Move *coord* to *time* along its proper motion, position only.

    Parameters
    ----------
    coord : astropy.coordinates.SkyCoord
        Catalog position, with or without velocity differentials.
    time : astropy.time.Time
        Epoch to propagate to.

    Returns
    -------
    astropy.coordinates.SkyCoord
        ICRS position at *time*, with no differentials.  Returned unchanged
        if *coord* carries no proper motion or no reference epoch.
    """
    icrs = coord.icrs
    if getattr(icrs, "obstime", None) is None or not getattr(icrs.data, "differentials", None):
        return coord
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ErfaWarning)
        moved = icrs.apply_space_motion(new_obstime=time)
    return SkyCoord(ra=moved.ra, dec=moved.dec, frame="icrs")


def rotate_healpix(
    map_in: np.ndarray,
    frame_in: BaseCoordinateFrame,
    frame_out: BaseCoordinateFrame,
) -> np.ndarray:
    """Rotate a HEALPix map between any two astropy coordinate frames.

    Parameters
    ----------
    map_in : ndarray
        HEALPix map, RING ordering.
    frame_in, frame_out : astropy frame
        Input and target frame for the map.

    Returns
    -------
    ndarray
        The map resampled onto *frame_out*, at the input resolution.
    """
    nside = hp.get_nside(map_in)
    npix = hp.nside2npix(nside)

    theta, phi = hp.pix2ang(nside, np.arange(npix))
    lat = 90 - np.degrees(theta)
    lon = np.degrees(phi)

    coords_out = SkyCoord(lon * u.deg, lat * u.deg, frame=frame_out)
    coords_in = coords_out.transform_to(frame_in)

    theta_in = np.pi / 2 - coords_in.spherical.lat.rad
    phi_in = coords_in.spherical.lon.rad

    return hp.get_interp_val(map_in, theta_in, phi_in)


def altaz_array(coord: SkyCoord) -> np.ndarray:
    """AltAz pairs as a plain array.

    Parameters
    ----------
    coord : astropy.coordinates.SkyCoord
        Already in an AltAz frame.

    Returns
    -------
    numpy.ndarray, shape (n_src, 2)
        ``(az, alt)`` in radians.
    """
    az = np.atleast_1d(np.asarray(coord.az.rad, dtype=float))
    alt = np.atleast_1d(np.asarray(coord.alt.rad, dtype=float))
    return np.column_stack([az, alt])


def altaz_track(coord: SkyCoord, times: Time, frames: Sequence[Any]) -> np.ndarray:
    """Where *coord* is, in each observation's AltAz frame.

    Proper motion is applied per epoch.

    Parameters
    ----------
    coord : astropy.coordinates.SkyCoord
        Scalar or array of catalog positions.
    times : astropy.time.Time
        One epoch per observation.
    frames : sequence of astropy.coordinates.AltAz
        One frame per observation, matching *times*.

    Returns
    -------
    numpy.ndarray, shape (nobs, n_src, 2)
        ``(az, alt)`` in radians.
    """
    n_src = 1 if coord.isscalar else len(coord)
    if n_src == 0:
        return np.zeros((len(times), 0, 2))
    return np.stack(
        [
            altaz_array(propagate_space_motion(coord, t).transform_to(f))
            for t, f in zip(times, frames, strict=True)
        ]
    )
