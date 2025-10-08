import numpy as np
import healpy as hp

def rotate_healpy_map(hp_map, from_frame, to_frame, nest=True):
    """
    Rotate a HEALPix map or stacked maps from one astropy frame to another.

    Parameters
    ----------
    hp_map : array-like
        The input HEALPix map (shape (npix,) or (npix, N), RING or NESTED ordering).
    from_frame : astropy.coordinates.BaseCoordinateFrame
        The original astropy frame of the map.
    to_frame : astropy.coordinates.BaseCoordinateFrame
        The target astropy frame to rotate into.
    nside : int, optional
        The nside of the map. If None, inferred from hp_map length.
    nest : bool, default=False
        Whether the input and output maps are in NESTED ordering.

    Returns
    -------
    rotated_map : array
        The rotated HEALPix map(s), same shape and ordering as input.
    """
    hp_map = np.asarray(hp_map)

    if nside is None:
        nside = hp.npix2nside(hp_map.shape[0])

    npix = hp.nside2npix(nside)

    if hp_map.shape[0] != npix:
        raise ValueError(f"Input map first dimension {hp_map.shape[0]} does not match nside={nside}")

    # Get pixel centers in input ordering
    theta, phi = hp.pix2ang(nside, np.arange(npix), nest=nest)
    lon = np.degrees(phi)
    lat = 90.0 - np.degrees(theta)

    sky = SkyCoord(lon * u.deg, lat * u.deg, frame=from_frame)
    sky_new = sky.transform_to(to_frame)

    # Convert transformed coordinates back to theta, phi
    theta_new = np.radians(90.0 - sky_new.spherical.lat.deg)
    phi_new = np.radians(90.0 - sky_new.spherical.lon.deg)

    # Find corresponding pixels in the original map
    pix_new = hp.ang2pix(nside, theta_new, phi_new, nest=nest)

    # Apply remapping
    if hp_map.ndim == 1:
        rotated_map = hp_map[pix_new]
    elif hp_map.ndim == 2:
        rotated_map = hp_map[pix_new, :]
    else:
        raise ValueError("hp_map must be 1D or 2D with shape (npix,) or (npix, N)")

    return rotated_map