from __future__ import annotations

import astropy.units as u
import healpy as hp
import numpy as np
from numpy.typing import ArrayLike

from nyx.core.units import to_angle_rad, to_wavelength_nm


class Geometry:
    """Resolution and grid configuration for rendering.

    Parameters
    ----------
    wvls : array-like or astropy Quantity
        Wavelength grid (converted to nm).
    nside : int
        HEALPix nside (power of 2) for hemisphere discretisation.
    ngrid : int
        Number of grid points per axis over the FOV for in-scattering.
    fov : float or astropy Quantity
        Field of view half-angle (converted to radians).
    """

    def __init__(
        self,
        wvls: ArrayLike | u.Quantity,
        nside: int,
        ngrid: int,
        fov: float | u.Quantity,
    ) -> None:
        self.wvls = to_wavelength_nm(wvls)
        self.nside = nside
        self.ngrid = ngrid
        self.fov = to_angle_rad(fov)

        # HEALPix hemisphere grid
        npix = hp.nside2npix(nside)
        theta, phi = hp.pix2ang(nside, np.arange(npix))
        self.mask = theta < np.pi / 2
        self.lon = phi[self.mask]  # azimuth, rad
        self.lat = np.pi / 2 - theta[self.mask]  # altitude, rad
        self.nsky = int(np.sum(self.mask))

        # FOV evaluation grid
        grid_1d = np.linspace(-self.fov, self.fov, ngrid)
        self.X, self.Y = np.meshgrid(grid_1d, grid_1d)

        self.pixel_area = hp.nside2pixarea(nside)
