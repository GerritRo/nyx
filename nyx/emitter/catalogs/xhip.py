from __future__ import annotations

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time

from nyx import ASSETS_PATH
from nyx.core.geometry import Geometry
from nyx.utils.spectra import (
    Bandpass,
    PicklesTRDSAtlas1998,
    SpectralModel,
    color_grid_spectrum,
    create_color_grid,
)

__all__ = ["EPOCH", "load_anderson2012", "spectral_model"]

#: Epoch the XHIP positions and proper motions are given for.
EPOCH = "J2000"


def load_anderson2012() -> tuple[SkyCoord, np.ndarray]:
    """Positions and photometry from the XHIP compilation (Anderson & Francis 2012).

    Returns
    -------
    coords : astropy.coordinates.SkyCoord
        Catalog positions with proper motions, at :data:`EPOCH`.
    photometry : numpy.ndarray, shape (n_stars, 2)
        Per-star ``[v_mag, v_minus_b]``.  Positive colour is bluer, matching
        the RP - BP convention of the Gaia path.
    """
    xhip = np.genfromtxt(
        ASSETS_PATH + "anderson2012_xhip_suppl.dat",
        skip_header=3,
        delimiter=",",
        names=True,
    )
    coords = SkyCoord(
        ra=xhip["RAJ2000"] * u.deg,
        dec=xhip["DEJ2000"] * u.deg,
        pm_ra_cosdec=xhip["pmRA"] * u.mas / u.yr,
        pm_dec=xhip["pmDE"] * u.mas / u.yr,
        obstime=Time(EPOCH),
        frame="icrs",
    )
    v_minus_b = xhip["Vmag"] - xhip["Bmag"]
    return coords, np.column_stack([xhip["Vmag"], v_minus_b])


def spectral_model(v_minus_b: np.ndarray, geo: Geometry) -> SpectralModel:
    """Pickles (1998) spectra indexed by the Johnson V-B colour.

    Needs network access on first use, for the Johnson passbands.

    Parameters
    ----------
    v_minus_b : numpy.ndarray
        Catalog colours; only their range is used, to size the colour axis.
    geo : Geometry

    Returns
    -------
    SpectralModel
        Reads conditions ``[v_mag, v_minus_b, active]``.
    """
    V = Bandpass.from_SVO("OSN/Johnson.V")
    B = Bandpass.from_SVO("OSN/Johnson.B")

    # Span exactly the colours present; the grid returns zero below its lower
    # edge, which would silently drop a star.
    spec_grid = create_color_grid(
        V,
        (V, B),
        [v_minus_b.min(), v_minus_b.max()],
        PicklesTRDSAtlas1998(),
        photon_flux=True,
    )
    return color_grid_spectrum(
        spec_grid,
        geo.wvls,
        color_fn=lambda c: c[..., 1],
        mag_fn=lambda c: c[..., 0],
        active_fn=lambda c: c[..., 2],
    )
