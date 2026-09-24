from __future__ import annotations

import healpy as hp
import numpy as np
from astropy.utils.data import download_file

from nyx.core.geometry import Geometry
from nyx.spectra import (
    Bandpass,
    SpectralGrid,
    SpectralModel,
    color_grid_spectrum,
    create_color_grid,
)

__all__ = ["EPOCH", "catalog_map", "color_grid", "load_dr3", "photometry", "spectral_model"]

#: Reference epoch of the Gaia DR3 astrometry.
EPOCH = "J2016.0"

#: Magnitude standing in for a catalog row with no photometry at all.
_NO_PHOTOMETRY_MAG = 99.0

_CATALOG_URL = "https://zenodo.org/records/15396676/files/gaiadr3.npy"
_FAINT_MAP_URL = "https://zenodo.org/records/15396676/files/gaia_mag15plus.npy"


def load_dr3() -> tuple[np.ndarray, np.ndarray]:
    """Download (once) and load the Gaia DR3 catalog and faint-star map."""
    catalog = np.load(download_file(_CATALOG_URL, cache=True))
    faint_map = np.load(download_file(_FAINT_MAP_URL, cache=True))
    return catalog, faint_map


def photometry(catalog: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(G, BP, RP)`` magnitudes, with missing BP/RP filled by median colour.

    Parameters
    ----------
    catalog : numpy.ndarray
        Structured catalog array.

    Returns
    -------
    g, bp, rp : numpy.ndarray
        One finite entry per catalog row.

    Raises
    ------
    ValueError
        If no source has both BP and RP measured.
    """
    g = np.asarray(catalog["phot_g_mean_mag"], dtype=float)
    bp = np.asarray(catalog["phot_bp_mean_mag"], dtype=float)
    rp = np.asarray(catalog["phot_rp_mean_mag"], dtype=float)

    measured = np.isfinite(bp) & np.isfinite(rp)
    if not measured.any():
        raise ValueError("catalog contains no source with both BP and RP measured")
    median_color = float(np.median((rp - bp)[measured]))

    g = np.where(np.isfinite(g), g, _NO_PHOTOMETRY_MAG)
    bp = np.where(measured, bp, g - median_color / 2)
    rp = np.where(measured, rp, g + median_color / 2)
    return g, bp, rp


def catalog_map(
    g: np.ndarray,
    bp: np.ndarray,
    rp: np.ndarray,
    ra: np.ndarray,
    dec: np.ndarray,
    npix: int,
) -> np.ndarray:
    """Bin catalog stars into a HEALPix linear-flux map.

    Returns
    -------
    numpy.ndarray, shape (3, npix)
        ``[G, BP, RP]``, nested ordering.
    """
    map_nside = hp.npix2nside(npix)
    hp_inds = hp.ang2pix(map_nside, ra, dec, nest=True, lonlat=True)
    return np.vstack([np.bincount(hp_inds, 10 ** (-0.4 * mag), npix) for mag in (g, bp, rp)])


def color_grid(bp: np.ndarray, rp: np.ndarray) -> SpectralGrid:
    """Reddened Pickles (1998) spectra indexed by the Gaia ``RP - BP`` colour.

    Parameters
    ----------
    bp, rp : numpy.ndarray
        Catalog BP and RP magnitudes; only their range is used, to size the
        colour axis of the grid.

    Returns
    -------
    SpectralGrid
    """
    G = Bandpass.from_SVO("GAIA/GAIA3.G")
    BP = Bandpass.from_SVO("GAIA/GAIA3.Gbp")
    RP = Bandpass.from_SVO("GAIA/GAIA3.Grp")

    return create_color_grid(
        G,
        (RP, BP),
        [float(np.min(rp - bp)), 0.5],
        SpectralGrid.from_pickles1998(),
        photon_flux=True,
    )


def spectral_model(grid: SpectralGrid, geo: Geometry, horizon_mask: bool = False) -> SpectralModel:
    """Wrap a :func:`color_grid` as a JAX model over Gaia conditions.

    Parameters
    ----------
    grid : SpectralGrid
        From :func:`color_grid`.
    geo : Geometry
    horizon_mask : bool
        Whether to read a fourth condition column as a 0/1 switch.

    Returns
    -------
    SpectralModel
        Reads conditions ``[G, BP, RP]``, plus ``active`` when *horizon_mask*.
    """
    return color_grid_spectrum(
        grid,
        geo.wvls,
        color_fn=lambda c: c[..., 2] - c[..., 1],  # RP - BP
        mag_fn=lambda c: c[..., 0],  # G
        active_fn=(lambda c: c[..., 3]) if horizon_mask else None,
    )
