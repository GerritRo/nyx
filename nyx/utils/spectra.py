from dataclasses import dataclass
from typing import Any

import astropy.units as u
import jax.numpy as jnp
import numpy as np
import numpy.lib.recfunctions as recfc
import scipy.integrate as si
from astropy.constants import c, h
from astropy.io import fits, votable
from astropy.utils.data import download_file
from scipy.interpolate import RegularGridInterpolator, UnivariateSpline

from nyx import ASSETS_PATH
from nyx.core.spectral import _conserve_interp
from nyx.core.units import (
    energy_flux_to_photon_flux,
    to_wavelength_nm,
)

# URLs
SVO_TABLE_URL = "https://svo2.cab.inta-csic.es/theory/fps/fps.php?ID="
CALSPEC_URL = "https://archive.stsci.edu/hlsps/reference-atlases/cdbs/current_calspec/"
SOLAR_SPECTRUM_URL = (
    "https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/solsys/solar_spec.fits"
)


# Bandpass and SpectralGrid


class Bandpass:
    def __init__(self, wvl, transmission):
        self.lam = wvl
        self.trx = transmission
        self.min = self.lam.min()
        self.max = self.lam.max()
        self.spline = UnivariateSpline(self.lam, self.trx, s=0, ext=1)
        self._vegazero = None

    @property
    def vegazero(self):
        """Vega zero-point flux (lazy, downloads Vega spectrum on first access)."""
        if self._vegazero is None:
            self._vegazero = self._calculate_vega_zero()
        return self._vegazero

    def __call__(self, lam):
        return self.spline(lam.to(self.lam.unit))

    def _calculate_vega_zero(self):
        f_down = download_file(CALSPEC_URL + "alpha_lyr_stis_011.fits", cache=True)
        with fits.open(f_down) as hdul:
            wvl = hdul[1].data["WAVELENGTH"] * u.angstrom
            flx = hdul[1].data["FLUX"] * u.erg / u.second / u.cm**2 / u.angstrom
        z = wvl * self(wvl) * flx
        return si.simpson(y=z.value, x=wvl.value) * z.unit * wvl.unit

    @classmethod
    def from_SVO(cls, filter_id, cache=True):
        f_down = download_file(SVO_TABLE_URL + filter_id, cache=cache)
        table = votable.parse_single_table(f_down)
        return cls(table.array.data["Wavelength"] * u.angstrom, table.array.data["Transmission"])

    @classmethod
    def from_csv(cls, file):
        arr = np.genfromtxt(file, delimiter=",", names=True)
        lam = arr["wvl"] * u.nm
        trx = recfc.drop_fields(arr, "wvl", usemask=False)
        return cls(lam, np.array(trx.tolist()).prod(axis=1))


@dataclass
class SpectralGrid:
    points: Any
    wvl: Any
    flx: Any

    def __call__(self, xi):
        if xi.size == 0:
            return self.flx
        rgi = RegularGridInterpolator(self.points, self.flx, bounds_error=False)
        return rgi(xi) * self.flx.unit

    def apply_bandpass(self, bandpass):
        mask = (self.wvl >= bandpass.min) & (self.wvl <= bandpass.max)
        wvl = self.wvl[mask]
        flx = np.einsum("a,...ab->...ab", bandpass(wvl), self.flx[..., mask, :])
        return SpectralGrid(self.points, wvl, flx)

    def __mul__(self, value):
        return SpectralGrid(self.points, self.wvl, np.einsum("...c,...->...c", self.flx, value))


# Data-loading utilities


def _validate_spectrum_arrays(wavelengths, flux):
    if wavelengths.ndim != 1:
        raise ValueError(f"Wavelengths must be 1D, got shape {wavelengths.shape}")
    n_wvl = len(wavelengths)
    if flux.ndim == 1:
        if len(flux) != n_wvl:
            raise ValueError(f"Flux length ({len(flux)}) must match wavelengths ({n_wvl})")
    elif flux.ndim == 2:
        if flux.shape[-1] != n_wvl:
            raise ValueError(f"Flux last dim ({flux.shape[-1]}) must match wavelengths ({n_wvl})")
    else:
        raise ValueError(f"Flux must be 1D or 2D, got shape {flux.shape}")
    diffs = np.diff(np.asarray(wavelengths))
    if not np.all(diffs > 0):
        raise ValueError("Wavelengths must be strictly monotonically increasing")


def load_solar_flux(wvl_out, normalize_at=None):
    """Load the HST/CALSPEC solar spectrum, resample to *wvl_out*.

    Parameters
    ----------
    wvl_out : array
        Target wavelengths in nm.
    normalize_at : float or None
        Wavelength (nm) at which flux is set to 1.0.
    """
    f_down = download_file(SOLAR_SPECTRUM_URL, cache=True)
    with fits.open(f_down) as hdul:
        wvl = np.ascontiguousarray(hdul[1].data["WAVELENGTH"] / 10.0, dtype=np.float64)
        flx = np.ascontiguousarray(hdul[1].data["FLUX"], dtype=np.float64)
    wvl_arr = jnp.asarray(wvl)
    flx_arr = jnp.asarray(flx)
    if normalize_at is not None:
        ref = jnp.interp(normalize_at, wvl_arr, flx_arr)
        flx_arr = flx_arr / ref
    return _conserve_interp(wvl_arr, flx_arr, wvl_out)


def prepare_flux(wavelengths, flux, wvl_out, validate=True, from_energy=False):
    """Convert and resample arbitrary flux data to a target wavelength grid.

    Parameters
    ----------
    wavelengths : array or Quantity
        Source wavelengths.
    flux : array or Quantity
        Source flux.
    wvl_out : array
        Target wavelengths in nm.
    validate : bool
        Validate input arrays.
    from_energy : bool
        If True, convert energy flux to photon flux.
    """
    if from_energy:
        if isinstance(wavelengths, u.Quantity):
            wvl_arr = jnp.asarray(wavelengths.to(u.nm).value)
        else:
            wvl_arr = jnp.asarray(wavelengths)
        flx_arr = energy_flux_to_photon_flux(wvl_arr, flux)
    else:
        wvl_arr = to_wavelength_nm(wavelengths)
        flx_arr = jnp.asarray(flux.value if hasattr(flux, "value") else flux)
    if validate:
        _validate_spectrum_arrays(wvl_arr, flx_arr)
    return _conserve_interp(wvl_arr, flx_arr, wvl_out)


# Spectral atlas loaders


def PicklesTRDSAtlas1998():
    """Load the Pickles (1998) TRDS stellar spectrum atlas."""
    file = np.genfromtxt(ASSETS_PATH + "pickles1998_trds_atlas.dat")
    return SpectralGrid(
        np.array([]), file[0] * u.angstrom, file[1:].T * u.erg / u.angstrom / u.s / u.cm**2
    )


def SolarSpectrumRieke2008():
    """Load the Rieke (2008) solar spectrum from STScI CALSPEC."""
    f_down = download_file(
        "https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/solsys/solar_spec.fits",
        cache=True,
    )
    with fits.open(f_down) as hdul:
        wvl = hdul[1].data["WAVELENGTH"] * u.angstrom
        flx = hdul[1].data["FLUX"] * u.erg / u.s / u.cm**2 / u.angstrom
    return wvl, flx


def create_color_grid(
    magnitude, color, color_range, spec_library, EBV_range=None, extmod=None, photon_flux=False
):
    """Create a synthetic color grid with dust reddening.

    Parameters
    ----------
    magnitude : Bandpass
        Bandpass for the magnitude system.
    color : tuple of Bandpass
        Two bandpasses defining the color (color[0] - color[1]).
    color_range : tuple
        (min, max) of the color range to interpolate over.
    spec_library : SpectralGrid
        Spectral library to redden.
    EBVs : array
        E(B-V) values to sample.
    extmod : extinction model, optional
        Dust extinction model. Defaults to G23(Rv=3.1).
    photon_flux : bool
        If True, convert output to photon flux units (photon/m^2/s/nm).
        If False, return in energy flux units divided by photon energy.
    """
    if EBV_range is None:
        EBV_range = [0, 10]

    EBVs = np.linspace(float(EBV_range[0]), float(EBV_range[1]), 20)
    if extmod is None:
        from dust_extinction.parameter_averages import G23

        extmod = G23(Rv=3.1)

    def calculate_magnitude(wvl, flx, bandpass):
        z = flx * bandpass(wvl) * wvl
        integral = si.simpson(y=z.value, x=wvl.value) * z.unit * wvl.unit
        return -2.5 * np.log10(integral / bandpass.vegazero)

    def redden_by_dust_extinction(EBVs_arr):
        wvl = spec_library.wvl
        flx = spec_library.flx.T
        flx = flx[:, np.newaxis, :] * extmod.extinguish(wvl, Ebv=EBVs_arr[..., np.newaxis])
        mag_corr = calculate_magnitude(wvl, flx, magnitude)
        return wvl, 10 ** (0.4 * mag_corr[..., np.newaxis]) * flx

    wvl, flx = redden_by_dust_extinction(EBVs)
    synth_color = calculate_magnitude(wvl, flx, color[0]) - calculate_magnitude(wvl, flx, color[1])

    color_space = np.linspace(float(color_range[0]), float(color_range[1]), 51)
    ebv_interp = np.zeros((len(synth_color), len(color_space)))
    for i, color_arr in enumerate(synth_color):
        c_sort = np.argsort(color_arr)
        ebv_interp[i] = np.interp(
            color_space, color_arr[c_sort], EBVs[c_sort], left=np.nan, right=np.nan
        )

    wvl, flx = redden_by_dust_extinction(ebv_interp)

    photon_energy = h * c / wvl[:, np.newaxis, np.newaxis]
    if photon_flux:
        flx = (flx.T / photon_energy).to(1 / (u.m**2 * u.s * u.nm)).value
    else:
        flx = flx.T / photon_energy

    return SpectralGrid(np.array([color_space]), wvl, np.transpose(flx, [1, 0, 2]))
