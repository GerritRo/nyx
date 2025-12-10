import warnings
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from .config import _config
from typing import Optional, Union
from enum import Enum
from dataclasses import dataclass

import numpy.lib.recfunctions as recfc
import astropy.units as u
import scipy.integrate as si
from astropy.utils.data import download_file
from astropy.io import fits, votable
from scipy.interpolate import UnivariateSpline, RegularGridInterpolator

SVO_TABLE_URL = "https://svo2.cab.inta-csic.es/theory/fps/fps.php?ID="
CALSPEC_URL = "https://archive.stsci.edu/hlsps/reference-atlases/cdbs/current_calspec/"
SOLAR_SPECTRUM_URL = "https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/solsys/solar_spec.fits"


class SpectralMethod(Enum):
    """Spectral interpolation methods"""
    LINEAR = 'linear'
    CONSERVE = 'conserve'


class Spectrum(eqx.Module):
    """
    Immutable container for spectral data with wavelengths and flux.

    This class uses equinox.Module for JAX compatibility, allowing it to be
    used with jit, vmap, grad, and other JAX transformations.

    Parameters
    ----------
    wavelengths : array, shape (M,)
        Wavelength grid (should be in consistent units, typically nm)
    flux : array, shape (M,) or (N, M)
        Flux values - single spectrum or batch of N spectra
    method : str, optional
        Default resampling method for this spectrum ('linear' or 'conserve')

    Examples
    --------
    >>> spec = Spectrum.from_solar()
    >>> resampled = spec.normalize_at(500.0).resample(wavelengths)
    """
    wavelengths: jnp.ndarray
    flux: jnp.ndarray
    method: str = eqx.field(static=True, default='conserve')

    def resample(self, wavelengths_out: jnp.ndarray,
                 method: Optional[str] = None) -> 'Spectrum':
        """
        Resample this spectrum to a new wavelength grid.

        Parameters
        ----------
        wavelengths_out : array, shape (K,)
            Target wavelength grid
        method : str, optional
            Resampling method ('linear' or 'conserve').
            If None, uses this spectrum's default method.

        Returns
        -------
        Spectrum
            New spectrum resampled to wavelengths_out
        """
        if method is None:
            method = self.method

        # Check for resolution degradation
        if _config.get('spectral_resolution_warning'):
            res_in = jnp.median(jnp.diff(self.wavelengths))
            res_out = jnp.median(jnp.diff(wavelengths_out))
            if res_out > 2 * res_in and method != 'conserve':
                warnings.warn(
                    f"Spectral resolution degraded from {res_in:.1f} to {res_out:.1f} nm. "
                    "Consider using 'conserve' method to preserve features.",
                    UserWarning
                )

        if method == 'linear':
            new_flux = _linear_interp(self.wavelengths, self.flux, wavelengths_out)
        elif method == 'conserve':
            new_flux = _conserve_interp(self.wavelengths, self.flux, wavelengths_out)
        else:
            raise ValueError(f"Unknown spectral method: {method}")

        return Spectrum(
            wavelengths=wavelengths_out,
            flux=new_flux,
            method=self.method
        )

    def normalize_at(self, wavelength: float) -> 'Spectrum':
        """
        Normalize this spectrum to unity at a given wavelength.

        Parameters
        ----------
        wavelength : float
            Wavelength at which to normalize (same units as self.wavelengths)

        Returns
        -------
        Spectrum
            New spectrum with flux normalized to 1.0 at the given wavelength
        """
        ref_value = jnp.interp(wavelength, self.wavelengths, self.flux)
        return Spectrum(
            wavelengths=self.wavelengths,
            flux=self.flux / ref_value,
            method=self.method
        )

    @classmethod
    def from_arrays(cls, wavelengths: jnp.ndarray, flux: jnp.ndarray,
                    method: str = 'conserve') -> 'Spectrum':
        """
        Create a Spectrum from wavelength and flux arrays.

        Parameters
        ----------
        wavelengths : array-like
            Wavelength values (units stripped if present)
        flux : array-like
            Flux values (units stripped if present)
        method : str, optional
            Default resampling method

        Returns
        -------
        Spectrum
            New Spectrum instance
        """
        # Handle astropy units
        if hasattr(wavelengths, 'value'):
            wavelengths = wavelengths.value
        if hasattr(flux, 'value'):
            flux = flux.value

        return cls(
            wavelengths=jnp.asarray(wavelengths),
            flux=jnp.asarray(flux),
            method=method
        )

    @classmethod
    def from_solar(cls, method: str = 'conserve') -> 'Spectrum':
        """
        Load the standard solar spectrum (Rieke 2008).

        Returns wavelengths in nm and flux in internal units.

        Returns
        -------
        Spectrum
            Solar spectrum instance
        """
        f_down = download_file(SOLAR_SPECTRUM_URL, cache=True)
        with fits.open(f_down) as hdul:
            wvl = np.ascontiguousarray(hdul[1].data['WAVELENGTH'] / 10.0, dtype=np.float64)
            flx = np.ascontiguousarray(hdul[1].data['FLUX'], dtype=np.float64)
        return cls(
            wavelengths=jnp.asarray(wvl),
            flux=jnp.asarray(flx),
            method=method
        )

def _linear_interp(wl_in: jnp.ndarray, flux_in: jnp.ndarray,
                   wl_out: jnp.ndarray) -> jnp.ndarray:
    """Simple linear interpolation, vectorized for multiple spectra."""
    if flux_in.ndim == 1:
        return jnp.interp(wl_out, wl_in, flux_in)
    elif flux_in.ndim == 2:
        # Vectorize along the first axis (N spectra)
        return jnp.stack(
            [jnp.interp(wl_out, wl_in, f) for f in flux_in],
            axis=0
        )
    else:
        raise ValueError("flux_in must be 1D or 2D")


def _conserve_interp(wl_in: jnp.ndarray, flux_in: jnp.ndarray,
                     wl_out: jnp.ndarray) -> jnp.ndarray:
    """
    Flux-conserving interpolation in JAX.

    Supports batched flux arrays of shape (N_spec, N_in).
    Preserves integrated flux when rebinning to coarser resolution.
    """
    def get_edges(wl):
        edges = jnp.zeros(len(wl) + 1)
        edges = edges.at[1:-1].set((wl[1:] + wl[:-1]) / 2)
        edges = edges.at[0].set(wl[0] - (wl[1] - wl[0]) / 2)
        edges = edges.at[-1].set(wl[-1] + (wl[-1] - wl[-2]) / 2)
        return edges

    edges_in = get_edges(wl_in)
    edges_out = get_edges(wl_out)

    delta_in = edges_in[1:] - edges_in[:-1]      # (N_in,)
    delta_out = edges_out[1:] - edges_out[:-1]   # (N_out,)

    # Compute all pairwise overlaps [N_out, N_in]
    overlap_lo = jnp.maximum(edges_out[:-1, None], edges_in[None, :-1])
    overlap_hi = jnp.minimum(edges_out[1:, None], edges_in[None, 1:])
    overlap = jnp.clip(overlap_hi - overlap_lo, 0, None)

    # Fraction of each input bin that overlaps each output bin
    frac = overlap / delta_in[None, :]           # (N_out, N_in)

    # Handle both 1D and 2D flux_in
    was_1d = flux_in.ndim == 1
    if was_1d:
        flux_in = flux_in[None, :]               # (1, N_in)

    # Multiply: (N_spec, N_in) * (N_out, N_in) -> broadcast to (N_spec, N_out, N_in)
    weighted = flux_in[:, None, :] * frac[None, :, :] * delta_in[None, None, :]

    # Integrate over input bins -> (N_spec, N_out)
    flux_out = jnp.sum(weighted, axis=-1) / delta_out[None, :]

    # If only one spectrum, return 1D array
    if was_1d:
        flux_out = flux_out[0]

    return flux_out


class Bandpass():
    def __init__(self, wvl, transmission):
        self.lam = wvl
        self.trx = transmission

        self.min = self.lam.min()
        self.max = self.lam.max()

        self.spline = UnivariateSpline(self.lam, self.trx, s=0, ext=1)
        self.vegazero = self._calculate_vega_zero()

    def __call__(self, lam):
        return self.spline(lam.to(self.lam.unit))

    def _calculate_vega_zero(self):
        f_down = download_file(CALSPEC_URL+'alpha_lyr_stis_011.fits', cache=True)
        hdul = fits.open(f_down)
        wvl = hdul[1].data['WAVELENGTH']*u.angstrom
        flx = hdul[1].data['FLUX']*u.erg/u.second/u.cm**2/u.angstrom

        return si.simpson(x=wvl, y=wvl*self(wvl)*flx)*u.erg/u.second/u.cm**2

    @classmethod
    def from_SVO(cls, filter_id, cache=True):
        f_down = download_file(SVO_TABLE_URL+filter_id, cache=cache)
        table = votable.parse_single_table(f_down)
        return cls(table.array.data['Wavelength']*u.angstrom, table.array.data['Transmission'])

    @classmethod
    def from_csv(cls, file):
        arr = np.genfromtxt(file, delimiter=",", names=True)
        lam = arr['wvl']*u.nm
        trx = recfc.drop_fields(arr, "wvl", usemask=False)
        return cls(lam, np.array(trx.tolist()).prod(axis=1))


@dataclass
class SpectralGrid:
    points: np.ndarray
    wvl: np.ndarray
    flx: np.ndarray

    def __call__(self, xi):
        if xi.size == 0:
            return self.flx
        else:
            rgi = RegularGridInterpolator(self.points, self.flx, bounds_error=False)
            return rgi(xi)*self.flx.unit

    def apply_bandpass(self, bandpass):
        mask = (self.wvl>=bandpass.min)&(self.wvl<=bandpass.max)

        wvl = self.wvl[mask]
        flx = np.einsum("a,...ab->...ab", bandpass(wvl), self.flx[..., mask, :])

        return SpectralGrid(self.points, wvl, flx)

    def __mul__(self, value):
        return SpectralGrid(self.points, self.wvl, np.einsum('...c,...->...c', self.flx, value))