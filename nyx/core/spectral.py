"""
Global Spectral Handler
"""
import warnings
import jax.numpy as jnp
import numpy as np
from .config import _config
from typing import Optional, Callable
from enum import Enum

import numpy.lib.recfunctions as recfc
import astropy.units as u
import scipy.integrate as si
from astropy.utils.data import download_file
from astropy.io import fits, votable
from scipy.interpolate import UnivariateSpline

SVO_TABLE_URL = "http://svo2.cab.inta-csic.es/theory/fps/fps.php?ID="
CALSPEC_URL = "https://archive.stsci.edu/hlsps/reference-atlases/cdbs/current_calspec/"

class SpectralMethod(Enum):
    """Spectral interpolation methods"""
    LINEAR = 'linear'
    CONSERVE = 'conserve'
    DRIZZLE = 'drizzle'
    CUBIC = 'cubic'


class SpectralHandler:
    """Handler for spectral interpolation and resampling"""
    
    @staticmethod
    def resample(wavelengths_in: jnp.ndarray, 
                flux_in: jnp.ndarray,
                wavelengths_out: jnp.ndarray,
                method: Optional[str] = None) -> jnp.ndarray:
        """
        Resample spectrum to new wavelength grid
        
        Parameters
        ----------
        wavelengths_in : Input wavelength grid
        flux_in : Input flux values
        wavelengths_out : Output wavelength grid
        method : Resampling method (uses global config if None)
        
        Returns
        -------
        Resampled flux values
        """
        if method is None:
            method = _config.get('spectral_method')
            
        # Check for resolution degradation
        if _config.get('spectral_resolution_warning'):
            res_in = jnp.median(jnp.diff(wavelengths_in))
            res_out = jnp.median(jnp.diff(wavelengths_out))
            if res_out > 2 * res_in and (method != 'conserve'):
                warnings.warn(
                    f"Spectral resolution degraded from {res_in:.1f} to {res_out:.1f} nm. "
                    "Consider using 'conserve' method to preserve features.",
                    UserWarning
                )
        
        if method == 'linear':
            return SpectralHandler._linear_interp(wavelengths_in, flux_in, wavelengths_out)
        elif method == 'conserve':
            return SpectralHandler._conserve_interp(wavelengths_in, flux_in, wavelengths_out)
        else:
            raise ValueError(f"Unknown spectral method: {method}")
    
    @staticmethod
    def _linear_interp(wl_in, flux_in, wl_out):
        """Simple linear interpolation"""
        return jnp.interp(wl_out, wl_in, flux_in)
    
    @staticmethod
    def _conserve_interp(wl_in, flux_in, wl_out):
        """
        Flux-conserving interpolation
        Integrates flux over input bins and redistributes to output bins
        """
        # Calculate bin edges
        def get_edges(wl):
            edges = jnp.zeros(len(wl) + 1)
            edges = edges.at[1:-1].set((wl[1:] + wl[:-1]) / 2)
            edges = edges.at[0].set(wl[0] - (wl[1] - wl[0]) / 2)
            edges = edges.at[-1].set(wl[-1] + (wl[-1] - wl[-2]) / 2)
            return edges
        
        edges_in = get_edges(wl_in)
        edges_out = get_edges(wl_out)
        
        flux_out = jnp.zeros(len(wl_out))
        
        # For each output bin, integrate overlapping input bins
        for i in range(len(wl_out)):
            out_lo, out_hi = edges_out[i], edges_out[i + 1]
            
            # Find overlapping input bins
            mask = (edges_in[:-1] < out_hi) & (edges_in[1:] > out_lo)
            
            # Calculate overlap fractions
            overlap_lo = jnp.maximum(edges_in[:-1], out_lo)
            overlap_hi = jnp.minimum(edges_in[1:], out_hi)
            overlap_frac = (overlap_hi - overlap_lo) / (edges_in[1:] - edges_in[:-1])
            
            # Sum weighted contributions
            flux_out = flux_out.at[i].set(
                jnp.sum(flux_in * overlap_frac * mask) * 
                (edges_in[1:] - edges_in[:-1]).mean() / (out_hi - out_lo)
            )
        
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
