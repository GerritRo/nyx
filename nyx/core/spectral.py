import warnings
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from .config import _config
from typing import Optional, Union, TYPE_CHECKING
from enum import Enum
from dataclasses import dataclass

import numpy.lib.recfunctions as recfc
import astropy.units as u
import scipy.integrate as si
from astropy.utils.data import download_file
from astropy.io import fits, votable
from scipy.interpolate import UnivariateSpline, RegularGridInterpolator
from astropy.constants import c, h

if TYPE_CHECKING:
    from nyx.units import Wavelength, Flux, Radiance

# Internal unit specifications
RADIANCE = u.photon / (u.s * u.m**2 * u.nm * u.sr)
FLUX = u.photon / (u.s * u.m**2 * u.nm)

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
                    method: str = 'conserve', validate: bool = True) -> 'Spectrum':
        """
        Create a Spectrum from wavelength and flux arrays.

        Parameters
        ----------
        wavelengths : array-like, Wavelength, or astropy Quantity
            Wavelength values. Accepts:
            - Raw arrays (assumed to be in nm)
            - nyx.units.Wavelength objects
            - astropy Quantity with wavelength units
        flux : array-like, Flux, Radiance, or astropy Quantity
            Flux values. Accepts:
            - Raw arrays (assumed to be in internal units)
            - nyx.units.Flux or nyx.units.Radiance objects
            - astropy Quantity (units stripped)
        method : str, optional
            Default resampling method
        validate : bool, optional
            Whether to validate wavelength monotonicity and dimension matching.
            Default is True.

        Returns
        -------
        Spectrum
            New Spectrum instance
        """
        # Import here to avoid circular imports
        from nyx.units import Wavelength as WavelengthType, Flux as FluxType, Radiance as RadianceType

        # Handle wavelength extraction
        if isinstance(wavelengths, WavelengthType):
            wvl_arr = wavelengths.value
        elif isinstance(wavelengths, u.Quantity):
            wvl_arr = wavelengths.to(u.nm).value
        elif hasattr(wavelengths, 'value'):
            wvl_arr = wavelengths.value
        else:
            wvl_arr = wavelengths

        # Handle flux extraction
        if isinstance(flux, (FluxType, RadianceType)):
            flx_arr = flux.value
        elif hasattr(flux, 'value'):
            flx_arr = flux.value
        else:
            flx_arr = flux

        wvl_arr = jnp.asarray(wvl_arr)
        flx_arr = jnp.asarray(flx_arr)

        if validate:
            _validate_spectrum_arrays(wvl_arr, flx_arr)

        return cls(
            wavelengths=wvl_arr,
            flux=flx_arr,
            method=method
        )

    @classmethod
    def from_flux(cls, wavelength: 'Wavelength', flux: Union['Flux', 'Radiance'],
                  method: str = 'conserve') -> 'Spectrum':
        """
        Create a Spectrum from Wavelength and Flux/Radiance unit objects.

        This is the preferred method for creating spectra with proper unit handling.
        It ensures wavelengths are in nm and flux is in internal photon units.

        Parameters
        ----------
        wavelength : Wavelength
            Wavelength array (nyx.units.Wavelength object)
        flux : Flux or Radiance
            Flux array (nyx.units.Flux or nyx.units.Radiance object)
        method : str, optional
            Default resampling method ('linear' or 'conserve')

        Returns
        -------
        Spectrum
            New Spectrum instance with validated arrays

        Examples
        --------
        >>> from nyx.units import Wavelength, Flux
        >>> import astropy.units as u
        >>> wvl = Wavelength([400, 500, 600] * u.nm)
        >>> flx = Flux([1e10, 1.2e10, 0.9e10] * u.ph/u.s/u.m**2/u.nm)
        >>> spec = Spectrum.from_flux(wvl, flx)
        """
        from nyx.units import Wavelength as WavelengthType, Flux as FluxType, Radiance as RadianceType

        if not isinstance(wavelength, WavelengthType):
            raise TypeError(
                f"wavelength must be a Wavelength object, got {type(wavelength).__name__}. "
                "Use Wavelength(array * u.nm) or Spectrum.from_arrays() for raw arrays."
            )
        if not isinstance(flux, (FluxType, RadianceType)):
            raise TypeError(
                f"flux must be a Flux or Radiance object, got {type(flux).__name__}. "
                "Use Flux(array * units) or Spectrum.from_arrays() for raw arrays."
            )

        wvl_arr = wavelength.value
        flx_arr = flux.value

        _validate_spectrum_arrays(wvl_arr, flx_arr)

        return cls(
            wavelengths=jnp.asarray(wvl_arr),
            flux=jnp.asarray(flx_arr),
            method=method
        )

    @classmethod
    def from_energy_flux(cls, wavelength: Union['Wavelength', u.Quantity, jnp.ndarray],
                         flux: u.Quantity, method: str = 'conserve') -> 'Spectrum':
        """
        Create a Spectrum from energy flux, converting to photon flux.

        This is the preferred method for creating spectra from energy flux data
        (e.g., W/m²/nm or erg/s/cm²/Angstrom). The conversion to photon flux
        is handled automatically using the provided wavelength grid.

        Parameters
        ----------
        wavelength : Wavelength, Quantity, or array
            Wavelength grid. If array, assumed to be in nm.
        flux : astropy Quantity
            Energy flux with units (e.g., u.W/u.m**2/u.nm or u.erg/u.s/u.cm**2/u.angstrom)
        method : str, optional
            Default resampling method ('linear' or 'conserve')

        Returns
        -------
        Spectrum
            New Spectrum with flux converted to internal photon units

        Examples
        --------
        >>> import astropy.units as u
        >>> wvl = [400, 500, 600] * u.nm
        >>> energy_flux = [1e-10, 1.2e-10, 0.9e-10] * u.W/u.m**2/u.nm
        >>> spec = Spectrum.from_energy_flux(wvl, energy_flux)
        """
        from nyx.units import Wavelength as WavelengthType

        # Extract wavelength array and quantity
        if isinstance(wavelength, WavelengthType):
            wvl_arr = wavelength.value
            wvl_quantity = wavelength.to(u.nm)
        elif isinstance(wavelength, u.Quantity):
            wvl_arr = wavelength.to(u.nm).value
            wvl_quantity = wavelength.to(u.nm)
        else:
            wvl_arr = jnp.asarray(wavelength)
            wvl_quantity = wvl_arr * u.nm

        # Convert energy flux to photon flux
        # E = h*c/λ, so photon_flux = energy_flux / (h*c/λ) = energy_flux * λ / (h*c)
        photon_energy = (h * c / wvl_quantity).to(u.J)
        energy_flux_normalized = flux.to(u.W / u.m**2 / u.nm)
        photon_flux = energy_flux_normalized / photon_energy
        flx_arr = (photon_flux * u.ph).to(FLUX).value

        wvl_arr = jnp.asarray(wvl_arr)
        flx_arr = jnp.asarray(flx_arr)

        _validate_spectrum_arrays(wvl_arr, flx_arr)

        return cls(
            wavelengths=wvl_arr,
            flux=flx_arr,
            method=method
        )

    @classmethod
    def from_radiance(cls, wavelength: Union['Wavelength', u.Quantity, jnp.ndarray],
                      flux: Union['Radiance', u.Quantity, jnp.ndarray],
                      method: str = 'conserve') -> 'Spectrum':
        """
        Create a Spectrum from radiance (diffuse flux) data.

        Handles both already-converted Radiance objects and raw astropy Quantities
        with radiance units. For energy radiance, conversion to photon radiance
        is performed automatically.

        Parameters
        ----------
        wavelength : Wavelength, Quantity, or array
            Wavelength grid. If array, assumed to be in nm.
        flux : Radiance, Quantity, or array
            Radiance data. Can be:
            - nyx.units.Radiance object (already in photon units)
            - astropy Quantity with radiance units (converted automatically)
            - Raw array (assumed to be in internal photon units)
        method : str, optional
            Default resampling method ('linear' or 'conserve')

        Returns
        -------
        Spectrum
            New Spectrum with radiance flux

        Examples
        --------
        >>> import astropy.units as u
        >>> wvl = [400, 500, 600] * u.nm
        >>> radiance = [1e8, 1.1e8, 0.95e8] * u.ph/u.s/u.m**2/u.nm/u.sr
        >>> spec = Spectrum.from_radiance(wvl, radiance)
        """
        from nyx.units import Wavelength as WavelengthType, Radiance as RadianceType

        # Extract wavelength array and quantity
        if isinstance(wavelength, WavelengthType):
            wvl_arr = wavelength.value
            wvl_quantity = wavelength.to(u.nm)
        elif isinstance(wavelength, u.Quantity):
            wvl_arr = wavelength.to(u.nm).value
            wvl_quantity = wavelength.to(u.nm)
        else:
            wvl_arr = jnp.asarray(wavelength)
            wvl_quantity = wvl_arr * u.nm

        # Extract flux array
        if isinstance(flux, RadianceType):
            # Already in internal photon units
            flx_arr = flux.value
        elif isinstance(flux, u.Quantity):
            # Check if already in photon units
            if u.photon in flux.unit.bases:
                flx_arr = flux.to(RADIANCE).value
            else:
                # Energy radiance - convert to photon radiance
                photon_energy = (h * c / wvl_quantity).to(u.J)
                energy_radiance = flux.to(u.W / u.m**2 / u.nm / u.sr)
                photon_radiance = energy_radiance / photon_energy
                flx_arr = (photon_radiance * u.ph).to(RADIANCE).value
        else:
            # Assume already in internal units
            flx_arr = jnp.asarray(flux)

        wvl_arr = jnp.asarray(wvl_arr)
        flx_arr = jnp.asarray(flx_arr)

        _validate_spectrum_arrays(wvl_arr, flx_arr)

        return cls(
            wavelengths=wvl_arr,
            flux=flx_arr,
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


def _validate_spectrum_arrays(wavelengths: jnp.ndarray, flux: jnp.ndarray) -> None:
    """
    Validate wavelength and flux arrays for Spectrum creation.

    Parameters
    ----------
    wavelengths : array, shape (M,)
        Wavelength grid
    flux : array, shape (M,) or (N, M)
        Flux values

    Raises
    ------
    ValueError
        If arrays have invalid shapes or wavelengths are not monotonic
    """
    # Ensure wavelengths is 1D
    if wavelengths.ndim != 1:
        raise ValueError(
            f"Wavelengths must be 1D, got shape {wavelengths.shape}"
        )

    n_wvl = len(wavelengths)

    # Check flux dimensions
    if flux.ndim == 1:
        if len(flux) != n_wvl:
            raise ValueError(
                f"Flux length ({len(flux)}) must match wavelengths length ({n_wvl})"
            )
    elif flux.ndim == 2:
        if flux.shape[-1] != n_wvl:
            raise ValueError(
                f"Flux last dimension ({flux.shape[-1]}) must match "
                f"wavelengths length ({n_wvl})"
            )
    else:
        raise ValueError(
            f"Flux must be 1D or 2D, got shape {flux.shape}"
        )

    # Check wavelength monotonicity
    diffs = jnp.diff(wavelengths)
    if not jnp.all(diffs > 0):
        if jnp.all(diffs < 0):
            raise ValueError(
                "Wavelengths are in decreasing order. "
                "Please reverse the arrays or use increasing wavelengths."
            )
        raise ValueError(
            "Wavelengths must be strictly monotonically increasing"
        )


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