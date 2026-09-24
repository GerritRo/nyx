from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import astropy.units as u
import jax
import jax.numpy as jnp
import numpy as np
import scipy.integrate as si
from astropy.constants import c, h, k_B
from astropy.io import fits, votable
from astropy.utils.data import download_file
from jax.scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import UnivariateSpline

from nyx import ASSETS_PATH
from nyx.core.units import energy_flux_to_photon_flux, to_wavelength_nm
from nyx.spectra.models import ParametricSpectrum
from nyx.spectra.resample import _conserve_interp, resample_flux

__all__ = [
    "V_ZERO_POINT",
    "Bandpass",
    "load_solar_spectrum_rieke2008",
    "SpectralGrid",
    "blackbody_photon_flux",
    "color_grid_spectrum",
    "create_color_grid",
    "load_solar_flux",
    "prepare_flux",
]

SVO_TABLE_URL = "https://svo2.cab.inta-csic.es/theory/fps/fps.php?ID="
CALSPEC_URL = "https://archive.stsci.edu/hlsps/reference-atlases/cdbs/current_calspec/"
SOLAR_SPECTRUM_URL = (
    "https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/solsys/solar_spec.fits"
)

#: Photon flux at 550 nm of a magnitude-zero source, ``photon / s / m^2 / nm``
V_ZERO_POINT = 1.0e8


def blackbody_photon_flux(wvls: Any, temperature_k: float, mag: float = 0.0) -> jax.Array:
    """Black-body photon flux density, in ``photon / s / m^2 / nm``.

    Normalised to *mag* against :data:`V_ZERO_POINT` at 550 nm.

    Parameters
    ----------
    wvls : array-like
        Wavelengths in nm.
    temperature_k : float
    mag : float

    Returns
    -------
    jax.Array
    """
    lam_nm = np.asarray(wvls.to(u.nm).value if isinstance(wvls, u.Quantity) else wvls, dtype=float)
    lam_m = lam_nm * 1e-9
    # Photon (not energy) radiance
    exponent = h.value * c.value / (lam_m * k_B.value * float(temperature_k))
    shape = 1.0 / (lam_m**4 * np.expm1(exponent))
    return jnp.asarray(V_ZERO_POINT * 10 ** (-0.4 * mag) * shape / np.interp(550.0, lam_nm, shape))


def _read_calspec(url: str, cache: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Raw ``WAVELENGTH`` (angstrom) and ``FLUX`` columns of a CALSPEC FITS file.

    Returns
    -------
    wavelength, flux : numpy.ndarray
    """
    path = download_file(url, cache=cache)
    with fits.open(path) as hdul:
        data = hdul[1].data
        return (
            np.asarray(data["WAVELENGTH"], dtype=np.float64),
            np.asarray(data["FLUX"], dtype=np.float64),
        )


class Bandpass:
    """A filter transmission curve.

    Parameters
    ----------
    wavelength : astropy.units.Quantity
    transmission : array-like
    """

    def __init__(self, wvls, transmission):
        self.lam = wvls
        self.trx = transmission
        self.min = self.lam.min()
        self.max = self.lam.max()
        self.spline = UnivariateSpline(self.lam, self.trx, s=0, ext=1)
        self._vegazero = None

    @property
    def vegazero(self):
        """Vega zero-point flux; downloads the Vega spectrum on first access."""
        if self._vegazero is None:
            wvls, flx = _read_calspec(CALSPEC_URL + "alpha_lyr_stis_012.fits")
            self._vegazero = self.integrate(
                wvls * u.angstrom, flx * u.erg / u.second / u.cm**2 / u.angstrom
            )
        return self._vegazero

    def __call__(self, lam):
        return self.spline(lam.to(self.lam.unit))

    def integrate(self, wvls, flx):
        """Photon-weighted band integral ``\\int lambda T(lambda) F(lambda) dlambda``.

        Parameters
        ----------
        wvls : astropy.units.Quantity
            Wavelengths of *flx*.
        flx : astropy.units.Quantity
            Flux, broadcastable against *wvls* along its last axis.

        Returns
        -------
        astropy.units.Quantity
        """
        z = flx * self(wvls) * wvls
        return si.simpson(y=z.value, x=wvls.value) * z.unit * wvls.unit

    def magnitude(self, wvls, flx):
        """Synthetic magnitude of *flx* in this band, against Vega.

        Parameters
        ----------
        wvls : astropy.units.Quantity
        flx : astropy.units.Quantity

        Returns
        -------
        numpy.ndarray
        """
        return -2.5 * np.log10(self.integrate(wvls, flx) / self.vegazero)

    @classmethod
    def from_SVO(cls, filter_id, cache=True):
        """Fetch a filter curve from the SVO service, e.g. ``"GAIA/GAIA3.G"``.

        Parameters
        ----------
        filter_id : str

        Returns
        -------
        Bandpass
        """
        f_down = download_file(SVO_TABLE_URL + filter_id, cache=cache)
        table = votable.parse_single_table(f_down)
        return cls(table.array.data["Wavelength"] * u.angstrom, table.array.data["Transmission"])


@dataclass
class SpectralGrid:
    """Spectra indexed by one or more parameters, wavelength on the last axis."""

    points: Any
    wvls: Any
    flx: Any

    @classmethod
    def from_pickles1998(cls) -> SpectralGrid:
        """Load the Pickles (1998) TRDS stellar spectrum atlas.

        Returns
        -------
        SpectralGrid
        """
        file = np.genfromtxt(ASSETS_PATH + "pickles1998_trds_atlas.dat")
        return cls(
            np.array([]), file[0] * u.angstrom, file[1:].T * u.erg / u.angstrom / u.s / u.cm**2
        )


def load_solar_spectrum_rieke2008() -> tuple[Any, Any]:
    """Load the Rieke (2008) solar spectrum from STScI CALSPEC.

    Returns
    -------
    wavelength, flux : astropy.units.Quantity
    """
    wvls, flx = _read_calspec(SOLAR_SPECTRUM_URL)
    return wvls * u.angstrom, flx * u.erg / u.s / u.cm**2 / u.angstrom


def _validate_spectrum_arrays(wvls, flux):
    if wvls.ndim != 1:
        raise ValueError(f"Wavelengths must be 1D, got shape {wvls.shape}")
    n_wvl = len(wvls)
    if flux.ndim == 1:
        if len(flux) != n_wvl:
            raise ValueError(f"Flux length ({len(flux)}) must match wvls ({n_wvl})")
    elif flux.ndim == 2:
        if flux.shape[-1] != n_wvl:
            raise ValueError(f"Flux last dim ({flux.shape[-1]}) must match wvls ({n_wvl})")
    else:
        raise ValueError(f"Flux must be 1D or 2D, got shape {flux.shape}")
    diffs = np.diff(np.asarray(wvls))
    if not np.all(diffs > 0):
        raise ValueError("Wavelengths must be strictly monotonically increasing")


def load_solar_flux(wvls, normalize_at=None):
    """Load the HST/CALSPEC solar spectrum and resample it to *wvls*.

    Stays in energy units; for photon flux use
    :func:`load_solar_spectrum_rieke2008` with :func:`prepare_flux`.

    Parameters
    ----------
    wvls : array-like
        Target wavelength grid in nm.
    normalize_at : float or None
        Wavelength in nm at which flux is set to 1.0, giving a
        dimensionless spectral shape.

    Returns
    -------
    jax.Array
    """
    wvls_spec, flx = _read_calspec(SOLAR_SPECTRUM_URL)
    wvl_arr = jnp.asarray(np.ascontiguousarray(wvls_spec / 10.0, dtype=np.float64))
    flx_arr = jnp.asarray(np.ascontiguousarray(flx, dtype=np.float64))
    if normalize_at is not None:
        ref = jnp.interp(normalize_at, wvl_arr, flx_arr)
        flx_arr = flx_arr / ref
    return _conserve_interp(wvl_arr, flx_arr, wvls)


def prepare_flux(wvls_in, flux, wvls_out, validate=True, from_energy=False):
    """Convert and resample arbitrary flux data to a target wavelength grid.

    Parameters
    ----------
    wvls_in : array-like or astropy.units.Quantity
        Source wavelength grid the flux is sampled on, in nm.
    flux : array-like or astropy.units.Quantity
    wvls_out : array-like
        Target wavelength grid to resample onto, in nm.
    validate : bool
        Whether to validate the input arrays.
    from_energy : bool
        Whether to convert energy flux to photon flux.

    Returns
    -------
    jax.Array
    """
    wvl_arr = to_wavelength_nm(wvls_in)
    if from_energy:
        flx_arr = energy_flux_to_photon_flux(wvl_arr, flux)
    else:
        flx_arr = jnp.asarray(flux.value if hasattr(flux, "value") else flux)
    if validate:
        _validate_spectrum_arrays(wvl_arr, flx_arr)
    return _conserve_interp(wvl_arr, flx_arr, wvls_out)


def create_color_grid(
    mag_band, color, color_range, spec_library, EBV_range=None, extmod=None, photon_flux=False
):
    """Create a synthetic colour grid with dust reddening.

    Parameters
    ----------
    mag_band : Bandpass
        Defines the magnitude system.
    color : tuple of Bandpass
        Two bandpasses defining the colour ``color[0] - color[1]``.
    color_range : tuple of float
        ``(min, max)`` colour range to interpolate over.
    spec_library : SpectralGrid
        Spectral library to redden.
    EBV_range : tuple of float, or None
        ``(min, max)`` E(B-V) range, sampled at 20 points.
    extmod : extinction model, optional
        Defaults to ``G23(Rv=3.1)``.
    photon_flux : bool
        Whether to convert the output to photon/m^2/s/nm.

    Returns
    -------
    SpectralGrid
    """
    if EBV_range is None:
        EBV_range = [0, 10]

    EBVs = np.linspace(float(EBV_range[0]), float(EBV_range[1]), 20)
    if extmod is None:
        from dust_extinction.parameter_averages import G23

        extmod = G23(Rv=3.1)

    def redden_by_dust_extinction(EBVs_arr):
        wvls = spec_library.wvls
        flx = spec_library.flx.T
        flx = flx[:, np.newaxis, :] * extmod.extinguish(wvls, Ebv=EBVs_arr[..., np.newaxis])
        mag_corr = mag_band.magnitude(wvls, flx)
        return wvls, 10 ** (0.4 * mag_corr[..., np.newaxis]) * flx

    wvls, flx = redden_by_dust_extinction(EBVs)
    synth_color = color[0].magnitude(wvls, flx) - color[1].magnitude(wvls, flx)

    color_space = np.linspace(float(color_range[0]), float(color_range[1]), 51)
    ebv_interp = np.zeros((len(synth_color), len(color_space)))
    for i, color_arr in enumerate(synth_color):
        c_sort = np.argsort(color_arr)
        ebv_interp[i] = np.interp(
            color_space, color_arr[c_sort], EBVs[c_sort], left=np.nan, right=np.nan
        )

    wvls, flx = redden_by_dust_extinction(ebv_interp)

    photon_energy = h * c / wvls[:, np.newaxis, np.newaxis]
    if photon_flux:
        flx = (flx.T / photon_energy).to(1 / (u.m**2 * u.s * u.nm)).value
    else:
        flx = flx.T / photon_energy

    return SpectralGrid(np.array([color_space]), wvls, np.transpose(flx, [1, 0, 2]))


def color_grid_spectrum(
    spec_grid: SpectralGrid,
    wvls: jax.Array,
    color_fn: Callable[[Any], Any],
    mag_fn: Callable[[Any], Any],
    active_fn: Callable[[Any], Any] | None = None,
) -> ParametricSpectrum:
    """A :class:`~nyx.spectra.models.ParametricSpectrum` from a colour-indexed grid.

    Evaluates ``10 ** (-0.4 * mag_fn(c)) * grid_interp(color_fn(c))``.

    Parameters
    ----------
    spec_grid : SpectralGrid
        Colour-indexed grid, e.g. from :func:`create_color_grid`.
    wvls : array-like
        Target wavelength grid to resample onto.
    color_fn : callable
        ``conditions -> colour``, e.g. ``lambda c: c[..., 2] - c[..., 1]``.
    mag_fn : callable
        ``conditions -> magnitude``, e.g. ``lambda c: c[..., 0]``.
    active_fn : callable or None
        ``conditions -> 0/1 mask`` switching individual sources off.

    Returns
    -------
    ParametricSpectrum

    Raises
    ------
    ValueError
        If any colour grid node is unreachable by the template library.
    """
    axis = np.asarray(spec_grid.points[0], dtype=float)
    wvl_native = to_wavelength_nm(spec_grid.wvls)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices
        flx_native = np.nanmedian(spec_grid.flx, axis=-1)
    unreachable = ~np.isfinite(flx_native).all(axis=-1)
    if unreachable.any():
        bad = axis[unreachable]
        raise ValueError(
            f"color_grid_spectrum: {unreachable.sum()} of {axis.size} colour grid "
            f"nodes are not reachable by any template in the library, spanning "
            f"colour {bad.min():.3f} to {bad.max():.3f}. Narrow the colour range "
            f"or widen the reddening range the grid was built over."
        )
    flux_resampled = resample_flux(wvl_native, jnp.asarray(flx_native), wvls)

    grid_axis = jnp.asarray(axis)
    grid_points = (grid_axis,)
    clip_min, clip_max = grid_axis[0], grid_axis[-1]

    def _interp_model(params: jax.Array, conditions: jax.Array) -> jax.Array:
        interpol = RegularGridInterpolator(
            grid_points,
            params,
            method="linear",
            fill_value=0,
        )
        color = jnp.clip(color_fn(conditions), clip_min, clip_max)
        mag = mag_fn(conditions)
        shapes = interpol(color)
        spectra = 10 ** (-0.4 * mag)[..., None] * shapes
        if active_fn is not None:
            spectra = active_fn(conditions)[..., None] * spectra
        return spectra

    return ParametricSpectrum(params=flux_resampled, _model_fn=_interp_model)
