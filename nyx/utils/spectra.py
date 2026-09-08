"""Spectra: the model types, the grid numerics, and the reference data."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import astropy.units as u
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import scipy.integrate as si
from astropy.constants import c, h
from astropy.io import fits, votable
from astropy.utils.data import download_file
from jax.scipy.interpolate import RegularGridInterpolator
from jax.typing import ArrayLike
from scipy.interpolate import UnivariateSpline

from nyx import ASSETS_PATH
from nyx.core.parameter import Parameter
from nyx.core.units import energy_flux_to_photon_flux, to_wavelength_nm

SVO_TABLE_URL = "https://svo2.cab.inta-csic.es/theory/fps/fps.php?ID="
CALSPEC_URL = "https://archive.stsci.edu/hlsps/reference-atlases/cdbs/current_calspec/"
SOLAR_SPECTRUM_URL = (
    "https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/solsys/solar_spec.fits"
)




def bin_edges(wvl: ArrayLike) -> jax.Array:
    """Bin edges of a wavelength grid: midpoints, half-extrapolated at the ends.

    Parameters
    ----------
    wvl : array-like
        Strictly increasing wavelength grid, at least two points.

    Returns
    -------
    jax.Array, shape (len(wvl) + 1,)
    """
    wl = jnp.asarray(wvl)
    edges = jnp.zeros(len(wl) + 1)
    edges = edges.at[1:-1].set((wl[1:] + wl[:-1]) / 2)
    edges = edges.at[0].set(wl[0] - (wl[1] - wl[0]) / 2)
    edges = edges.at[-1].set(wl[-1] + (wl[-1] - wl[-2]) / 2)
    return edges


def bin_widths(wvl: ArrayLike) -> jax.Array:
    """Width of each bin of a wavelength grid: the per-nm quadrature weight.

    Parameters
    ----------
    wvl : array-like
        Strictly increasing wavelength grid, at least two points.

    Returns
    -------
    jax.Array, shape (len(wvl),)
    """
    return jnp.diff(bin_edges(wvl))


def _linear_interp(wl_in: ArrayLike, flux_in: jax.Array, wl_out: ArrayLike) -> jax.Array:
    if flux_in.ndim == 1:
        return jnp.interp(wl_out, wl_in, flux_in)
    return jnp.stack([jnp.interp(wl_out, wl_in, f) for f in flux_in], axis=0)


def _check_coverage(covered: jax.Array, wl_in: jax.Array, wl_out: jax.Array) -> None:
    """Raise if any output bin lies entirely outside the input grid."""
    if isinstance(covered, jax.core.Tracer):
        return
    gaps = np.flatnonzero(np.asarray(covered) <= 0)
    if gaps.size == 0:
        return
    wl_out_np, wl_in_np = np.asarray(wl_out), np.asarray(wl_in)
    raise ValueError(
        f"resample_flux: {gaps.size} of {wl_out_np.size} output bins lie outside "
        f"the input wavelength grid ({wl_in_np[0]:.3f}-{wl_in_np[-1]:.3f} nm). "
        f"Uncovered output wavelengths span {wl_out_np[gaps[0]]:.3f}-"
        f"{wl_out_np[gaps[-1]]:.3f} nm. Narrow the output grid, extend the "
        f"input table, or pass method='linear' to hold the edge value instead."
    )


def _conserve_interp(wl_in: jax.Array, flux_in: jax.Array, wl_out: jax.Array) -> jax.Array:
    if len(wl_in) < 2 or len(wl_out) < 2:
        return _linear_interp(wl_in, flux_in, wl_out)

    edges_in = bin_edges(wl_in)
    edges_out = bin_edges(wl_out)

    overlap_lo = jnp.maximum(edges_out[:-1, None], edges_in[None, :-1])
    overlap_hi = jnp.minimum(edges_out[1:, None], edges_in[None, 1:])
    overlap = jnp.clip(overlap_hi - overlap_lo, 0, None)

    was_1d = flux_in.ndim == 1
    if was_1d:
        flux_in = flux_in[None, :]

    weighted = flux_in[:, None, :] * overlap[None, :, :]
    covered = jnp.sum(overlap, axis=-1)
    _check_coverage(covered, wl_in, wl_out)
    flux_out = jnp.sum(weighted, axis=-1) / jnp.where(covered > 0, covered, jnp.nan)

    if was_1d:
        flux_out = flux_out[0]
    return flux_out


def resample_flux(
    wvl_in: jax.Array, flux_in: jax.Array, wvl_out: jax.Array, method: str = "conserve"
) -> jax.Array:
    """Resample flux from one wavelength grid to another.

    Parameters
    ----------
    wvl_in : array-like
        Source wavelengths in nm.
    flux_in : array-like
        Source flux, 1-D or ``(batch, wavelength)``.
    wvl_out : array-like
        Target wavelengths in nm.
    method : str
        ``'conserve'`` or ``'linear'``.

    Returns
    -------
    jax.Array
    """
    if method == "conserve":
        return _conserve_interp(wvl_in, flux_in, wvl_out)
    elif method == "linear":
        return _linear_interp(wvl_in, flux_in, wvl_out)
    else:
        raise ValueError(f"Unknown method: {method}")



class SpectralModel(eqx.Module):
    """Base class for spectral models: source conditions to spectra.

    Any :class:`~nyx.core.parameter.Parameter` fields a subclass declares are
    discovered and trained automatically.
    """

    def __call__(self, conditions: jax.Array | None = None) -> jax.Array:
        raise NotImplementedError


class StoredSpectrum(SpectralModel):
    """A fixed, pre-computed spectrum array."""

    spectra: jax.Array

    def __call__(self, conditions: jax.Array | None = None) -> jax.Array:
        return self.spectra


class PassThroughSpectrum(SpectralModel):
    """Pass conditions through unchanged as the spectrum."""

    def __call__(self, conditions: jax.Array | None = None) -> jax.Array:
        if conditions is None:
            raise ValueError("PassThroughSpectrum requires `conditions`; got None.")
        return conditions


class ParametricSpectrum(SpectralModel):
    """Parametric spectral model: ``model_fn(params, conditions) -> spectra``.

    ``params`` may be a raw array or pytree (non-trainable), or a
    :class:`~nyx.core.parameter.Parameter` (trainable); ``model_fn`` sees a
    raw array either way.
    """

    params: object  # raw array/pytree, or Parameter
    _model_fn: Callable[..., jax.Array] = eqx.field(static=True)

    def __call__(self, conditions: jax.Array | None = None) -> jax.Array:
        p = self.params.value if isinstance(self.params, Parameter) else self.params
        return self._model_fn(p, conditions)



#: Photon flux at 550 nm of a magnitude-zero source, ``photon / s / m^2 / nm``.
#: Sets what "magnitude" means here; any consistent choice cancels in a ratio.
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
    from astropy.constants import c, h, k_B

    lam_nm = np.asarray(wvls.to(u.nm).value if isinstance(wvls, u.Quantity) else wvls, dtype=float)
    lam_m = lam_nm * 1e-9
    # Photon (not energy) radiance, so one power of lambda less than Planck.
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
        return hdul[1].data["WAVELENGTH"], hdul[1].data["FLUX"]


class Bandpass:
    """A filter transmission curve, and the photometry it defines.

    Parameters
    ----------
    wavelength : astropy.units.Quantity
    transmission : array-like
    """

    def __init__(self, wvl, transmission):
        self.lam = wvl
        self.trx = transmission
        self.min = self.lam.min()
        self.max = self.lam.max()
        self.spline = UnivariateSpline(self.lam, self.trx, s=0, ext=1)
        self._vegazero = None

    @property
    def vegazero(self):
        """Vega zero-point flux; downloads the Vega spectrum on first access."""
        if self._vegazero is None:
            wvl, flx = _read_calspec(CALSPEC_URL + "alpha_lyr_stis_012.fits")
            self._vegazero = self.integrate(
                wvl * u.angstrom, flx * u.erg / u.second / u.cm**2 / u.angstrom
            )
        return self._vegazero

    def __call__(self, lam):
        return self.spline(lam.to(self.lam.unit))

    def integrate(self, wvl, flx):
        """Photon-weighted band integral ``\\int lambda T(lambda) F(lambda) dlambda``.

        Parameters
        ----------
        wvl : astropy.units.Quantity
            Wavelengths of *flx*.
        flx : astropy.units.Quantity
            Flux, broadcastable against *wvl* along its last axis.

        Returns
        -------
        astropy.units.Quantity
        """
        z = flx * self(wvl) * wvl
        return si.simpson(y=z.value, x=wvl.value) * z.unit * wvl.unit

    def magnitude(self, wvl, flx):
        """Synthetic magnitude of *flx* in this band, against Vega.

        Parameters
        ----------
        wvl : astropy.units.Quantity
        flx : astropy.units.Quantity

        Returns
        -------
        numpy.ndarray
        """
        return -2.5 * np.log10(self.integrate(wvl, flx) / self.vegazero)

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
    wvl: Any
    flx: Any


def PicklesTRDSAtlas1998() -> SpectralGrid:
    """Load the Pickles (1998) TRDS stellar spectrum atlas.

    Returns
    -------
    SpectralGrid
    """
    file = np.genfromtxt(ASSETS_PATH + "pickles1998_trds_atlas.dat")
    return SpectralGrid(
        np.array([]), file[0] * u.angstrom, file[1:].T * u.erg / u.angstrom / u.s / u.cm**2
    )


def SolarSpectrumRieke2008() -> tuple[Any, Any]:
    """Load the Rieke (2008) solar spectrum from STScI CALSPEC.

    Returns
    -------
    wavelength, flux : astropy.units.Quantity
    """
    wvl, flx = _read_calspec(SOLAR_SPECTRUM_URL)
    return wvl * u.angstrom, flx * u.erg / u.s / u.cm**2 / u.angstrom



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
    """Load the HST/CALSPEC solar spectrum and resample it to *wvl_out*.

    Stays in energy units; for photon flux use
    :func:`SolarSpectrumRieke2008` with :func:`prepare_flux`.

    Parameters
    ----------
    wvl_out : array-like
        Target wavelengths in nm.
    normalize_at : float or None
        Wavelength in nm at which flux is set to 1.0, giving a
        dimensionless spectral shape.

    Returns
    -------
    jax.Array
    """
    wvl, flx = _read_calspec(SOLAR_SPECTRUM_URL)
    wvl_arr = jnp.asarray(np.ascontiguousarray(wvl / 10.0, dtype=np.float64))
    flx_arr = jnp.asarray(np.ascontiguousarray(flx, dtype=np.float64))
    if normalize_at is not None:
        ref = jnp.interp(normalize_at, wvl_arr, flx_arr)
        flx_arr = flx_arr / ref
    return _conserve_interp(wvl_arr, flx_arr, wvl_out)


def prepare_flux(wavelengths, flux, wvl_out, validate=True, from_energy=False):
    """Convert and resample arbitrary flux data to a target wavelength grid.

    Parameters
    ----------
    wavelengths : array-like or astropy.units.Quantity
    flux : array-like or astropy.units.Quantity
    wvl_out : array-like
        Target wavelengths in nm.
    validate : bool
        Whether to validate the input arrays.
    from_energy : bool
        Whether to convert energy flux to photon flux.

    Returns
    -------
    jax.Array
    """
    wvl_arr = to_wavelength_nm(wavelengths)
    if from_energy:
        flx_arr = energy_flux_to_photon_flux(wvl_arr, flux)
    else:
        flx_arr = jnp.asarray(flux.value if hasattr(flux, "value") else flux)
    if validate:
        _validate_spectrum_arrays(wvl_arr, flx_arr)
    return _conserve_interp(wvl_arr, flx_arr, wvl_out)



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
        wvl = spec_library.wvl
        flx = spec_library.flx.T
        flx = flx[:, np.newaxis, :] * extmod.extinguish(wvl, Ebv=EBVs_arr[..., np.newaxis])
        mag_corr = mag_band.magnitude(wvl, flx)
        return wvl, 10 ** (0.4 * mag_corr[..., np.newaxis]) * flx

    wvl, flx = redden_by_dust_extinction(EBVs)
    synth_color = color[0].magnitude(wvl, flx) - color[1].magnitude(wvl, flx)

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


def color_grid_spectrum(
    spec_grid: SpectralGrid,
    wavelengths: jax.Array,
    color_fn: Callable[[Any], Any],
    mag_fn: Callable[[Any], Any],
    active_fn: Callable[[Any], Any] | None = None,
) -> ParametricSpectrum:
    """A :class:`ParametricSpectrum` from a colour-indexed grid.

    Evaluates ``10 ** (-0.4 * mag_fn(c)) * grid_interp(color_fn(c))``.

    Parameters
    ----------
    spec_grid : SpectralGrid
        Colour-indexed grid, e.g. from :func:`create_color_grid`.
    wavelengths : array-like
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
    wvl_native = to_wavelength_nm(spec_grid.wvl)

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
    flux_resampled = resample_flux(wvl_native, jnp.asarray(flx_native), wavelengths)

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
