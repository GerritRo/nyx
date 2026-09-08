from collections.abc import Callable

import astropy.units as u
import equinox as eqx
import jax
import jax.numpy as jnp

# Load jax_healpy (later force it to float32)
import jax_healpy as jhp
import numpy as np

from nyx.core.coordinates import offset_to_altaz, safe_arcsin
from nyx.core.parameter import Parameter
from nyx.core.protocols import InstrumentModel
from nyx.instrument._interpolation import (
    PixelLattice,
    integrate_response,
    interpolate_pixel_rates,
    interpolate_regular_grid,
    project_lattice,
    response_centroid,
)
from nyx.utils.spectra import bin_widths

jax.config.update("jax_enable_x64", False)


def _as_lattice(grid, values) -> PixelLattice:
    """Accept either a :class:`PixelLattice` or explicit sample coordinates."""
    if isinstance(grid, PixelLattice):
        return grid
    return PixelLattice.from_grid(grid, values)


# Base class


class _BaseApertureInstrument(InstrumentModel):
    """An instrument whose pixels have a centre, a throughput weight and a bandpass.

    Every subclass is trainable in ``efficiency``, ``pixel_efficiency``,
    ``shift`` and ``rotation``, the last two per observation.  The forward
    model uses ``efficiency * pixel_efficiency[i]``, so the two are
    degenerate: freeze one before fitting for a unique MLE.
    """

    efficiency: eqx.AbstractVar[Parameter]
    weight: eqx.AbstractVar[jax.Array]
    pixel_efficiency: eqx.AbstractVar[Parameter]
    bandpass_values: eqx.AbstractVar[jax.Array]
    shift: eqx.AbstractVar[Parameter]
    rotation: eqx.AbstractVar[Parameter]
    _eval_grid: eqx.AbstractVar[jax.Array]
    _horizon_theta: eqx.AbstractVar[float]

    def _init_common(self, geo, bandpass, n_pix):
        """Assign the parameters and frozen geometry every subclass shares."""
        self.efficiency = Parameter.from_value(1.0, scale=1.0)
        self.pixel_efficiency = Parameter.from_value(jnp.ones(n_pix), scale=1.0)
        self.shift = Parameter.from_value(jnp.zeros(2), scale=1e-3, per_obs=True)
        self.rotation = Parameter.from_value(0.0, scale=1e-2, per_obs=True)

        wvls = np.asarray(geo.wvls)
        self._bandpass_func = bandpass
        self.bandpass_values = jnp.asarray(bandpass(wvls * u.nm) * np.asarray(bin_widths(geo.wvls)))
        self._eval_grid = jnp.asarray(np.linspace(-geo.fov, geo.fov, geo.ngrid))
        self._horizon_theta = float(np.pi / 2 - np.min(geo.lat))
        self._geo_signature = geo.signature

    @property
    def bandpass(self):
        """Spectral transmission curve, shape ``(n_wvl,)``, excluding efficiency."""
        return self.bandpass_values

    @property
    def centers(self):
        """Pixel centres in the offset frame. Shape [n_pix, 2]."""
        raise NotImplementedError

    def _correction_matrix(self):
        """Rotation taking the nominal offset frame to the detector frame.

        Returns
        -------
        jax.Array, shape (3, 3)
        """
        dlon, dlat = self.shift.value[0], self.shift.value[1]
        rot = self.rotation.value

        cl, sl = jnp.cos(dlon), jnp.sin(dlon)
        ca, sa = jnp.cos(dlat), jnp.sin(dlat)
        cr, sr = jnp.cos(rot), jnp.sin(rot)

        # Rz(-dlon):
        Rz = jnp.array([[cl, sl, 0.0], [-sl, cl, 0.0], [0.0, 0.0, 1.0]])
        # Ry(dlat):
        Ry = jnp.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]])
        # Rx(-rot):
        Rx = jnp.array([[1.0, 0.0, 0.0], [0.0, cr, sr], [0.0, -sr, cr]])

        return Rx @ Ry @ Rz

    def corrected_pm(self, pm):
        """Pointing matrix corrected for shift and rotation.

        Parameters
        ----------
        pm : jax.Array, shape (3, 3)

        Returns
        -------
        jax.Array, shape (3, 3)
        """
        return self._correction_matrix() @ pm

    def _nominal_centers(self):
        """Pixel centres in the nominal offset frame, where the FOV grid lives.

        Returns
        -------
        jax.Array, shape (n_pix, 2)
        """
        dR_inv = self._correction_matrix().T  # orthogonal inverse
        # Lattice axis 0 is longitude
        lon, lat = self.centers[:, 0], self.centers[:, 1]
        p = jnp.stack(
            [jnp.cos(lat) * jnp.cos(lon), jnp.cos(lat) * jnp.sin(lon), jnp.sin(lat)], axis=-1
        )
        p_nom = jnp.einsum("ij,...j->...i", dR_inv, p)
        nom_lon = jnp.arctan2(p_nom[..., 1], p_nom[..., 0])
        nom_lat = safe_arcsin(p_nom[..., 2])
        return jnp.stack([nom_lon, nom_lat], axis=-1)

    def project_scattered(self, eval_grid_values):
        """Sample the FOV scattering grid at each pixel.

        Parameters
        ----------
        eval_grid_values : jax.Array, shape (n_lon, n_lat)

        Returns
        -------
        jax.Array, shape (n_pixels,)
        """
        centers_nom = self._nominal_centers()  # [lon, lat] per pixel
        rates = interpolate_pixel_rates(self._eval_grid, eval_grid_values, centers_nom)
        return rates * self.weight * self.pixel_efficiency.value

    def project_diffuse(self, hp_values, pm):
        """Sample a band-integrated HEALPix sky at each pixel.

        Parameters
        ----------
        hp_values : jax.Array, shape (npix,)
        pm : jax.Array, shape (3, 3)
            Nominal pointing matrix.

        Returns
        -------
        jax.Array, shape (n_pixels,)
        """
        R = self.corrected_pm(pm)
        # Lattice axis 0 is longitude
        lon, lat = self.centers[:, 0], self.centers[:, 1]
        az, alt = offset_to_altaz(lon, lat, R)
        theta = jnp.minimum(jnp.pi / 2 - alt, self._horizon_theta)
        phi = az
        rates = jhp.get_interp_val(hp_values, theta, phi)
        return rates * self.weight * self.pixel_efficiency.value

    def save(self, filepath, wavelength_range=(200, 1000), wavelength_samples=1000, metadata=None):
        """Write to HDF5; see :func:`~nyx.instrument.io.save_instrument`."""
        from nyx.instrument.io import save_instrument

        save_instrument(self, filepath, wavelength_range, wavelength_samples, metadata)

    @classmethod
    def load(cls, filepath, geo):
        """Read from HDF5; see :func:`~nyx.instrument.io.load_instrument`."""
        from nyx.instrument.io import load_instrument

        return load_instrument(filepath, geo)

    @classmethod
    def from_iactrace(
        cls,
        geo,
        telescope,
        camera,
        *,
        half_angle,
        step,
        wavelengths,
        window=None,
        chunk_size=256,
        progress=False,
        **scan_kwargs,
    ):
        """Build an instrument by ray-tracing an iactrace telescope and camera.

        Needs the optional ``nyx[iactrace]`` dependency.

        Parameters
        ----------
        geo : Geometry
        telescope : iactrace.Telescope
        camera : iactrace.Camera
        half_angle : float
            Half-width of the field to scan, in radians; must clear the
            camera radius over the focal length.
        step : float
            Field-angle lattice spacing, in radians; around an eighth of the
            angular pixel pitch resolves a pixel response.
        wavelengths : array-like
            Wavelength grid in nm the bandpass is tabulated on.
        window : int or None
            Side length of each pixel's response window, in nodes; ``None``
            measures it.
        chunk_size : int
            Field directions traced per render call.
        progress : bool
            Whether to print scan progress to stderr.
        **scan_kwargs
            Anything else :func:`iactrace.analysis.effective_aperture` takes.

        Returns
        -------
        EffectiveApertureInstrument
        """
        scan_kwargs.update(
            half_angle=half_angle,
            step=step,
            wavelengths=wavelengths,
            window=window,
            chunk_size=chunk_size,
            progress=progress,
        )
        from nyx.instrument._iactrace import build_from_iactrace

        return build_from_iactrace(geo, telescope, camera, **scan_kwargs)

    @classmethod
    def from_iactrace_table(cls, geo, table):
        """Build an instrument from an already-scanned effective-aperture table.

        Parameters
        ----------
        geo : Geometry
        table : path-like or EffectiveApertureTable
            A ``.npz`` file written by ``iactrace.io.save_aperture_table``, or
            a table in hand.  Reading a file needs only numpy.

        Returns
        -------
        EffectiveApertureInstrument
        """
        from nyx.instrument._iactrace import build_from_table

        return build_from_table(geo, table)


# Lattice-backed instruments


class _LatticeApertureInstrument(_BaseApertureInstrument):
    """An aperture instrument whose response is tabulated on a shared lattice.

    Adds a :class:`~nyx.instrument._interpolation.PixelLattice` and a
    per-pixel response window on it.
    """

    lattice: eqx.AbstractVar[PixelLattice]
    pixel_values: eqx.AbstractVar[jax.Array]

    @property
    def centers(self):
        """Pixel centres in the offset frame. Shape [n_pix, 2]."""
        return response_centroid(self.lattice, self.pixel_values)

    @property
    def grid(self):
        """Per-pixel response sample coordinates, shape ``(n_pix, 2, grid_dim)``."""
        return self.lattice.grid

    def project_catalog(self, source_coords, source_fluxes):
        """Project point sources onto pixels.

        Parameters
        ----------
        source_coords : jax.Array, shape (n_sources, 2)
            Offset-frame coordinates, already in the detector frame.
        source_fluxes : jax.Array, shape (n_sources,)
            Already band-integrated and extincted by the render pipeline.

        Returns
        -------
        jax.Array, shape (n_pixels,)
        """
        weights = project_lattice(
            self.lattice,
            self.pixel_values,
            source_coords,
            source_fluxes,
        )
        return weights * self.pixel_efficiency.value


# Effective-aperture instrument


class EffectiveApertureInstrument(_LatticeApertureInstrument):
    """Effective-aperture instrument with a fixed response table."""

    efficiency: Parameter
    pixel_efficiency: Parameter
    shift: Parameter
    rotation: Parameter

    # Frozen pixel geometry
    lattice: PixelLattice
    _weight: jax.Array  # [n_pix]
    _centers: jax.Array  # [n_pix, 2]
    _pixel_values: jax.Array  # [n_pix, grid_dim, grid_dim]
    bandpass_values: jax.Array  # [n_wvl]
    _eval_grid: jax.Array  # (ngrid,)

    # Static fields last: they carry a default, so nothing may follow them.
    _bandpass_func: Callable = eqx.field(static=True)
    _horizon_theta: float = eqx.field(static=True)
    _geo_signature: tuple = eqx.field(static=True)

    def __init__(self, geo, bandpass, grid, values):
        """Build the instrument.

        Parameters
        ----------
        geo : Geometry
        bandpass : callable
            Maps a wavelength Quantity to transmission.
        grid : array-like, shape (n_pixels, 2, grid_dim), or PixelLattice
            Pixel sub-grid coordinates in radians, from which the shared
            response lattice is recovered; or that lattice directly.
        values : array-like, shape (n_pixels, grid_dim, grid_dim)
            Pixel response at the sub-grid points.
        """
        values = np.asarray(values)
        self._init_common(geo, bandpass, n_pix=values.shape[0])

        self.lattice = _as_lattice(grid, values)
        self._pixel_values = jnp.asarray(values)
        self._weight = integrate_response(self.lattice, self._pixel_values)
        self._centers = response_centroid(self.lattice, self._pixel_values)

    @property
    def weight(self):
        """Simpson-integrated pixel weight, shape ``(n_pix,)``."""
        return self._weight

    @property
    def centers(self):
        """Pixel centres in the offset frame, shape ``(n_pix, 2)``; precomputed."""
        return self._centers

    @property
    def pixel_values(self):
        """Pixel response values, shape ``(n_pix, grid_dim, grid_dim)``."""
        return self._pixel_values


# Effective-aperture with mirror misalignment


class EffectiveApertureMisalignmentInstrument(_LatticeApertureInstrument):
    """Effective-aperture instrument with fittable mirror misalignment.

    Holds a response table parameterised by a trainable ``(sigma_x,
    sigma_y)``, interpolated bilinearly in sigma at render time.
    """

    efficiency: Parameter
    pixel_efficiency: Parameter
    shift: Parameter
    rotation: Parameter
    sigma_x: Parameter
    sigma_y: Parameter

    # Frozen pixel geometry
    lattice: PixelLattice
    bandpass_values: jax.Array  # [n_wvl]
    _eval_grid: jax.Array  # (ngrid,)

    # 5-D response table and sigma grid metadata
    all_pixel_values: jax.Array  # [Nsigma_x, Nsigma_y, npix, Nx, Ny]
    sigma_x_coords: jax.Array  # [Nsigma_x]
    sigma_y_coords: jax.Array  # [Nsigma_y]

    # Static fields last: they carry a default, so nothing may follow them.
    _bandpass_func: Callable = eqx.field(static=True)
    _horizon_theta: float = eqx.field(static=True)
    _geo_signature: tuple = eqx.field(static=True)
    _sx0: float = eqx.field(static=True)
    _sx_step: float = eqx.field(static=True)
    _nsx: int = eqx.field(static=True)
    _sy0: float = eqx.field(static=True)
    _sy_step: float = eqx.field(static=True)
    _nsy: int = eqx.field(static=True)

    def __init__(
        self,
        geo,
        bandpass,
        grid,
        all_values,
        sigma_x_coords,
        sigma_y_coords,
        sigma_x_init=0.0,
        sigma_y_init=0.0,
    ):
        """Build the instrument.

        Parameters
        ----------
        geo : Geometry
        bandpass : callable
            Maps a wavelength Quantity to transmission.
        grid : array-like, shape (n_pixels, 2, grid_dim), or PixelLattice
            Pixel sub-grid coordinates in radians, from which the shared
            response lattice is recovered; or that lattice directly.
        all_values : array-like, shape (Nsigma_x, Nsigma_y, n_pixels, Nx, Ny)
            Pixel response for each sigma combination.
        sigma_x_coords, sigma_y_coords : array-like
            Regularly-spaced misalignment grids the table is tabulated on.
        sigma_x_init, sigma_y_init : float
            Starting misalignment.
        """
        all_values = np.asarray(all_values)
        sigma_x_coords = np.asarray(sigma_x_coords, dtype=np.float64)
        sigma_y_coords = np.asarray(sigma_y_coords, dtype=np.float64)
        self._init_common(geo, bandpass, n_pix=all_values.shape[2])

        self.sigma_x = Parameter.from_value(float(sigma_x_init), scale=1.0)
        self.sigma_y = Parameter.from_value(float(sigma_y_init), scale=1.0)

        self.lattice = _as_lattice(grid, all_values)
        self.all_pixel_values = jnp.asarray(all_values)
        self.sigma_x_coords = jnp.asarray(sigma_x_coords)
        self.sigma_y_coords = jnp.asarray(sigma_y_coords)

        # Sigma grid metadata, static so interpolation indices are traceable.
        self._nsx, self._nsy = len(sigma_x_coords), len(sigma_y_coords)
        self._sx0, self._sy0 = float(sigma_x_coords[0]), float(sigma_y_coords[0])
        self._sx_step = float(sigma_x_coords[1] - sigma_x_coords[0]) if self._nsx > 1 else 1.0
        self._sy_step = float(sigma_y_coords[1] - sigma_y_coords[0]) if self._nsy > 1 else 1.0

    def _interp(self, data):
        """Interpolate ``data`` at the current ``(sigma_x, sigma_y)``."""
        return interpolate_regular_grid(
            self.sigma_x.value,
            self.sigma_y.value,
            self._sx0,
            self._sx_step,
            self._nsx,
            self._sy0,
            self._sy_step,
            self._nsy,
            data,
        )

    @property
    def pixel_values(self):
        """Pixel response at the current misalignment, shape ``(npix, Nx, Ny)``."""
        return self._interp(self.all_pixel_values)

    @property
    def weight(self):
        """Simpson-integrated pixel weight at the current misalignment, shape ``(npix,)``."""
        return integrate_response(self.lattice, self.pixel_values)

    @property
    def centers(self):
        """Pixel centres at the current misalignment, shape ``(npix, 2)``."""
        return response_centroid(self.lattice, self.pixel_values)
