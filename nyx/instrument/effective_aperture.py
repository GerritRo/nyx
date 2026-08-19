from collections.abc import Callable

import astropy.units as u
import equinox as eqx
import jax
import jax.numpy as jnp

# Load jax_healpy and force to float 32
import jax_healpy as jhp  # noqa: E402
import numpy as np

from nyx.core.coordinates import offset_to_altaz, safe_arcsin
from nyx.core.parameter import Parameter
from nyx.core.protocols import InstrumentModel
from nyx.core.spectral import bin_widths
from nyx.instrument._interpolation import (
    PixelLattice,
    integrate_response,
    interpolate_pixel_rates,
    interpolate_regular_grid,
    project_lattice,
)

jax.config.update("jax_enable_x64", False)


def _as_lattice(grid, values) -> PixelLattice:
    """Accept either a :class:`PixelLattice` or explicit sample coordinates."""
    if isinstance(grid, PixelLattice):
        return grid
    return PixelLattice.from_grid(grid, values)


# Base class


class _BaseApertureInstrument(InstrumentModel):
    weight: eqx.AbstractVar[jax.Array]
    pixel_values: eqx.AbstractVar[jax.Array]
    pixel_efficiency: eqx.AbstractVar[Parameter]
    bandpass_values: eqx.AbstractVar[jax.Array]
    shift: eqx.AbstractVar[Parameter]
    rotation: eqx.AbstractVar[Parameter]
    _eval_grid: eqx.AbstractVar[jax.Array]
    _horizon_theta: eqx.AbstractVar[float]
    lattice: eqx.AbstractVar[PixelLattice]

    @property
    def bandpass(self):
        """Spectral transmission curve (no efficiency). Shape [n_wvl]."""
        return self.bandpass_values

    @property
    def centers(self):
        """Pixel centres in the offset frame. Shape [n_pix, 2]."""
        return self.lattice.centers

    @property
    def grid(self):
        """Per-pixel response sample coordinates. Shape [n_pix, 2, grid_dim]."""
        return self.lattice.grid

    def _correction_matrix(self):
        """3x3 correction matrix mapping the nominal offset frame to the
        detector frame.

        Returns
        -------
        dR : jax.Array, shape (3, 3)
        """
        dlon, dlat = self.shift.value[0], self.shift.value[1]
        rot = self.rotation.value[0]

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
        """Pointing matrix corrected for detector shift and rotation.

        Parameters
        ----------
        pm : jax.Array, shape (3, 3)
            Nominal pointing matrix from observation geometry.

        Returns
        -------
        jax.Array, shape (3, 3)
        """
        return self._correction_matrix() @ pm

    def _nominal_centers(self):
        """Pixel centers mapped from detector frame to nominal offset frame.

        Applies the inverse correction (``dR.T``) so that pixel positions
        can be looked up on the FOV scattering grid, which is defined in
        the nominal offset frame.

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
        """Project scattering eval grid onto pixels.

        Maps pixel centres from the detector frame back to the nominal
        offset frame (undoing shift + rotation) before sampling the
        scattering grid.

        Parameters
        ----------
        eval_grid_values : shape [n_lat, n_lon]
            The FOV evaluation grid, laid out row-by-latitude and
            column-by-longitude; see :func:`interpolate_pixel_rates`,
            which is where that pairing is enforced.

        Returns
        -------
        rates : shape [n_pixels]
        """
        centers_nom = self._nominal_centers()  # [lon, lat] per pixel
        rates = interpolate_pixel_rates(
            self._eval_grid,
            self._eval_grid,
            eval_grid_values,
            centers_nom,
        )
        return rates * self.weight * self.pixel_efficiency.value

    def project_diffuse(self, hp_values, pm):
        """Project HEALPix values onto pixels.

        Computes corrected AltAz for each pixel centre via the corrected
        pointing matrix, then samples the HEALPix sky map.

        Parameters
        ----------
        hp_values : shape [n_hp] (already band-integrated)
        pm : jax.Array, shape (3, 3)
            Nominal pointing matrix.

        Returns
        -------
        rates : shape [n_pixels]
        """
        R = self.corrected_pm(pm)
        # Lattice axis 0 is longitude
        lon, lat = self.centers[:, 0], self.centers[:, 1]
        az, alt = offset_to_altaz(lon, lat, R)
        theta = jnp.minimum(jnp.pi / 2 - alt, self._horizon_theta)
        phi = az
        rates = jhp.get_interp_val(hp_values, theta, phi)
        return rates * self.weight * self.pixel_efficiency.value

    def project_catalog(self, source_coords, source_fluxes):
        """Project point sources onto pixels.

        Source coordinates are already in the detector frame (transformed
        via the corrected pointing matrix in the render pipeline).

        Parameters
        ----------
        source_coords : shape [n_sources, 2]
            Offset-frame coords in the detector frame.
        source_fluxes : shape [n_sources] (already band-integrated + extincted)

        Returns
        -------
        rates : shape [n_pixels]
        """
        weights = project_lattice(
            self.lattice,
            self.pixel_values,
            source_coords,
            source_fluxes,
        )
        return weights * self.pixel_efficiency.value

    def save(self, filepath, wavelength_range=(200, 1000), wavelength_samples=1000, metadata=None):
        """Save instrument to HDF5 file.

        Delegates to :func:`nyx.instrument.io.save_instrument`.
        """
        from nyx.instrument.io import save_instrument

        save_instrument(self, filepath, wavelength_range, wavelength_samples, metadata)

    @classmethod
    def load(cls, filepath, geo):
        """Load instrument from HDF5 file.

        Delegates to :func:`nyx.instrument.io.load_instrument`.

        Parameters
        ----------
        filepath : str or Path
        geo : Geometry

        Returns
        -------
        instrument instance
        """
        from nyx.instrument.io import load_instrument

        return load_instrument(filepath, geo)


# Effective-aperture instrument


class EffectiveApertureInstrument(_BaseApertureInstrument):
    """Effective-aperture instrument.

    Trainable parameters: efficiency, pixel_efficiency, shift, rotation.
    Frozen data: pixel geometry, bandpass, eval grid.

    Notes
    -----
    The forward model uses ``efficiency * pixel_efficiency[i]`` for each
    pixel, so the absolute scale of ``pixel_efficiency`` is degenerate
    with ``efficiency``.  Freeze one of them before fitting (e.g.
    ``model = nyx.core.parameter.freeze(model, 'efficiency')``, or freeze
    a single reference pixel) to obtain a unique MLE and finite errors.
    """

    # Trainable parameters.
    # shift and rotation carry ``per_obs=True``.
    efficiency: Parameter
    pixel_efficiency: Parameter
    shift: Parameter
    rotation: Parameter

    # Frozen pixel geometry
    lattice: PixelLattice
    _weight: jax.Array  # [n_pix]
    _pixel_values: jax.Array  # [n_pix, grid_dim, grid_dim]
    bandpass_values: jax.Array  # [n_wvl]
    _eval_grid: jax.Array  # (ngrid,)

    # Fields with defaults last (eqx.field() or explicit default)
    _bandpass_func: Callable = eqx.field(static=True)
    _horizon_theta: float = eqx.field(static=True)

    def __init__(self, geo, bandpass, grid, values):
        """
        Parameters
        ----------
        geo : Geometry
            Resolution configuration (provides wavelengths, FOV grid).
        bandpass : callable
            Function mapping wavelength Quantity -> transmission.
        grid : array, shape (n_pixels, 2, grid_dim), or PixelLattice
            Pixel sub-grid coordinates in radians, from which the shared
            response lattice is recovered; or that lattice directly.
        values : array, shape (n_pixels, grid_dim, grid_dim)
            Pixel response values at sub-grid points.
        """
        values = np.asarray(values)

        self.efficiency = Parameter.from_value(1.0, scale=1.0)
        self.pixel_efficiency = Parameter.from_value(jnp.ones(values.shape[0]), scale=1.0)
        self.shift = Parameter.from_value(jnp.zeros(2), scale=1e-3, per_obs=True)
        self.rotation = Parameter.from_value(jnp.array([0.0]), scale=1e-2, per_obs=True)

        self.lattice = _as_lattice(grid, values)
        self._pixel_values = jnp.asarray(values)
        self._weight = integrate_response(self.lattice, self._pixel_values)
        self._bandpass_func = bandpass

        wvls = geo.wvls
        bp_values = bandpass(np.asarray(wvls) * u.nm) * np.asarray(bin_widths(wvls))
        self.bandpass_values = jnp.asarray(bp_values)
        self._eval_grid = jnp.asarray(np.linspace(-geo.fov, geo.fov, geo.ngrid))
        self._horizon_theta = float(np.pi / 2 - np.min(geo.lat))

    @property
    def weight(self):
        """Simpson-integrated pixel weight. Shape [n_pix]."""
        return self._weight

    @property
    def pixel_values(self):
        """Pixel response values. Shape [n_pix, grid_dim, grid_dim]."""
        return self._pixel_values


# Effective-aperture with mirror misalignment


class EffectiveApertureMisalignmentInstrument(_BaseApertureInstrument):
    """Effective-aperture instrument with mirror misalignment fitting.

    Stores a 5-D pixel response table parameterised by misalignment
    ``(sigma_x, sigma_y)``. At render time the table is interpolated
    bilinearly in sigma-space to obtain the effective ``(npix, Nx, Ny)``
    response for the current misalignment, after which the standard
    projection logic applies.

    Trainable parameters: efficiency, pixel_efficiency, shift, rotation,
    sigma_x, sigma_y.

    Notes
    -----
    The forward model uses ``efficiency * pixel_efficiency[i]`` for each
    pixel, so the absolute scale of ``pixel_efficiency`` is degenerate
    with ``efficiency``.  Freeze one of them before fitting (e.g.
    ``model = nyx.core.parameter.freeze(model, 'efficiency')``, or freeze
    a single reference pixel) to obtain a unique MLE and finite errors.
    """

    # Trainable parameters.
    # shift and rotation carry ``per_obs=True``.
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

    # Fields with defaults last (eqx.field() or explicit default)
    _bandpass_func: Callable = eqx.field(static=True)
    _horizon_theta: float = eqx.field(static=True)
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
        """
        Parameters
        ----------
        geo : Geometry
            Resolution configuration (provides wavelengths, FOV grid).
        bandpass : callable
            Function mapping wavelength Quantity -> transmission.
        grid : array, shape (n_pixels, 2, grid_dim), or PixelLattice
            Pixel sub-grid coordinates in radians, from which the shared
            response lattice is recovered; or that lattice directly.
        all_values : array, shape (Nsigma_x, Nsigma_y, n_pixels, Nx, Ny)
            Pixel response values for each sigma combination.
        sigma_x_coords : array, shape (Nsigma_x,)
            Regularly-spaced sigma_x grid values.
        sigma_y_coords : array, shape (Nsigma_y,)
            Regularly-spaced sigma_y grid values.
        sigma_x_init, sigma_y_init : float
            Initial misalignment parameter values.
        """
        all_values = np.asarray(all_values)
        sigma_x_coords = np.asarray(sigma_x_coords, dtype=np.float64)
        sigma_y_coords = np.asarray(sigma_y_coords, dtype=np.float64)

        # Trainable parameters
        self.efficiency = Parameter.from_value(1.0, scale=1.0)
        self.pixel_efficiency = Parameter.from_value(jnp.ones(all_values.shape[2]), scale=1.0)
        self.shift = Parameter.from_value(jnp.zeros(2), scale=1e-3, per_obs=True)
        self.rotation = Parameter.from_value(jnp.array([0.0]), scale=1e-2, per_obs=True)
        self.sigma_x = Parameter.from_value(float(sigma_x_init), scale=1.0)
        self.sigma_y = Parameter.from_value(float(sigma_y_init), scale=1.0)

        # Frozen pixel geometry
        self.lattice = _as_lattice(grid, all_values)

        # 5-D response table
        self.all_pixel_values = jnp.asarray(all_values)
        self.sigma_x_coords = jnp.asarray(sigma_x_coords)
        self.sigma_y_coords = jnp.asarray(sigma_y_coords)

        # Sigma grid metadata (static)
        nsx = len(sigma_x_coords)
        nsy = len(sigma_y_coords)
        self._sx0 = float(sigma_x_coords[0])
        self._sx_step = float(sigma_x_coords[1] - sigma_x_coords[0]) if nsx > 1 else 1.0
        self._nsx = nsx
        self._sy0 = float(sigma_y_coords[0])
        self._sy_step = float(sigma_y_coords[1] - sigma_y_coords[0]) if nsy > 1 else 1.0
        self._nsy = nsy

        # Bandpass
        self._bandpass_func = bandpass
        wvls = geo.wvls
        bp_values = bandpass(np.asarray(wvls) * u.nm) * np.asarray(bin_widths(wvls))
        self.bandpass_values = jnp.asarray(bp_values)

        # FOV eval grid
        self._eval_grid = jnp.asarray(np.linspace(-geo.fov, geo.fov, geo.ngrid))
        self._horizon_theta = float(np.pi / 2 - np.min(geo.lat))

    def _interp(self, data):
        """Interpolate data array at current (sigma_x, sigma_y)."""
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
        """Pixel response at current (sigma_x, sigma_y). Shape [npix, Nx, Ny]."""
        return self._interp(self.all_pixel_values)

    @property
    def weight(self):
        """Simpson-integrated pixel weight at current misalignment. Shape [npix].

        Integrating the sigma-interpolated response is exactly equivalent to
        interpolating pre-integrated weights -- both the quadrature and the
        sigma blend are linear -- so no separate weight table is stored.
        """
        return integrate_response(self.lattice, self.pixel_values)
