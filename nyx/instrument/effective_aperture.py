from collections.abc import Callable

import astropy.units as u
import equinox as eqx
import jax
import jax.numpy as jnp

# Load jax_healpy (later force it to float32)
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
    response_centroid,
)

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
    ``shift`` and ``rotation``, the last two per observation.

    Notes
    -----
    The forward model uses ``efficiency * pixel_efficiency[i]`` for each
    pixel, so the absolute scale of ``pixel_efficiency`` is degenerate
    with ``efficiency``.  Freeze one of them before fitting (e.g.
    ``model = nyx.core.parameter.freeze(model, 'efficiency')``, or freeze
    a single reference pixel) to obtain a unique MLE and finite errors.
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
        """Spectral transmission curve, without efficiency. Shape [n_wvl]."""
        return self.bandpass_values

    @property
    def centers(self):
        """Pixel centres in the offset frame. Shape [n_pix, 2]."""
        raise NotImplementedError

    def _correction_matrix(self):
        """Rotation (3, 3) taking the nominal offset frame to the detector frame."""
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
        """Nominal pointing matrix (3, 3), corrected for shift and rotation."""
        return self._correction_matrix() @ pm

    def _nominal_centers(self):
        """Pixel centres (n_pix, 2) mapped back to the nominal offset frame.

        The FOV scattering grid lives in that frame, so looking pixels up
        on it means undoing the detector correction first.
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
        """Sample the FOV scattering grid (n_lon, n_lat) at each pixel.

        Returns rates [n_pixels].
        """
        centers_nom = self._nominal_centers()  # [lon, lat] per pixel
        rates = interpolate_pixel_rates(self._eval_grid, eval_grid_values, centers_nom)
        return rates * self.weight * self.pixel_efficiency.value

    def project_diffuse(self, hp_values, pm):
        """Sample a band-integrated HEALPix sky [n_hp] at each pixel.

        ``pm`` is the nominal pointing matrix (3, 3).  Returns rates
        [n_pixels].
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
        """Write to HDF5; see :func:`nyx.instrument.io.save_instrument`."""
        from nyx.instrument.io import save_instrument

        save_instrument(self, filepath, wavelength_range, wavelength_samples, metadata)

    @classmethod
    def load(cls, filepath, geo):
        """Read from HDF5; see :func:`nyx.instrument.io.load_instrument`."""
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

        Requires the optional ``nyx[iactrace]`` dependency.

        Parameters
        ----------
        geo : Geometry
        telescope : iactrace.Telescope
            The optics.
        camera : iactrace.Camera
        half_angle : float
            Half-width of the field to scan, in radians.
        step : float
            Field-angle lattice spacing, in radians.
        wavelengths : array-like
            Wavelength grid in nm the bandpass is tabulated on.
        window : int or None
            Side length of each pixel's response window, in nodes. None
            measures it.
        chunk_size : int
            Field directions traced per render call.
        progress : bool
            Print scan progress to stderr.
        **scan_kwargs
            Anything else :func:`iactrace.analysis.effective_aperture` accepts.

        Examples
        --------
        ::

            import astropy.units as u
            import jax
            import jax.numpy as jnp
            import numpy as np
            from iactrace import Camera, Telescope

            from nyx.core.geometry import Geometry
            from nyx.instrument import EffectiveApertureInstrument

            geo = Geometry(wvls=jnp.linspace(300, 700, 50) * u.nm,
                           nside=16, ngrid=2, fov=3.5 * u.deg)

            telescope = Telescope.from_yaml("CT3.yaml", n_samples=100, key=jax.random.key(0))
            camera = Camera.from_yaml("HESS1U.yaml")

            instrument = EffectiveApertureInstrument.from_iactrace(
                geo, telescope, camera,
                half_angle=np.radians(3.0),      # camera subtends 2.5 deg, plus margin
                step=np.radians(0.16 / 8),       # pixel pitch / 8
                wavelengths=np.linspace(300, 700, 64),
                progress=True,
            )
            instrument.save("CT3.h5")

        The scan is the expensive step, so do it once and reload afterwards::

            instrument = EffectiveApertureInstrument.load("CT3.h5", geo)

        Sampling
        --------
        *Angular* sampling is the field and the step you scan it at. ``half_angle``
        has to clear the camera: take its radius over the focal length, so H.E.S.S.
        CT3 at 0.66 m and 15.03 m subtends ``atan(0.66/15.03) = 2.5 deg`` and 3 deg
        is a sensible field. ``step`` sets how well each pixel's response *shape*
        is resolved: work from the angular pixel pitch, so CT3's 0.16 deg pixels
        give ``0.16/8 = 0.02 deg`` for production and ``0.16/2`` for a draft.
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
        """Build an instrument from an already-scanned effective-aperture table::

        table = iactrace.analysis.effective_aperture(telescope, camera)
        inst = EffectiveApertureInstrument.from_iactrace_table(geo, table)
        """
        from nyx.instrument._iactrace import build_from_table

        return build_from_table(geo, table)


# Lattice-backed instruments


class _LatticeApertureInstrument(_BaseApertureInstrument):
    """An aperture instrument whose response is tabulated on a shared lattice.

    Adds the frozen-table half of the model: a :class:`PixelLattice` and a
    per-pixel response window on it, from which pixel centres and the
    point-source projection follow.
    """

    lattice: eqx.AbstractVar[PixelLattice]
    pixel_values: eqx.AbstractVar[jax.Array]

    @property
    def centers(self):
        """Pixel centres in the offset frame. Shape [n_pix, 2]."""
        return response_centroid(self.lattice, self.pixel_values)

    @property
    def grid(self):
        """Per-pixel response sample coordinates. Shape [n_pix, 2, grid_dim]."""
        return self.lattice.grid

    def project_catalog(self, source_coords, source_fluxes):
        """Project point sources onto pixels, returning rates [n_pixels].

        ``source_coords`` (n_sources, 2) are offset-frame coordinates
        already in the detector frame, and ``source_fluxes`` (n_sources,)
        already band-integrated and extincted -- the render pipeline does
        both before calling this.
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
    """Effective-aperture instrument with a fixed response table.

    Frozen data: pixel geometry, bandpass, eval grid.
    """

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
        """
        Parameters
        ----------
        geo : Geometry
        bandpass : callable
            Maps a wavelength Quantity to transmission.
        grid : array (n_pixels, 2, grid_dim) or PixelLattice
            Pixel sub-grid coordinates in radians, from which the shared
            response lattice is recovered; or that lattice directly.
        values : array (n_pixels, grid_dim, grid_dim)
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
        """Simpson-integrated pixel weight. Shape [n_pix]."""
        return self._weight

    @property
    def centers(self):
        """Pixel centres in the offset frame. Shape [n_pix, 2].

        Precomputed: the table is frozen, so the first moment is too.
        """
        return self._centers

    @property
    def pixel_values(self):
        """Pixel response values. Shape [n_pix, grid_dim, grid_dim]."""
        return self._pixel_values


# Effective-aperture with mirror misalignment


class EffectiveApertureMisalignmentInstrument(_LatticeApertureInstrument):
    """Effective-aperture instrument with fittable mirror misalignment.

    Stores a 5-D pixel response table parameterised by misalignment
    ``(sigma_x, sigma_y)``, both trainable.  At render time the table is
    interpolated bilinearly in sigma-space to give the effective
    ``(npix, Nx, Ny)`` response, after which projection is as usual.
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
        """
        Parameters
        ----------
        geo : Geometry
        bandpass : callable
            Maps a wavelength Quantity to transmission.
        grid : array (n_pixels, 2, grid_dim) or PixelLattice
            Pixel sub-grid coordinates in radians, from which the shared
            response lattice is recovered; or that lattice directly.
        all_values : array (Nsigma_x, Nsigma_y, n_pixels, Nx, Ny)
            Pixel response for each sigma combination.
        sigma_x_coords, sigma_y_coords : array
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
        """Interpolate ``data`` at the current (sigma_x, sigma_y)."""
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
        """Simpson-integrated pixel weight at current misalignment. Shape [npix]."""
        return integrate_response(self.lattice, self.pixel_values)

    @property
    def centers(self):
        """Pixel centres at the current misalignment. Shape [npix, 2]."""
        return response_centroid(self.lattice, self.pixel_values)
