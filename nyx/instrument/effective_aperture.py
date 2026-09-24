from collections.abc import Callable

import astropy.units as u
import equinox as eqx
import jax
import jax.numpy as jnp
import jax_healpy as jhp
import numpy as np

from nyx.core.coordinates import offset_to_altaz, safe_arcsin
from nyx.core.parameter import Parameter
from nyx.core.protocols import InstrumentModel
from nyx.instrument.interpolation import (
    PixelLattice,
    integrate_response,
    interpolate_pixel_rates,
    project_lattice,
    response_centroid,
)
from nyx.spectra import bin_widths

__all__ = [
    "EffectiveApertureInstrument",
]

# Arcsecond in radians.
ARCSEC = np.pi / (180.0 * 3600.0)

# Characteristic size of a facet tilt RMS
_MISALIGNMENT_SCALE = 10.0


def _as_lattice(grid, values) -> PixelLattice:
    """Accept either a :class:`PixelLattice` or explicit sample coordinates."""
    if isinstance(grid, PixelLattice):
        return grid
    return PixelLattice.from_grid(grid, values)


def _as_misalignment(misalignment) -> tuple[Parameter | None, Parameter | None]:
    """Split a facet tilt RMS in arcseconds into the two axis parameters.

    Returns
    -------
    tuple of (Parameter or None, Parameter or None)
        ``(None, None)`` when misalignment is switched off.
    """
    if misalignment is None:
        return None, None
    sigma = np.asarray(misalignment, dtype=float)
    if sigma.shape not in ((), (2,)):
        raise ValueError(
            f"misalignment is one facet tilt RMS in arcseconds, or one per axis as "
            f"[lon, lat]; got shape {sigma.shape}"
        )
    lon, lat = np.broadcast_to(sigma, (2,))
    return (
        Parameter.from_value(float(lon), scale=_MISALIGNMENT_SCALE),
        Parameter.from_value(float(lat), scale=_MISALIGNMENT_SCALE),
    )


class EffectiveApertureInstrument(InstrumentModel):
    """An instrument whose pixels have a centre, a throughput weight and a bandpass.

    The response is a fixed table on a shared
    :class:`~nyx.instrument.interpolation.PixelLattice`, as produced by an
    iactrace ray-tracing scan. Build one with :meth:`from_iactrace_table`
    from a saved scan, or :meth:`from_iactrace` to scan from scratch.

    Trainable in ``efficiency``, ``pixel_efficiency``, ``shift`` and
    ``rotation``, the last two per observation, and in ``misalignment_lon``
    and ``misalignment_lat`` when that option is switched on. The forward model uses
    ``efficiency * pixel_efficiency[i]``, so the first two are degenerate:
    freeze one before fitting for a unique MLE.
    """

    efficiency: Parameter
    pixel_efficiency: Parameter
    shift: Parameter
    rotation: Parameter
    misalignment_lon: Parameter | None
    misalignment_lat: Parameter | None

    # Frozen pixel geometry
    lattice: PixelLattice
    bandpass_values: jax.Array  # [n_wvl]
    _weight: jax.Array  # [n_pix]
    _centers: jax.Array  # [n_pix, 2]
    _pixel_values: jax.Array  # [n_pix, grid_dim, grid_dim]
    _eval_grid: jax.Array  # (ngrid,)

    # Static fields
    _bandpass_func: Callable = eqx.field(static=True)
    _horizon_theta: float = eqx.field(static=True)
    _geo_signature: tuple = eqx.field(static=True)

    def __init__(self, geo, bandpass, grid, values, *, misalignment=None):
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
            Pixel response at the sub-grid points, of the **aligned** optics.
        misalignment : float, pair of floats, or None
            Initial facet tilt RMS in arcseconds, one value or one per axis as
            ``[lon, lat]``. ``None`` means no misalignment.
        """
        values = np.asarray(values)
        n_pix = values.shape[0]
        self.efficiency = Parameter.from_value(1.0, scale=1.0)
        self.pixel_efficiency = Parameter.from_value(jnp.ones(n_pix), scale=1.0)
        self.shift = Parameter.from_value(jnp.zeros(2), scale=1e-3, per_obs=True)
        self.rotation = Parameter.from_value(0.0, scale=1e-2, per_obs=True)
        self.misalignment_lon, self.misalignment_lat = _as_misalignment(misalignment)

        wvls = np.asarray(geo.wvls)
        self._bandpass_func = bandpass
        self.bandpass_values = jnp.asarray(bandpass(wvls * u.nm) * np.asarray(bin_widths(geo.wvls)))
        self._eval_grid = jnp.asarray(np.linspace(-geo.fov, geo.fov, geo.ngrid))
        self._horizon_theta = float(np.pi / 2 - np.min(geo.lat))
        self._geo_signature = geo.signature

        self.lattice = _as_lattice(grid, values)
        self._pixel_values = jnp.asarray(values)
        self._weight = integrate_response(self.lattice, self._pixel_values)
        self._centers = response_centroid(self.lattice, self._pixel_values)

    @property
    def bandpass(self):
        """Spectral transmission curve, shape ``(n_wvl,)``, excluding efficiency."""
        return self.bandpass_values

    @property
    def centers(self):
        """Pixel centres in the offset frame, shape ``(n_pix, 2)``; precomputed."""
        return self._centers

    @property
    def weight(self):
        """Simpson-integrated pixel weight, shape ``(n_pix,)``."""
        return self._weight

    @property
    def pixel_values(self):
        """Pixel response values, shape ``(n_pix, grid_dim, grid_dim)``."""
        return self._pixel_values

    @property
    def grid(self):
        """Per-pixel response sample coordinates, shape ``(n_pix, 2, grid_dim)``."""
        return self.lattice.grid

    @property
    def blur(self):
        """Misalignment blur width in lattice nodes, ``[lon, lat]``, or ``None``."""
        if self.misalignment_lon is None or self.misalignment_lat is None:
            return None
        sigma = jnp.stack([self.misalignment_lon.value, self.misalignment_lat.value])
        return 2.0 * sigma * ARCSEC / self.lattice.step

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
        dR_inv = self._correction_matrix().T
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

    def project_catalog(self, source_coords, source_fluxes):
        """Project point sources onto pixels, blurred by any misalignment.

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
            blur=self.blur,
        )
        return weights * self.pixel_efficiency.value

    @classmethod
    def from_iactrace(
        cls,
        geo,
        telescope,
        camera,
        *,
        half_angle,
        step,
        wvls,
        window=None,
        chunk_size=256,
        progress=False,
        misalignment=None,
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
        wvls : array-like
            Wavelength grid in nm the bandpass is tabulated on.
        window : int or None
            Side length of each pixel's response window, in nodes; ``None``
            measures it.
        chunk_size : int
            Field directions traced per render call.
        progress : bool
            Whether to print scan progress to stderr.
        misalignment : float, pair of floats, or None
            Initial facet blur model in arcseconds.
        **scan_kwargs
            Anything else :func:`iactrace.analysis.effective_aperture` takes.

        Returns
        -------
        EffectiveApertureInstrument
        """
        scan_kwargs.update(
            half_angle=half_angle,
            step=step,
            wavelengths=wvls,
            window=window,
            chunk_size=chunk_size,
            progress=progress,
        )
        from nyx.instrument.aperture_table import build_from_iactrace

        return cls(
            **build_from_iactrace(geo, telescope, camera, **scan_kwargs),
            misalignment=misalignment,
        )

    @classmethod
    def from_iactrace_table(cls, geo, table, *, misalignment=None):
        """Build an instrument from an already-scanned effective-aperture table.

        Parameters
        ----------
        geo : Geometry
        table : path-like or EffectiveApertureTable
            A ``.npz`` file written by ``iactrace.io.save_aperture_table``.
        misalignment : float, pair of floats, or None
            Initial facet tilt RMS in arcseconds.

        Returns
        -------
        EffectiveApertureInstrument
        """
        from nyx.instrument.aperture_table import build_from_table

        return cls(**build_from_table(geo, table), misalignment=misalignment)
