from __future__ import annotations

import astropy.units as u
import healpy as hp
import jax.numpy as jnp
import numpy as np
from astropy.coordinates import ICRS, SkyCoord
from astropy.time import Time

from nyx.core.records import PerObs, SourceObsData
from nyx.emitter.catalog import CatalogEmitter
from nyx.emitter.catalogs import gaia, xhip
from nyx.emitter.catalogs.astrometry import altaz_track, rotate_healpix
from nyx.spectra import SpectralModel

__all__ = ["BrightStars", "Stars", "gaia_star_field"]


class Stars(CatalogEmitter):
    """Star catalog emitter with resolved bright stars and a diffuse map.

    The map carries the complete sky and drives in-scattering; stars inside
    the field of view are additionally rendered as points, and their flux is
    subtracted from the quasi-point FOV pixels to avoid double-counting.

    Parameters
    ----------
    geo : Geometry
        Resolution configuration.
    spectral_model : SpectralModel
        Maps ``(n_sources, n_cond)`` conditions to ``(n_sources, n_wvl)``
        spectra.
    sky_map : numpy.ndarray, shape (n_cond, npix)
        Complete sky in linear flux, nested ordering.
    bright_ra, bright_dec : numpy.ndarray or None
        Positions of the resolved stars, in degrees.
    bright_conditions : numpy.ndarray or None, shape (n_bright, n_cond)
    brightness : array-like or None
        Fittable amplitude on the whole catalog.
    transform : str or None
        Domain of *brightness*.
    """

    def __init__(
        self,
        geo,
        spectral_model: SpectralModel,
        sky_map: np.ndarray,
        *,
        bright_ra: np.ndarray | None = None,
        bright_dec: np.ndarray | None = None,
        bright_conditions: np.ndarray | None = None,
        brightness=None,
        transform: str | None = "log",
    ):
        n_cond = np.shape(sky_map)[0]
        bright_ra = np.zeros(0) if bright_ra is None else np.asarray(bright_ra)
        bright_dec = np.zeros(0) if bright_dec is None else np.asarray(bright_dec)
        super().__init__(
            geo,
            spectral_model,
            SkyCoord(bright_ra * u.deg, bright_dec * u.deg, frame="icrs"),
            brightness,
            transform,
        )
        self._bright_conditions = (
            np.zeros((0, n_cond)) if bright_conditions is None else bright_conditions
        )
        self._nside = geo.nside
        self._sky_map = sky_map
        self._map_nside = hp.npix2nside(sky_map.shape[1])

    def _repr_parts(self) -> list[str]:
        return [f"{len(self._coords)} sources", f"map nside={self._map_nside}"]

    def _pop_conditions(self, idx: int) -> np.ndarray:
        """This catalog's own photometry for the star.

        A star popped from here stays in the flux subtracted from the map
        pixel that holds it, so total flux is conserved: it leaves the
        point-source list without leaving the sky.
        """
        return self._bright_conditions[idx]

    def _subtract_resolved_flux(self, fov_pix, fov_flux, cat_idx):
        """Remove resolved star flux from the quasi-point FOV pixels.

        Parameters
        ----------
        fov_pix : numpy.ndarray
        fov_flux : numpy.ndarray, shape (n_cond, n_fov_pix)
        cat_idx : numpy.ndarray
            Every resolved star in the field, popped ones included: a popped
            star is still rendered individually, by a different emitter.

        Returns
        -------
        numpy.ndarray
        """
        if len(cat_idx) == 0:
            return fov_flux
        resolved_coords = self._coords[cat_idx]
        star_pix = hp.ang2pix(
            self._map_nside,
            resolved_coords.ra.deg,
            resolved_coords.dec.deg,
            nest=True,
            lonlat=True,
        )
        resolved_flux = 10 ** (-0.4 * self._bright_conditions[cat_idx])
        fov_lookup = {int(p): i for i, p in enumerate(fov_pix)}
        for j, sp in enumerate(star_pix):
            fi = fov_lookup.get(int(sp))
            if fi is not None:
                fov_flux[:, fi] -= resolved_flux[j]
        return np.clip(fov_flux, 0, None)

    def _prepare(self, obs) -> SourceObsData:
        """Extract resolved + quasi-point sources and build complete diffuse map.

        Parameters
        ----------
        obs : Observation

        Returns
        -------
        SourceObsData
        """
        # Resolved catalog stars in FOV
        target_icrs = obs.target_icrs.transform_to("icrs")
        cat_idx = self._index.query(
            target_icrs.ra.deg,
            target_icrs.dec.deg,
            np.degrees(obs.geo.fov),
        )
        # A star taken out with pop() is rendered by its own PointSource, so
        # it leaves the resolved list.
        point_idx = cat_idx[~self._taken[cat_idx]]
        resolved_cond = self._bright_conditions[point_idx]
        resolved_coords = self._coords[point_idx]

        # FOV map pixels as quasi-point sources (with one-pixel margin).
        # Resolved star flux is subtracted to avoid double-counting.
        center_vec = hp.ang2vec(target_icrs.ra.deg, target_icrs.dec.deg, lonlat=True)
        fov_margin = hp.nside2resol(self._map_nside)
        fov_pix = hp.query_disc(
            self._map_nside,
            center_vec,
            obs.geo.fov + fov_margin,
            nest=True,
            inclusive=True,
        )
        fov_flux = self._sky_map[:, fov_pix].copy()
        fov_flux = self._subtract_resolved_flux(fov_pix, fov_flux, cat_idx)
        quasi_cond = -2.5 * np.log10(np.clip(fov_flux, 1e-30, None)).T
        theta, phi = hp.pix2ang(self._map_nside, fov_pix, nest=True)
        quasi_skycoords = SkyCoord(phi * u.rad, (np.pi / 2 - theta) * u.rad, frame="icrs")

        source_conditions = jnp.asarray(np.vstack([resolved_cond, quasi_cond]))

        # Rotate diffuse map to AltAz and transform coords per observation.
        m_low = hp.ud_grade(
            self._sky_map, nside_out=self._nside, power=-2, order_in="NEST", order_out="RING"
        )

        diffuse_list = []
        for i in range(obs.nobs):
            m_rot = -2.5 * np.log10(
                np.clip(
                    rotate_healpix(m_low, ICRS, obs.altaz_frames[i]),
                    1e-30,
                    None,
                )
            )
            diffuse_list.append(jnp.asarray(m_rot.T)[obs.geo.mask])

        # Resolved stars lead the quasi-point map pixels.
        source_coords = np.concatenate(
            [
                altaz_track(resolved_coords, obs.times, obs.altaz_frames),
                altaz_track(quasi_skycoords, obs.times, obs.altaz_frames),
            ],
            axis=1,
        )

        return SourceObsData(
            diffuse_conditions=PerObs(jnp.stack(diffuse_list)),
            diffuse_norm=jnp.array(1.0 / obs.geo.pixel_area),
            source_conditions=source_conditions,
            source_coords=PerObs(jnp.asarray(source_coords)),
            direct=False,
        )

    @classmethod
    def from_gaia_dr3(cls, geo, lim_mag: float = 15.0, **kwargs) -> Stars:
        """Stars from the Gaia DR3 catalog, with a Pickles (1998) spectral model.

        Parameters
        ----------
        geo : Geometry
        lim_mag : float
            Limiting magnitude; brighter stars are resolved individually.
        **kwargs
            Passed to :class:`Stars`, e.g. ``brightness``.

        Returns
        -------
        Stars
        """
        catalog, faint_map = gaia.load_dr3()
        g, bp, rp = gaia.photometry(catalog)

        bright_mask = g < lim_mag
        bright_ra = catalog["ra"][bright_mask]
        bright_dec = catalog["dec"][bright_mask]
        bright_conditions = np.column_stack([g[bright_mask], bp[bright_mask], rp[bright_mask]])

        npix = len(faint_map[0])
        faint_flux = 10 ** (-0.4 * faint_map) + 1e-10
        sky_map = faint_flux + gaia.catalog_map(g, bp, rp, catalog["ra"], catalog["dec"], npix)

        spectral_model = gaia.spectral_model(gaia.color_grid(bp, rp), geo)

        return cls(
            geo,
            spectral_model,
            sky_map,
            bright_ra=bright_ra,
            bright_dec=bright_dec,
            bright_conditions=bright_conditions,
            **kwargs,
        )


class BrightStars(CatalogEmitter):
    """Pure point-source star catalog, with no diffuse map.

    Each star is rendered on its own, with in-scattering computed
    individually.

    Parameters
    ----------
    geo : Geometry
        Resolution configuration.
    spectral_model : SpectralModel
        Maps ``(n_stars, 3)`` conditions ``[v_mag, v_minus_b, active]`` to
        ``(n_stars, n_wvl)`` spectra.
    coords : astropy.coordinates.SkyCoord
        Catalog positions with proper motions, at :data:`~nyx.emitter.catalogs.xhip.EPOCH`.
    photometry : numpy.ndarray, shape (n_stars, 2)
        Per-star ``[v_mag, v_minus_b]``.
    brightness : array-like or None
        Fittable amplitude on the whole catalog -- a photometric zero-point;
        see :class:`~nyx.emitter.base.Emitter`.
    transform : str or None
        Domain of *brightness*.
    """

    def __init__(
        self,
        geo,
        spectral_model: SpectralModel,
        coords: SkyCoord,
        photometry: np.ndarray,
        brightness=None,
        transform: str | None = "log",
    ):
        super().__init__(geo, spectral_model, coords, brightness, transform)
        self._photometry = np.asarray(photometry)

    def _repr_parts(self) -> list[str]:
        return [f"{len(self._coords)} sources"]

    _pop_inscatter = True

    def _pop_conditions(self, idx: int) -> np.ndarray:
        """This catalog's photometry for the star, with ``active=1`` appended."""
        return np.append(self._photometry[idx], 1.0)

    def _prepare(self, obs) -> SourceObsData:
        """Propagate positions to each observation epoch and mask the horizon.

        Parameters
        ----------
        obs : Observation

        Returns
        -------
        SourceObsData
            ``source_conditions`` carries ``[v_mag, v_minus_b, active]``.
        """
        kept = ~self._taken
        coords, photometry = self._coords[kept], self._photometry[kept]

        track = altaz_track(coords, obs.times, obs.altaz_frames)  # (nobs, n_src, 2)
        active = (track[..., 1] > 0).astype(float)  # above the horizon
        conditions = np.stack([np.column_stack([photometry, a]) for a in active])

        return SourceObsData(
            source_conditions=PerObs(jnp.asarray(conditions)),
            source_coords=PerObs(jnp.asarray(track)),
            inscatter=True,
        )

    @classmethod
    def from_anderson2012(cls, geo, **kwargs) -> BrightStars:
        """Bright stars from the XHIP compilation (Anderson & Francis 2012).

        Spectra come from the Johnson V-B colour index against the Pickles
        (1998) library.

        Parameters
        ----------
        geo : Geometry
        **kwargs
            Passed to :class:`BrightStars`, e.g. ``brightness``.

        Returns
        -------
        BrightStars
        """
        coords, photometry = xhip.load_anderson2012()
        spectral_model = xhip.spectral_model(photometry[:, 1], geo)
        return cls(geo, spectral_model, coords, photometry, **kwargs)


def gaia_star_field(geo, lim_mag: float = 7.0):
    """Split Gaia DR3 into all-sky point sources and the background they leave.

    The catalog is partitioned exactly once, so no star is counted twice.
    Needs network access on first use, for the catalog and the Gaia
    passbands.

    Parameters
    ----------
    geo : Geometry
        Resolution configuration.
    lim_mag : float
        Split magnitude in the Gaia ``G`` band; the naked-eye limit is
        around 6.5.

    Returns
    -------
    points : BrightStars
        All-sky point sources brighter than *lim_mag*.
    background : Stars
        The remainder as a diffuse map, plus the pre-integrated faint-star
        map.
    """
    catalog, faint_map = gaia.load_dr3()
    g, bp, rp = gaia.photometry(catalog)
    bright = g < lim_mag

    npix = len(faint_map[0])
    faint_flux = 10 ** (-0.4 * faint_map) + 1e-10
    faint = ~bright
    background_map = faint_flux + gaia.catalog_map(
        g[faint], bp[faint], rp[faint], catalog["ra"][faint], catalog["dec"][faint], npix
    )

    grid = gaia.color_grid(bp, rp)
    background = Stars(geo, gaia.spectral_model(grid, geo), background_map)

    coords = SkyCoord(
        ra=catalog["ra"][bright] * u.deg,
        dec=catalog["dec"][bright] * u.deg,
        pm_ra_cosdec=np.zeros(int(bright.sum())) * u.mas / u.yr,
        pm_dec=np.zeros(int(bright.sum())) * u.mas / u.yr,
        obstime=Time(gaia.EPOCH),
        frame="icrs",
    )
    points = BrightStars(
        geo,
        gaia.spectral_model(grid, geo, horizon_mask=True),
        coords,
        np.column_stack([g[bright], bp[bright], rp[bright]]),
    )
    return points, background
