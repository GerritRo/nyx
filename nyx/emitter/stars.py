import astropy.units as u
import healpy as hp
import jax.numpy as jnp
import numpy as np
from astropy.coordinates import ICRS, SkyCoord
from astropy.utils.data import download_file

from nyx.core.coordinates import HEALPixCatalog, rotate_healpix
from nyx.core.protocols import SourceObsData, set_source_weight
from nyx.core.spectral import ParametricSpectrum, SpectralModel
from nyx.emitter._base import BaseEmitter
from nyx.utils.spectra import Bandpass, PicklesTRDSAtlas1998, create_color_grid


class Stars(BaseEmitter):
    """Star catalog emitter with resolved bright stars and diffuse map.

    The complete sky map provides atmospheric in-scattering via the
    catalog scattering path.  Bright resolved stars and FOV HEALPix
    pixels (quasi-point sources) are extracted for direct rendering
    (extinction + pixel projection).

    Resolved star flux is subtracted from quasi-point pixels to avoid
    double-counting within the point source path (resolved stars appear
    both as individual sources and embedded in quasi-point pixel flux).
    The diffuse map stays complete (no FOV masking). The map-vs-point
    split handles that separation.

    Parameters
    ----------
    geo : Geometry
        Resolution configuration.
    spectral_model : SpectralModel
        Maps ``(n_sources, n_cond)`` conditions to
        ``(n_sources, n_wvl)`` spectra.
    bright_ra, bright_dec : np.ndarray
        Positions of resolved (bright) stars in degrees.
    bright_conditions : np.ndarray, shape (n_bright, n_cond)
        Spectral conditions for resolved stars.
    sky_map : np.ndarray, shape (n_cond, npix)
        Complete sky in linear flux (nested ordering).
    """

    def __init__(
        self,
        geo,
        spectral_model: SpectralModel,
        bright_ra: np.ndarray,
        bright_dec: np.ndarray,
        bright_conditions: np.ndarray,
        sky_map: np.ndarray,
    ):
        self._wvls = geo.wvls
        self._nside = geo.nside
        self._sky_map = sky_map
        self._map_nside = hp.npix2nside(sky_map.shape[1])
        self._bright_conditions = bright_conditions
        self._coords = SkyCoord(
            bright_ra * u.deg,
            bright_dec * u.deg,
            frame="icrs",
        )
        self._spectral_model = spectral_model
        self._hpx_index = HEALPixCatalog(bright_ra, bright_dec)
        self._lightcurves: list[tuple[int, np.ndarray]] = []

    def resolved_in_fov(self, obs) -> dict[str, np.ndarray]:
        """Resolved (individually rendered) stars in the FOV, in index order.

        The returned position ``i`` is exactly the ``index`` accepted by
        :meth:`add_lightcurve`: both use the same FOV catalog query, so the
        ordering is guaranteed to match the point sources built by
        :meth:`prepare`.

        Parameters
        ----------
        obs : Observation

        Returns
        -------
        dict
            ``ra_deg``, ``dec_deg`` and ``conditions`` (``[G, BP, RP]``) of
            each resolved star, ordered by index.
        """
        target_icrs = obs.target_icrs.transform_to("icrs")
        cat_idx = self._hpx_index.query(
            target_icrs.ra.deg,
            target_icrs.dec.deg,
            np.degrees(obs.geom.fov),
        )
        coords = self._coords[cat_idx]
        return {
            "ra_deg": coords.ra.deg,
            "dec_deg": coords.dec.deg,
            "conditions": self._bright_conditions[cat_idx],
        }

    def add_lightcurve(self, index: int, curve) -> None:
        """Modulate one resolved in-FOV star's brightness by a per-frame curve.

        The curve multiplies the star's flux at every time step, which (because
        the render is linear in each source's spectrum) is exactly an
        occultation or variable-star light curve.

        Parameters
        ----------
        index : int
            Index into the resolved in-FOV star list (see
            :meth:`resolved_in_fov`).  Only resolved stars (brighter than
            ``lim_mag``) can be modulated individually.
        curve : array-like
            Per-observation multiplicative factor (``1.0`` leaves the star
            unchanged, ``0.0`` fully blocks it).  Either ``(nobs,)`` for an
            achromatic factor, or ``(nobs, n_wvl)`` evaluated on ``geo.wvls``
            for a wavelength-dependent factor (e.g. a chromatic occultation).
            The leading axis must match the number of observation times at
            :meth:`prepare` time.
        """
        self._lightcurves.append((int(index), np.asarray(curve, dtype=float)))

    def _subtract_resolved_flux(self, fov_pix, fov_flux, cat_idx):
        """Remove resolved star flux from quasi-point FOV pixels.

        Resolved stars are rendered as individual point sources; their
        flux must be removed from the quasi-point pixel that contains
        them to avoid double-counting within the point source path.
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

    def prepare(self, obs) -> SourceObsData:
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
        cat_idx = self._hpx_index.query(
            target_icrs.ra.deg,
            target_icrs.dec.deg,
            np.degrees(obs.geom.fov),
        )
        resolved_cond = self._bright_conditions[cat_idx]
        resolved_coords = self._coords[cat_idx]

        # FOV map pixels as quasi-point sources (with one-pixel margin).
        # Resolved star flux is subtracted to avoid double-counting within
        # the point source path (resolved stars + quasi-point pixels).
        center_vec = hp.ang2vec(target_icrs.ra.deg, target_icrs.dec.deg, lonlat=True)
        fov_margin = hp.nside2resol(self._map_nside)
        fov_pix = hp.query_disc(
            self._map_nside,
            center_vec,
            obs.geom.fov + fov_margin,
            nest=True,
            inclusive=True,
        )
        fov_flux = self._sky_map[:, fov_pix].copy()
        fov_flux = self._subtract_resolved_flux(fov_pix, fov_flux, cat_idx)
        quasi_cond = -2.5 * np.log10(np.clip(fov_flux, 1e-30, None)).T
        theta, phi = hp.pix2ang(self._map_nside, fov_pix, nest=True)
        quasi_skycoords = SkyCoord(phi * u.rad, (np.pi / 2 - theta) * u.rad, frame="icrs")

        source_conditions = jnp.asarray(np.vstack([resolved_cond, quasi_cond]))

        # Per-source light curves (occultations, variable stars).  Resolved
        # stars occupy indices [0, n_resolved) in both source_conditions and
        # the per-obs source_coords stack, so the registered index maps
        # directly onto a source column.  A curve may be achromatic ((nobs,))
        # or wavelength-dependent ((nobs, n_wvl)); ``set_source_weight`` handles
        # the array shape (2-D or 3-D) and promotion, shared with Scene editing.
        n_resolved = len(resolved_cond)
        n_src, n_wvl = source_conditions.shape[0], len(self._wvls)
        source_weights = None
        for idx, curve in self._lightcurves:
            if not 0 <= idx < n_resolved:
                raise IndexError(
                    f"lightcurve index {idx} out of range; {n_resolved} resolved stars in FOV"
                )
            source_weights = set_source_weight(
                source_weights, idx, curve, nobs=obs.nobs, n_src=n_src, n_wvl=n_wvl
            )

        # Rotate diffuse map to AltAz and transform coords per observation.
        # Complete sky_map is used for scattering (no FOV masking)
        # resolved + quasi-point sources go through direct extinction.
        m_low = hp.ud_grade(
            self._sky_map, nside_out=self._nside, power=-2, order_in="NEST", order_out="RING"
        )

        diffuse_list, coords_list = [], []
        for i in range(obs.nobs):
            m_rot = -2.5 * np.log10(
                np.clip(
                    rotate_healpix(m_low, ICRS, obs.altaz_frames[i]),
                    1e-30,
                    None,
                )
            )
            diffuse_list.append(jnp.asarray(m_rot.T)[obs.geom.mask])

            if len(resolved_coords) > 0:
                aa = resolved_coords.transform_to(obs.altaz_frames[i])
                star_xy = np.column_stack([aa.az.rad, aa.alt.rad])
            else:
                star_xy = np.zeros((0, 2))
            qa = quasi_skycoords.transform_to(obs.altaz_frames[i])
            quasi_xy = np.column_stack([qa.az.rad, qa.alt.rad])
            coords_list.append(jnp.asarray(np.vstack([star_xy, quasi_xy])))

        return SourceObsData(
            diffuse_conditions=jnp.stack(diffuse_list),
            diffuse_norm=jnp.array(1.0 / obs.geom.pixel_area),
            source_conditions=source_conditions,
            source_coords=jnp.stack(coords_list),
            source_weights=source_weights,
            direct=False,
            _per_obs=("diffuse_conditions", "source_coords")
            + (("source_weights",) if source_weights is not None else ()),
        )

    @classmethod
    def from_gaia_dr3(cls, geo, lim_mag: float = 15.0) -> "Stars":
        """Stars with Gaia DR3 catalog + Pickles (1998) spectral model.

        Parameters
        ----------
        geo : Geometry
            Resolution configuration.
        lim_mag : float
            Limiting magnitude.  Brighter stars are resolved individually.
        """
        catalog = np.load(
            download_file(
                "https://zenodo.org/records/15396676/files/gaiadr3.npy",
                cache=True,
            )
        )
        faint_map = np.load(
            download_file(
                "https://zenodo.org/records/15396676/files/gaia_mag15plus.npy",
                cache=True,
            )
        )

        bright_mask = catalog["phot_g_mean_mag"] < lim_mag
        bright_ra = catalog["ra"][bright_mask]
        bright_dec = catalog["dec"][bright_mask]
        bright_conditions = np.column_stack(
            [
                catalog["phot_g_mean_mag"][bright_mask],
                np.nan_to_num(catalog["phot_bp_mean_mag"][bright_mask].astype(float), nan=21.0),
                np.nan_to_num(catalog["phot_rp_mean_mag"][bright_mask].astype(float), nan=21.0),
            ]
        )

        npix = len(faint_map[0])
        faint_flux = 10 ** (-0.4 * faint_map) + 1e-10
        catalog_map = _build_gaia_catalog_map(catalog, npix)
        sky_map = faint_flux + catalog_map

        spectral_model = _build_gaia_spectral_model(catalog, geo)

        return cls(geo, spectral_model, bright_ra, bright_dec, bright_conditions, sky_map)


# Gaia DR3


def _build_gaia_catalog_map(catalog, npix):
    """Bin all catalog stars into a HEALPix linear-flux map.

    Returns ``(3, npix)`` in ``[G, BP, RP]`` order, nested ordering.
    """
    map_nside = hp.npix2nside(npix)

    hp_inds = hp.ang2pix(
        map_nside,
        catalog["ra"],
        catalog["dec"],
        nest=True,
        lonlat=True,
    )

    g_mag = catalog["phot_g_mean_mag"]
    bp_mag = np.nan_to_num(catalog["phot_bp_mean_mag"], nan=21)
    rp_mag = np.nan_to_num(catalog["phot_rp_mean_mag"], nan=21)

    return np.vstack(
        [
            np.bincount(hp_inds, 10 ** (-0.4 * g_mag), npix),
            np.bincount(hp_inds, 10 ** (-0.4 * bp_mag), npix),
            np.bincount(hp_inds, 10 ** (-0.4 * rp_mag), npix),
        ]
    )


def _build_gaia_spectral_model(catalog, geo):
    """Build a Pickles (1998) spectral model for Gaia photometry.

    Uses the full catalog color range so the model is independent
    of lim_mag.
    """
    bp = np.nan_to_num(catalog["phot_bp_mean_mag"], nan=21)
    rp = np.nan_to_num(catalog["phot_rp_mean_mag"], nan=21)
    rp_bp = rp - bp

    G = Bandpass.from_SVO("GAIA/GAIA3.G")
    BP = Bandpass.from_SVO("GAIA/GAIA3.Gbp")
    RP = Bandpass.from_SVO("GAIA/GAIA3.Grp")

    spec_grid = create_color_grid(
        G,
        (RP, BP),
        [np.nanmin(rp_bp), 0.5],
        PicklesTRDSAtlas1998(),
        photon_flux=True,
    )
    return ParametricSpectrum.from_color_grid(
        spec_grid,
        geo.wvls,
        color_fn=lambda c: c[..., 2] - c[..., 1],  # RP - BP
        mag_fn=lambda c: c[..., 0],  # G
    )