"""Stars, as a catalog map and as individual point sources.

Every star in nyx comes out of this module.  A catalog is used in two
shapes, and which one an emitter takes is decided by what is looking at
it, not by which star it is:

- :class:`Stars` is the shape a Cherenkov camera sees.  It carries the
  *complete* sky as a diffuse HEALPix map, so the in-scattering integral
  is over every star there is, and resolves individually only the stars
  that fall inside the field of view.
- :class:`BrightStars` is a pure point-source catalog: no map at all,
  each star rendered on its own like the Moon.  That is what a
  whole-hemisphere view needs, and what a supplementary catalog of
  saturated stars is.

Two catalogs feed them, and they are disjoint:

- **Gaia DR3** -- :meth:`Stars.from_gaia_dr3` for the camera's view, and
  :func:`gaia_star_field` for the hemisphere's, which partitions the same
  catalog exactly once into bright points plus the background they leave.
- **XHIP** (Anderson & Francis 2012) -- :meth:`BrightStars.from_anderson2012`,
  which supplies the very brightest stars, the ones Gaia saturates on and
  therefore reports badly or not at all.

Both paths infer spectra the same way: a colour index against the Pickles
(1998) template library, interpolated over a colour grid.
"""

from __future__ import annotations

import warnings

import astropy.units as u
import healpy as hp
import jax.numpy as jnp
import numpy as np
from astropy.coordinates import ICRS, SkyCoord
from astropy.time import Time
from astropy.utils.data import download_file
from erfa import ErfaWarning

from nyx import ASSETS_PATH
from nyx.core.coordinates import HEALPixCatalog, rotate_healpix
from nyx.core.protocols import SourceObsData, set_source_weight
from nyx.core.spectral import ParametricSpectrum, SpectralModel
from nyx.emitter._base import BaseEmitter
from nyx.utils.spectra import Bandpass, PicklesTRDSAtlas1998, create_color_grid

__all__ = ["BrightStars", "Stars", "gaia_star_field"]

#: Magnitude standing in for a catalog row with no photometry at all.
_NO_PHOTOMETRY_MAG = 99.0

#: Reference epoch of the Gaia DR3 astrometry.
_GAIA_DR3_EPOCH = "J2016.0"

#: Epoch the XHIP positions and proper motions are given for.
_XHIP_EPOCH = "J2000"

_GAIA_CATALOG_URL = "https://zenodo.org/records/15396676/files/gaiadr3.npy"
_GAIA_FAINT_MAP_URL = "https://zenodo.org/records/15396676/files/gaia_mag15plus.npy"


# The emitters


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
        self._geo_signature = geo.signature
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

        # Per-source light curves (occultations, variable stars).
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
            per_obs=("diffuse_conditions", "source_coords")
            + (("source_weights",) if source_weights is not None else ()),
        )

    @classmethod
    def from_gaia_dr3(cls, geo, lim_mag: float = 15.0) -> Stars:
        """Stars with Gaia DR3 catalog + Pickles (1998) spectral model.

        Parameters
        ----------
        geo : Geometry
            Resolution configuration.
        lim_mag : float
            Limiting magnitude.  Brighter stars are resolved individually.
        """
        catalog, faint_map = _load_gaia_dr3()
        g, bp, rp = _gaia_photometry(catalog)

        bright_mask = g < lim_mag
        bright_ra = catalog["ra"][bright_mask]
        bright_dec = catalog["dec"][bright_mask]
        bright_conditions = np.column_stack([g[bright_mask], bp[bright_mask], rp[bright_mask]])

        npix = len(faint_map[0])
        faint_flux = 10 ** (-0.4 * faint_map) + 1e-10
        catalog_map = _build_gaia_catalog_map(g, bp, rp, catalog["ra"], catalog["dec"], npix)
        sky_map = faint_flux + catalog_map

        spectral_model = _build_gaia_spectral_model(bp, rp, geo)

        return cls(geo, spectral_model, bright_ra, bright_dec, bright_conditions, sky_map)


class BrightStars(BaseEmitter):
    """Supplementary catalog of stars too bright for Gaia.

    Gaia's detectors saturate on the very brightest stars, which are
    therefore missing or unreliable in DR3 -- and those are exactly the
    stars that dominate a Cherenkov camera's point-source background.
    :class:`Stars` covers Gaia alone, so this emitter
    supplies the remainder from an extended Hipparcos compilation.

    Unlike :class:`Stars`, these stars have no
    representation in a diffuse map, so they are rendered exactly like
    the Moon: a pure point source whose in-scattering is computed
    individually via ``scatter_sources``.

    The catalog is rendered at a fixed source count.  Stars below the
    horizon are masked by the ``active`` condition rather than filtered
    out, because :meth:`nyx.core.scene.Scene.render` vmaps over the
    observation axis and needs the same number of sources for every
    observation.

    Parameters
    ----------
    geo : Geometry
        Resolution configuration.
    spectral_model : SpectralModel
        Maps ``(n_stars, 3)`` conditions ``[v_mag, v_minus_b, active]``
        to ``(n_stars, n_wvl)`` spectra.
    coords : astropy.coordinates.SkyCoord
        Catalog positions carrying proper motions, at :data:`_XHIP_EPOCH`.
    photometry : np.ndarray, shape (n_stars, 2)
        Per-star ``[v_mag, v_minus_b]``.  Time-independent.
    """

    def __init__(
        self,
        geo,
        spectral_model: SpectralModel,
        coords: SkyCoord,
        photometry: np.ndarray,
    ):
        self._wvls = geo.wvls
        self._spectral_model = spectral_model
        self._geo_signature = geo.signature
        self._coords = coords
        self._photometry = np.asarray(photometry)

    def _propagate(self, time) -> SkyCoord:
        """Propagate the catalog positions to *time*, position only.

        The catalog has no parallax, so ERFA substitutes a default distance
        and warns; that is exactly the intended pure on-sky extrapolation.
        It also hands back a radial velocity, and a coordinate carrying both
        proper motion and radial velocity but no usable distance cannot be
        converted to AltAz -- so only the propagated position is kept.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ErfaWarning)
            moved = self._coords.apply_space_motion(new_obstime=time)
        return SkyCoord(ra=moved.ra, dec=moved.dec, frame="icrs")

    def prepare(self, obs) -> SourceObsData:
        """Propagate positions to each observation epoch and mask the horizon.

        Parameters
        ----------
        obs : Observation

        Returns
        -------
        SourceObsData
            ``source_conditions`` carries ``[v_mag, v_minus_b, active]``.
            Pure point source, so in-scattering is computed per star.
        """
        conditions_list = []
        coords_list = []
        for i in range(obs.nobs):
            aa = self._propagate(obs.times[i]).transform_to(obs.altaz_frames[i])

            active = (aa.alt.rad > 0).astype(float)
            conditions_list.append(jnp.asarray(np.column_stack([self._photometry, active])))
            coords_list.append(jnp.asarray(np.column_stack([aa.az.rad, aa.alt.rad])))

        return SourceObsData(
            source_conditions=jnp.stack(conditions_list),
            source_coords=jnp.stack(coords_list),
            inscatter=True,
            per_obs=("source_conditions", "source_coords"),
        )

    @classmethod
    def from_anderson2012(cls, geo) -> BrightStars:
        """Bright stars from the XHIP compilation (Anderson & Francis 2012).

        Spectra are inferred from the Johnson V-B colour index against the
        Pickles (1998) template library, the same way
        :meth:`Stars.from_gaia_dr3` uses Gaia RP-BP.

        Parameters
        ----------
        geo : Geometry
            Resolution configuration.

        Notes
        -----
        Positions are propagated from J2000 with proper motion alone.  The
        catalog carries no parallax or radial velocity, so this is a plain
        on-sky extrapolation -- adequate here, since the fastest star in
        the catalog drifts under two arcminutes per quarter century.

        Requires network access on first use to fetch the Johnson
        passbands from the SVO Filter Profile Service.
        """
        xhip = np.genfromtxt(
            ASSETS_PATH + "anderson2012_xhip_suppl.dat",
            skip_header=3,
            delimiter=",",
            names=True,
        )

        coords = SkyCoord(
            ra=xhip["RAJ2000"] * u.deg,
            dec=xhip["DEJ2000"] * u.deg,
            pm_ra_cosdec=xhip["pmRA"] * u.mas / u.yr,
            pm_dec=xhip["pmDE"] * u.mas / u.yr,
            obstime=Time(_XHIP_EPOCH),
            frame="icrs",
        )

        # Positive is bluer, matching the RP - BP convention of the Gaia path.
        v_minus_b = xhip["Vmag"] - xhip["Bmag"]
        photometry = np.column_stack([xhip["Vmag"], v_minus_b])

        V = Bandpass.from_SVO("OSN/Johnson.V")
        B = Bandpass.from_SVO("OSN/Johnson.B")

        # Span exactly the colours present; the grid returns zero below its
        # lower edge, which would silently drop a star.
        spec_grid = create_color_grid(
            V,
            (V, B),
            [v_minus_b.min(), v_minus_b.max()],
            PicklesTRDSAtlas1998(),
            photon_flux=True,
        )
        spectral_model = ParametricSpectrum.from_color_grid(
            spec_grid,
            geo.wvls,
            color_fn=lambda c: c[..., 1],
            mag_fn=lambda c: c[..., 0],
            active_fn=lambda c: c[..., 2],
        )
        return cls(geo, spectral_model, coords, photometry)


def gaia_star_field(geo, lim_mag: float = 7.0):
    """Split Gaia DR3 into all-sky point sources and the background they leave.

    :meth:`Stars.from_gaia_dr3` resolves individual stars only inside the
    field of view, because that is all a Cherenkov camera ever sees, and
    its diffuse map holds the complete catalog so that the in-scattering
    integral does too.  Rendering the whole hemisphere -- a photograph,
    an all-sky map -- needs the opposite split: bright stars as sharp
    points *everywhere*, and everything else as a smooth map.

    This builds that pair with the catalog partitioned exactly once, so
    no star is counted twice::

        points, background = gaia_star_field(geo, lim_mag=7.0)
        emitters = {'stars': points, 'background': background, 'airglow': ...}

    Parameters
    ----------
    geo : Geometry
        Resolution configuration.
    lim_mag : float
        Split magnitude in the Gaia ``G`` band.  Brighter stars become
        point sources, fainter ones stay in the map.  The naked-eye
        limit is around 6.5; going much deeper costs catalog rows
        without adding anything a picture can show.

    Returns
    -------
    points : BrightStars
        Catalog stars brighter than *lim_mag*, over the whole sky, as
        individual point sources.
    background : Stars
        The unresolved remainder as a diffuse map: everything fainter
        than *lim_mag* plus the pre-integrated faint-star map.

    Notes
    -----
    Gaia saturates on the very brightest stars, so pair this with
    :meth:`BrightStars.from_anderson2012`, which supplies exactly
    those and is disjoint from the Gaia catalog.

    Requires network access on first use, for the catalog and for the
    Gaia passbands from the SVO Filter Profile Service.
    """
    catalog, faint_map = _load_gaia_dr3()
    g, bp, rp = _gaia_photometry(catalog)
    bright = g < lim_mag

    npix = len(faint_map[0])
    faint_flux = 10 ** (-0.4 * faint_map) + 1e-10
    faint = ~bright
    catalog_map = _build_gaia_catalog_map(
        g[faint], bp[faint], rp[faint], catalog["ra"][faint], catalog["dec"][faint], npix
    )
    # The colour grid spans the whole catalog, so both halves interpolate
    # the same spectral library over the same axis.
    background = Stars(
        geo,
        _build_gaia_spectral_model(bp, rp, geo),
        np.zeros(0),
        np.zeros(0),
        np.zeros((0, 3)),
        faint_flux + catalog_map,
    )

    coords = SkyCoord(
        ra=catalog["ra"][bright] * u.deg,
        dec=catalog["dec"][bright] * u.deg,
        pm_ra_cosdec=np.zeros(int(bright.sum())) * u.mas / u.yr,
        pm_dec=np.zeros(int(bright.sum())) * u.mas / u.yr,
        obstime=Time(_GAIA_DR3_EPOCH),
        frame="icrs",
    )
    points = BrightStars(
        geo,
        _build_gaia_spectral_model(bp, rp, geo, horizon_mask=True),
        coords,
        np.column_stack([g[bright], bp[bright], rp[bright]]),
    )
    return points, background


# Catalogs


def _load_gaia_dr3():
    """Download (once) and load the Gaia DR3 catalog and faint-star map."""
    catalog = np.load(download_file(_GAIA_CATALOG_URL, cache=True))
    faint_map = np.load(download_file(_GAIA_FAINT_MAP_URL, cache=True))
    return catalog, faint_map


def _gaia_photometry(catalog):
    """
    ``(G, BP, RP)`` magnitudes with missing BP/RP filled by colour.

    Returns
    -------
    g, bp, rp : np.ndarray
        Magnitudes, one entry per catalog row, all finite.
    """
    g = np.asarray(catalog["phot_g_mean_mag"], dtype=float)
    bp = np.asarray(catalog["phot_bp_mean_mag"], dtype=float)
    rp = np.asarray(catalog["phot_rp_mean_mag"], dtype=float)

    measured = np.isfinite(bp) & np.isfinite(rp)
    if not measured.any():
        raise ValueError("catalog contains no source with both BP and RP measured")
    median_color = float(np.median((rp - bp)[measured]))

    g = np.where(np.isfinite(g), g, _NO_PHOTOMETRY_MAG)
    bp = np.where(measured, bp, g - median_color / 2)
    rp = np.where(measured, rp, g + median_color / 2)
    return g, bp, rp


def _build_gaia_catalog_map(g, bp, rp, ra, dec, npix):
    """Bin all catalog stars into a HEALPix linear-flux map.

    Returns ``(3, npix)`` in ``[G, BP, RP]`` order, nested ordering.
    """
    map_nside = hp.npix2nside(npix)
    hp_inds = hp.ang2pix(map_nside, ra, dec, nest=True, lonlat=True)
    return np.vstack([np.bincount(hp_inds, 10 ** (-0.4 * mag), npix) for mag in (g, bp, rp)])


def _build_gaia_spectral_model(bp, rp, geo, horizon_mask: bool = False):
    """Build a Pickles (1998) spectral model for Gaia photometry.

    Parameters
    ----------
    bp, rp : np.ndarray
        Catalog BP and RP magnitudes; only their range is used, to size
        the colour axis of the interpolation grid.
    geo : Geometry
        Resolution configuration.
    horizon_mask : bool
        Read a fourth condition column as a 0/1 switch, which is what
        :class:`BrightStars` appends to mask the stars
        that are below the horizon.
    """
    rp_bp = rp - bp

    G = Bandpass.from_SVO("GAIA/GAIA3.G")
    BP = Bandpass.from_SVO("GAIA/GAIA3.Gbp")
    RP = Bandpass.from_SVO("GAIA/GAIA3.Grp")

    spec_grid = create_color_grid(
        G,
        (RP, BP),
        [float(np.min(rp_bp)), 0.5],
        PicklesTRDSAtlas1998(),
        photon_flux=True,
    )
    return ParametricSpectrum.from_color_grid(
        spec_grid,
        geo.wvls,
        color_fn=lambda c: c[..., 2] - c[..., 1],  # RP - BP
        mag_fn=lambda c: c[..., 0],  # G
        active_fn=(lambda c: c[..., 3]) if horizon_mask else None,
    )
