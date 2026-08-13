import warnings

import astropy.units as u
import jax.numpy as jnp
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
from erfa import ErfaWarning

from nyx import ASSETS_PATH
from nyx.core.protocols import SourceObsData
from nyx.core.spectral import ParametricSpectrum, SpectralModel
from nyx.emitter._base import BaseEmitter
from nyx.utils.spectra import Bandpass, PicklesTRDSAtlas1998, create_color_grid

# Epoch the XHIP positions and proper motions are given for.
_XHIP_EPOCH = "J2000"


class BrightStars(BaseEmitter):
    """Supplementary catalog of stars too bright for Gaia.

    Gaia's detectors saturate on the very brightest stars, which are
    therefore missing or unreliable in DR3 -- and those are exactly the
    stars that dominate a Cherenkov camera's point-source background.
    :class:`~nyx.emitter.Stars` covers Gaia alone, so this emitter
    supplies the remainder from an extended Hipparcos compilation.

    Unlike :class:`~nyx.emitter.Stars`, these stars have no
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
            _per_obs=("source_conditions", "source_coords"),
        )

    @classmethod
    def from_anderson2012(cls, geo) -> "BrightStars":
        """Bright stars from the XHIP compilation (Anderson & Francis 2012).

        Spectra are inferred from the Johnson V-B colour index against the
        Pickles (1998) template library, the same way
        :meth:`~nyx.emitter.Stars.from_gaia_dr3` uses Gaia RP-BP.

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