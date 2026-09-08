"""The Moon as a point source, with the ROLO reflectance model."""

import astropy
import astropy.units as u
import jax
import jax.numpy as jnp
import numpy as np

from nyx import ASSETS_PATH
from nyx.core.records import SourceObsData
from nyx.emitter.base import Emitter
from nyx.emitter.catalogs.astrometry import altaz_array
from nyx.utils.spectra import (
    ParametricSpectrum,
    SolarSpectrumRieke2008,
    SpectralModel,
    prepare_flux,
)

# ROLO coefficients and geometry, after Jones et al. (2013).
_ROLO_P1 = 4.06054
_ROLO_P2 = 12.8802
_ROLO_P3 = np.deg2rad(-30.5858)
_ROLO_P4 = np.deg2rad(16.7498)
_OMEGA_MOON = 6.4236e-5
_MEAN_MOON_DIST = 384400.0  # km

# Reduction of the ROLO albedo recommended by Noll et al. (2012).
_ROLO_ALBEDO_SCALE = 0.87

# Selenographic longitude of the observer, i.e. the Moon's libration in
# longitude.  Not modelled explicitly, so the median (zero) is used; the
# physical range of +/- 8 deg moves the albedo by well under a per cent.
_LIBRATION_LON = 0.0


class Moon(Emitter):
    """Moon emission model.

    Parameters
    ----------
    geo : Geometry
        Resolution configuration.
    spectral_model : ParametricSpectrum
        Maps ``(n_src, 4)`` conditions
        ``[phase_angle, distance_scale, libration_lon, moon_active]``
        to ``(n_src, n_wvl)`` spectra.
    brightness : array-like or None
        Fittable overall amplitude; see :class:`~nyx.emitter.base.Emitter`.
        The ROLO albedo is otherwise fixed, so this is the way to fit an
        overall lunar scale.
    transform : str or None
        Domain of *brightness*.
    """

    def __init__(self, geo, spectral_model: SpectralModel, brightness=None, transform="log"):
        super().__init__(geo, spectral_model, brightness, transform)

    def _prepare(self, obs) -> SourceObsData:
        """Query moon position and return per-observation data.

        Parameters
        ----------
        obs : Observation

        Returns
        -------
        SourceObsData
            ``source_conditions`` carries
            ``[phase_angle, distance_scale, libration_lon, moon_active]``.
        """
        nobs = obs.nobs

        sun_positions = [astropy.coordinates.get_sun(t) for t in obs.times]
        moon_positions = [astropy.coordinates.get_body("moon", t) for t in obs.times]

        conditions_list = []
        coords_list = []
        for i in range(nobs):
            sun = sun_positions[i]
            moon = moon_positions[i]
            sun_angle = moon.separation(sun)
            alpha = astropy.coordinates.Angle("180°") - sun_angle

            coord = moon.transform_to(obs.altaz_frames[i])
            moon_active = float(coord.alt.rad > 0)

            # Reflected sunlight scales with both legs of the Sun-Moon-Earth
            # path: the illumination the Moon receives and the distance it is
            # observed from.  The solar spectrum is tabulated at 1 AU.
            obs_factor = (_MEAN_MOON_DIST / moon.distance.to(u.km).value) ** 2
            sun_factor = (1.0 / sun.distance.to(u.AU).value) ** 2

            conditions_list.append(
                jnp.array(
                    [
                        [
                            alpha.rad,
                            obs_factor * sun_factor,
                            _LIBRATION_LON,
                            moon_active,
                        ]
                    ]
                )
            )
            coords_list.append(jnp.asarray(altaz_array(coord)))

        return SourceObsData(
            source_conditions=jnp.stack(conditions_list),
            source_coords=jnp.stack(coords_list),
            inscatter=True,
            per_obs_fields=("source_conditions", "source_coords"),
        )

    @classmethod
    def from_jones2013(cls, geo, **kwargs) -> "Moon":
        """Moon with the ROLO model of Jones et al. (2013).

        Parameters
        ----------
        geo : Geometry
        **kwargs
            Passed to :class:`Moon`, e.g. ``brightness``.

        Returns
        -------
        Moon
        """
        rolo = np.genfromtxt(ASSETS_PATH + "jones2013_lunar_rolo.dat", delimiter=",")
        solar_wvl, solar_flx = SolarSpectrumRieke2008()

        wvls = geo.wvls
        solar_resampled = prepare_flux(solar_wvl, solar_flx, wvls, from_energy=True)

        rolo_wvl = rolo[:, 0]
        rolo_coeffs = rolo[:, 1:]

        spectral_model = ParametricSpectrum(
            params={
                "solar_spectrum": jnp.asarray(solar_resampled),
                "rolo_coeffs": jnp.asarray(rolo_coeffs),
            },
            _model_fn=_make_rolo_model_fn(rolo_wvl, wvls),
        )
        return cls(geo, spectral_model, **kwargs)


# ---- internals


def _make_rolo_model_fn(rolo_wvl, target_wvl):
    """Build the JAX ROLO spectral model, closing over the wavelength grids.

    Parameters
    ----------
    rolo_wvl : jax.Array, shape (25,)
        ROLO reference wavelengths in nm.
    target_wvl : jax.Array, shape (n_wvl,)
        Target wavelength grid in nm.

    Returns
    -------
    callable
        ``(params, conditions) -> spectra``.
    """
    rolo_wvl = jnp.asarray(rolo_wvl)
    target_wvl = jnp.asarray(target_wvl)

    def model_fn(params, conditions):
        """Evaluate the ROLO model of Kieffer & Stone (2005).

        Parameters
        ----------
        params : dict
            ``solar_spectrum`` : (n_wvl,) resampled solar photon flux.
            ``rolo_coeffs`` : (25, 10) ROLO coefficients per band.
        conditions : jax.Array, shape (n_src, 4)
            ``[phase_angle, distance_scale, libration_lon, moon_active]``.

        Returns
        -------
        jax.Array, shape (n_src, n_wvl)
        """
        solar = params["solar_spectrum"]
        p = params["rolo_coeffs"]

        g = conditions[:, 0:1]
        dist_scale = conditions[:, 1:2]
        libration_lon = conditions[:, 2:3]
        active = conditions[:, 3:4]

        # Phase angle and libration are independent geometry: the libration
        # polynomial is fitted over +/- 8 deg only and its fifth-power term
        # diverges well before 90 deg, so it must not be driven by ``g``.
        sum_a = p[:, 0] + p[:, 1] * g + p[:, 2] * g**2 + p[:, 3] * g**3
        sum_b = p[:, 4] * libration_lon + p[:, 5] * libration_lon**3 + p[:, 6] * libration_lon**5
        sum_c = (
            p[:, 7] * jnp.exp(-g / _ROLO_P1)
            + p[:, 8] * jnp.exp(-g / _ROLO_P2)
            + p[:, 9] * jnp.cos((g - _ROLO_P3) / _ROLO_P4)
        )
        bands = jnp.exp(sum_a + sum_b + sum_c) * _ROLO_ALBEDO_SCALE  # (n_src, 25)

        # Interpolate ROLO bands to target wavelengths.  The 25 bands span
        # 350-1059.5 nm; outside that ``jnp.interp`` holds the outermost band
        # flat, so the albedo is extrapolated rather than cut off -- Cherenkov
        # cameras are sensitive below 350 nm, where the fit has no bands.
        interp = jax.vmap(lambda fp: jnp.interp(target_wvl, rolo_wvl, fp))(bands)

        norm = _OMEGA_MOON / jnp.pi * interp * dist_scale
        return active * norm * solar

    return model_fn
