import astropy
import astropy.units as u
import jax
import jax.numpy as jnp
import numpy as np

from nyx import ASSETS_PATH
from nyx.core.protocols import SourceObsData
from nyx.core.spectral import ParametricSpectrum, SpectralModel
from nyx.emitter._base import BaseEmitter
from nyx.utils.spectra import SolarSpectrumRieke2008, prepare_flux


class Moon(BaseEmitter):
    """Moon emission model.

    Parameters
    ----------
    geo : Geometry
        Resolution configuration.
    spectral_model : ParametricSpectrum
        Spectral model mapping ``(n_src, 4)`` conditions
        ``[phase_angle, moon_dist_km, sun_angle, moon_active]``
        to ``(n_src, n_wvl)`` spectra.
    """

    def __init__(self, geo, spectral_model: SpectralModel):
        self._spectral_model = spectral_model
        self._wvls = geo.wvls

    def prepare(self, obs) -> SourceObsData:
        """Query moon position and return per-observation data.

        Parameters
        ----------
        obs : Observation

        Returns
        -------
        SourceObsData
            ``source_conditions`` carries
            ``[phase_angle, moon_dist_km, sun_angle, moon_active]``.
            Moon is a pure point source (scattering via ``scatter_sources``).
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

            conditions_list.append(
                jnp.array(
                    [
                        [
                            alpha.rad,
                            moon.distance.to(u.km).value,
                            sun_angle.rad,
                            moon_active,
                        ]
                    ]
                )
            )
            coords_list.append(jnp.array([coord.az.rad, coord.alt.rad]).reshape(1, 2))

        return SourceObsData(
            source_conditions=jnp.stack(conditions_list),
            source_coords=jnp.stack(coords_list),
            inscatter=True,
            _per_obs=("source_conditions", "source_coords"),
        )

    @classmethod
    def from_jones2013(cls, geo) -> "Moon":
        """Moon with ROLO model (Jones et al. 2013).

        Parameters
        ----------
        geo : Geometry
            Resolution configuration.
        """
        rolo = np.genfromtxt(ASSETS_PATH + "jones2013_lunar_rolo.dat", delimiter=",")
        solar_wvl, solar_flx = SolarSpectrumRieke2008()

        wvls = geo.wvls
        solar_resampled = prepare_flux(solar_wvl, solar_flx, wvls, from_energy=True)

        rolo_wvl = rolo[:22, 0]
        rolo_coeffs = rolo[:22, 1:]

        spectral_model = ParametricSpectrum(
            params={
                "solar_spectrum": jnp.asarray(solar_resampled),
                "rolo_coeffs": jnp.asarray(rolo_coeffs),
            },
            _model_fn=_make_rolo_model_fn(rolo_wvl, wvls),
        )
        return cls(geo, spectral_model)


# Jones2013

_ROLO_P1 = 4.06054
_ROLO_P2 = 12.8802
_ROLO_P3 = np.deg2rad(-30.5858)
_ROLO_P4 = np.deg2rad(16.7498)
_OMEGA_MOON = 6.4177e-5
_MEAN_MOON_DIST = 384400.0  # km


def _make_rolo_model_fn(rolo_wvl, target_wvl):
    """Creates JAX-compatible ROLO spectral model function.

    Wavelength grids are captured in the closure (static).  The returned
    function maps ``(params, conditions) -> spectra``.

    Parameters
    ----------
    rolo_wvl : jax.Array, shape (22,)
        ROLO reference wavelengths in nm.
    target_wvl : jax.Array, shape (n_wvl,)
        Target wavelength grid in nm.
    """
    rolo_wvl = jnp.asarray(rolo_wvl)
    target_wvl = jnp.asarray(target_wvl)

    def model_fn(params, conditions):
        """Evaluate ROLO model.

        Parameters
        ----------
        params : dict
            ``solar_spectrum`` : (n_wvl,) resampled solar photon flux.
            ``rolo_coeffs`` : (22, 10) ROLO coefficients per band.
        conditions : jax.Array, shape (n_src, 4)
            ``[phase_angle, moon_dist_km, sun_angle, moon_active]``.

        Returns
        -------
        jax.Array, shape (n_src, n_wvl)
        """
        solar = params["solar_spectrum"]
        p = params["rolo_coeffs"]

        g = conditions[:, 0:1]
        dist = conditions[:, 1:2]
        s = conditions[:, 2:3]
        active = conditions[:, 3:4]

        sum_a = p[:, 0] + p[:, 1] * g + p[:, 2] * g**2 + p[:, 3] * g**3
        sum_b = p[:, 4] * s + p[:, 5] * s**3 + p[:, 6] * s**5
        sum_c = (
            p[:, 7] * jnp.exp(-g / _ROLO_P1)
            + p[:, 8] * jnp.exp(-g / _ROLO_P2)
            + p[:, 9] * jnp.cos((g - _ROLO_P3) / _ROLO_P4)
        )
        bands = jnp.exp(sum_a + sum_b + sum_c)  # (n_src, 22)

        # Interpolate ROLO bands to target wavelengths
        interp = jax.vmap(lambda fp: jnp.interp(target_wvl, rolo_wvl, fp))(bands)

        norm = _OMEGA_MOON / jnp.pi * interp * (_MEAN_MOON_DIST / dist) ** 2
        return active * norm * solar

    return model_fn
