"""Zodiacal light: sunlight scattered by interplanetary dust, after Leinert et al. (1998)."""

import astropy.units as u
import jax.numpy as jnp
import numpy as np
from jax.scipy.interpolate import RegularGridInterpolator

from nyx import ASSETS_PATH
from nyx.core.records import SourceObsData
from nyx.core.units import energy_flux_to_photon_flux
from nyx.emitter.base import Emitter
from nyx.utils.spectra import ParametricSpectrum, SpectralModel, load_solar_flux

__all__ = ["ZodiacalLight"]

# Wavelength the Leinert (1998) colour correction is normalised at, in nm.
_REFERENCE_WAVELENGTH = 500.0

# Solar elongations the colour correction is tabulated at, in degrees.
_ELONGATION_RANGE = (30.0, 90.0)

# Reddening slopes as (blue-ward of the reference, red-ward of it), at the
# near and far end of _ELONGATION_RANGE respectively.
_SLOPE_NEAR = (1.2, 0.8)
_SLOPE_FAR = (0.9, 0.6)


# ---- internals


def _leinert_weights(alpha, beta, leinert_points, leinert_values, wvls):
    """Leinert zodiacal light weights, with colour correction.

    Parameters
    ----------
    alpha, beta : numpy.ndarray, shape (nsky,)
        Ecliptic longitude and latitude in radians.
    leinert_points : tuple of numpy.ndarray
        ``(alpha_grid, beta_grid)`` in radians.
    leinert_values : numpy.ndarray, shape (n_alpha, n_beta)
    wvls : numpy.ndarray, shape (n_wvl,)
        Wavelengths in nm.

    Returns
    -------
    numpy.ndarray, shape (nsky, n_wvl)
    """
    # Fold alpha into [0, pi] and take abs(beta) for symmetry
    alpha_folded = np.abs((alpha + np.pi) % (2 * np.pi) - np.pi)
    beta_abs = np.abs(beta)

    # Interpolation on the Leinert table
    interp = RegularGridInterpolator(
        leinert_points,
        np.asarray(leinert_values),
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    points = np.stack([alpha_folded.ravel(), beta_abs.ravel()], axis=-1)
    weights = np.asarray(interp(points)).reshape(alpha_folded.shape)

    # Color correction.  Leinert (1998) gives a broken slope: the reddening
    # is steeper blue-ward of the 500 nm reference than red-ward of it, and
    # shallower the further from the Sun one looks.
    eps = np.arccos(
        np.clip(
            np.cos(alpha_folded) * np.cos(beta_abs),
            -1.0,
            1.0,
        )
    )
    elon_deg = np.clip(np.rad2deg(eps), *_ELONGATION_RANGE)
    # 0 at the near end of the elongation range, 1 at the far end.
    far_frac = (elon_deg - _ELONGATION_RANGE[0]) / (_ELONGATION_RANGE[1] - _ELONGATION_RANGE[0])

    wvl_arr = np.asarray(wvls)
    is_blue = wvl_arr < _REFERENCE_WAVELENGTH
    slope_near = np.where(is_blue, _SLOPE_NEAR[0], _SLOPE_NEAR[1])
    slope_far = np.where(is_blue, _SLOPE_FAR[0], _SLOPE_FAR[1])
    slope = slope_near + far_frac[:, None] * (slope_far - slope_near)

    color_corr = 1 + slope * np.log10(wvl_arr / _REFERENCE_WAVELENGTH)

    return jnp.asarray(weights[:, None] * color_corr)


def _zodi_model_fn(base_spectra):
    """Multiply the Leinert weights and colour correction by a reference spectrum.

    Parameters
    ----------
    base_spectra : jax.Array, shape (n_wvl,)

    Returns
    -------
    callable
        ``(params, conditions) -> radiance``.
    """

    def fn(_params, conditions):
        if conditions is None:
            return base_spectra
        return conditions * base_spectra

    return fn


class ZodiacalLight(Emitter):
    """Zodiacal light with the Leinert spatial model.

    Parameters
    ----------
    geo : Geometry
        Resolution configuration.
    spectral_model : SpectralModel
    brightness : array-like or None
        Fittable overall amplitude; see :class:`~nyx.emitter.base.Emitter`.
        This emitter has no other free parameter, so it is the only way to
        fit the zodiacal level.
    transform : str or None
        Domain of *brightness*.
    """

    def __init__(self, geo, spectral_model: SpectralModel, brightness=None, transform="log"):
        super().__init__(geo, spectral_model, brightness, transform)
        wvls = geo.wvls

        # Load Leinert table
        zod = np.genfromtxt(ASSETS_PATH + "leinert1998_zodiacal_light.dat", delimiter=",")
        self._leinert_points = (
            np.deg2rad(zod[1:, 0]),
            np.deg2rad(zod[0, 1:]),
        )
        self._leinert_values = zod[1:, 1:]

        self._wvls = wvls

    def _prepare(self, obs) -> SourceObsData:
        """Precompute Leinert weights per HEALPix pixel per observation.

        Parameters
        ----------
        obs : Observation

        Returns
        -------
        SourceObsData
            ``diffuse_conditions`` holds the weights times colour correction.
        """
        diffuse_list = []
        for i in range(obs.nobs):
            # Get ecliptic coords for this observation's sky hemisphere
            sref_coords = obs.get_sky_coords("sref")[i]
            alpha = sref_coords.alpha.rad
            beta = sref_coords.beta.rad

            # Leinert interpolation + color correction
            weights = _leinert_weights(
                alpha,
                beta,
                self._leinert_points,
                self._leinert_values,
                self._wvls,
            )
            diffuse_list.append(weights)

        return SourceObsData(
            diffuse_conditions=jnp.stack(diffuse_list),  # (nobs, nsky, n_wvl)
            per_obs_fields=("diffuse_conditions",),
        )

    @classmethod
    def from_leinert1998(cls, geo, **kwargs) -> "ZodiacalLight":
        """Zodiacal light with a solar spectrum, after Leinert et al. (1998).

        Parameters
        ----------
        geo : Geometry
        **kwargs
            Passed to :class:`ZodiacalLight`, e.g. ``brightness``.

        Returns
        -------
        ZodiacalLight
        """
        wvls = geo.wvls

        spec_samp = load_solar_flux(wvls, normalize_at=500.0)

        ref_flux = np.ones_like(np.asarray(wvls)) * (1e-8 * u.W / u.m**2 / u.sr / u.micron)
        energy_to_photon = energy_flux_to_photon_flux(wvls, ref_flux)

        base_spectra = jnp.asarray(spec_samp * energy_to_photon)
        spectral_model = ParametricSpectrum(
            params=None,
            _model_fn=_zodi_model_fn(base_spectra),
        )
        return cls(geo, spectral_model, **kwargs)
