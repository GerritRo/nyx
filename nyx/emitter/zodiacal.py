import astropy.units as u
import jax.numpy as jnp
import numpy as np
from jax.scipy.interpolate import RegularGridInterpolator

from nyx import ASSETS_PATH
from nyx.core.protocols import SourceObsData
from nyx.core.spectral import ParametricSpectrum, SpectralModel
from nyx.core.units import energy_flux_to_photon_flux
from nyx.emitter._base import BaseEmitter
from nyx.utils.spectra import load_solar_flux


def _leinert_weights(alpha, beta, leinert_points, leinert_values, wvls):
    """Compute Leinert zodiacal light weights with color correction.

    Parameters
    ----------
    alpha : array
        Ecliptic longitude in radians, shape (nsky,).
    beta : array
        Ecliptic latitude in radians, shape (nsky,).
    leinert_points : tuple
        (alpha_grid, beta_grid) as arrays in radians.
    leinert_values : array
        Leinert table values, shape (n_alpha, n_beta).
    wvls : array
        Wavelengths in nm, shape (n_wvl,).

    Returns
    -------
    array
        Weight array, shape (nsky, n_wvl).
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

    # Color correction
    eps = np.arccos(
        np.clip(
            np.cos(alpha_folded) * np.cos(beta_abs),
            -1.0,
            1.0,
        )
    )
    elon_deg = np.clip(np.rad2deg(eps), 30, 90)
    elon_f = -0.3 * (elon_deg - 30) / 60
    color_corr = 1 + (1.2 + elon_f[:, None]) * np.log10(np.asarray(wvls) / 500)

    return jnp.asarray(weights[:, None] * color_corr)


def _zodi_model_fn(base_spectra):
    """Multiply Leinert weights x color correction by the reference spectrum."""

    def fn(_params, conditions):
        if conditions is None:
            return base_spectra
        return conditions * base_spectra

    return fn


class ZodiacalLight(BaseEmitter):
    """Zodiacal light source with Leinert spatial model.

    No trainable parameters.

    Parameters
    ----------
    geo : Geometry
        Resolution configuration (provides wavelengths).
    spectral_model : SpectralModel
        Spectral model for the zodiacal light spectrum.
    """

    def __init__(self, geo, spectral_model: SpectralModel):
        wvls = geo.wvls

        # Load Leinert table
        zod = np.genfromtxt(ASSETS_PATH + "leinert1998_zodiacal_light.dat", delimiter=",")
        self._leinert_points = (
            np.deg2rad(zod[1:, 0]),
            np.deg2rad(zod[0, 1:]),
        )
        self._leinert_values = zod[1:, 1:]

        self._wvls = wvls
        self._spectral_model = spectral_model

    def prepare(self, obs) -> SourceObsData:
        """Precompute Leinert weights per HEALPix pixel per observation.

        Parameters
        ----------
        obs : Observation

        Returns
        -------
        SourceObsData
            With Leinert weights x color correction as diffuse_conditions.
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
            _per_obs=("diffuse_conditions",),
        )

    @classmethod
    def from_leinert1998(cls, geo) -> "ZodiacalLight":
        """Zodiacal light with solar spectrum (Leinert et al. 1998).

        Parameters
        ----------
        geo : Geometry
            Resolution configuration (provides wavelengths).
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
        return cls(geo, spectral_model)
