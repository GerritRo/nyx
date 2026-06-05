import astropy.units as u
import jax.numpy as jnp
import numpy as np

from nyx import ASSETS_PATH
from nyx.core.parameter import Parameter
from nyx.core.protocols import SourceObsData
from nyx.core.spectral import ParametricSpectrum, SpectralModel, resample_flux
from nyx.core.units import RADIANCE
from nyx.emitter._base import BaseEmitter


def _airglow_model_fn(base_spectra):
    """Create airglow model function: SFU-scaled van Rhijn x base spectrum.

    Parameters
    ----------
    base_spectra : jax.Array, shape (n_wvl,)
        Base airglow spectral radiance (captured in closure).

    Returns
    -------
    callable
        ``(sfu, conditions) -> radiance`` where *sfu* is a trainable
        scalar and *conditions* are van Rhijn weights ``(..., 1)``.
    """

    def fn(sfu, conditions):
        scaling = 0.2 + 0.00614 * sfu
        if conditions is None:
            return scaling * base_spectra
        return scaling * conditions * base_spectra

    return fn


def _van_rhijn(altitude, height_km):
    """Van Rhijn function for airglow zenith dependence.

    Parameters
    ----------
    altitude : array
        Altitude angle in radians.
    height_km : float
        Emission layer height in km.

    Returns
    -------
    array
        Van Rhijn weight at each position.
    """
    R_earth = 6378.0  # km
    zenith = np.pi / 2 - altitude
    return 1.0 / np.sqrt(1 - (R_earth / (R_earth + height_km)) ** 2 * np.sin(zenith) ** 2)


class Airglow(BaseEmitter):
    """Airglow source with van Rhijn weighting.

    Parameters
    ----------
    geo : Geometry
        Resolution configuration (provides wavelengths).
    spectral_model : SpectralModel
        Spectral model for airglow emission.
    height_km : float
        Airglow emission layer height in km.
    """

    def __init__(self, geo, spectral_model: SpectralModel, height_km: float = 90.0):
        self._spectral_model = spectral_model
        self._height_km = height_km

    def prepare(self, obs) -> SourceObsData:
        """Precompute van Rhijn weights on the HEALPix hemisphere.

        Parameters
        ----------
        obs : Observation

        Returns
        -------
        SourceObsData
            With van Rhijn weights as diffuse_conditions.
        """
        altitudes = obs.geom.lat  # (nsky,) HEALPix altitudes in rad
        vr = _van_rhijn(altitudes, self._height_km)
        conditions = vr[:, None]  # (nsky, 1), broadcasts with (n_wvl,)

        # Same for all observations (AltAz grid is fixed)
        return SourceObsData(
            diffuse_conditions=jnp.broadcast_to(
                conditions[None],
                (obs.nobs,) + conditions.shape,
            ),
            _per_obs=("diffuse_conditions",),
        )

    @classmethod
    def from_eso_skycalc(cls, geo, sfu: float = 100.0, height_km: float = 90.0) -> "Airglow":
        """Airglow with ESO SkyCalc spectrum + SFU scaling.

        Trainable: sfu (solar flux units) via ParametricSpectrum.

        Parameters
        ----------
        geo : Geometry
            Resolution configuration (provides wavelengths).
        sfu : float
            Initial solar flux units value.
        height_km : float
            Airglow emission layer height in km.
        """
        wvls = geo.wvls

        ag_array = np.genfromtxt(ASSETS_PATH + "eso_skycalc_airglow_130sfu.dat")
        wvl_src = ag_array[:, 0] * u.nm
        flx_src = ag_array[:, 1] * u.ph / u.s / u.m**2 / u.micron / u.arcsec**2

        radiance_values = flx_src.to(RADIANCE).value
        base_flux = resample_flux(wvl_src.to(u.nm).value, radiance_values, wvls)

        spectral_model = ParametricSpectrum(
            params=Parameter.from_value(float(sfu)),
            _model_fn=_airglow_model_fn(jnp.asarray(base_flux)),
        )
        return cls(geo, spectral_model, height_km)
