import astropy.units as u
import jax.numpy as jnp
import numpy as np
from astropy.time import Time

from nyx import ASSETS_PATH
from nyx.atmosphere.components import thin_shell_airmass
from nyx.core.parameter import Parameter
from nyx.core.protocols import SourceObsData
from nyx.core.spectral import ParametricSpectrum, SpectralModel, resample_flux
from nyx.core.units import RADIANCE
from nyx.emitter._base import BaseEmitter


def _airglow_model_fn(base_spectra):
    """Close over ``base_spectra`` (n_wvl,) to give SFU(t) x van Rhijn x spectrum.

    The returned ``(coeffs, conditions) -> radiance`` takes the trainable
    SFU polynomial coefficients lowest order first -- a scalar being a
    constant SFU -- and conditions ``(..., 2)`` pairing the van Rhijn
    weight with the hours since the reference epoch, the latter identical
    across sky pixels.
    """

    def fn(coeffs, conditions):
        if conditions is None:
            return (0.2 + 0.00614 * jnp.ravel(coeffs)[0]) * base_spectra
        van_rhijn, hours = conditions[..., :1], conditions[..., 1:2]
        sfu = jnp.polyval(jnp.atleast_1d(coeffs)[::-1], hours)
        return (0.2 + 0.00614 * sfu) * van_rhijn * base_spectra

    return fn


def _van_rhijn(altitude, height_km):
    """Van Rhijn weight at each altitude (radians).

    The path length through a thin emitting shell at ``height_km``,
    relative to the vertical one.
    """
    return thin_shell_airmass(np.pi / 2 - np.asarray(altitude), height_km)


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
    t_ref : astropy.time.Time or None
        Epoch the light curve is measured from.  ``None`` adopts the first
        observation time the emitter is prepared for; see :attr:`t_ref`.
    """

    def __init__(
        self,
        geo,
        spectral_model: SpectralModel,
        height_km: float = 90.0,
        t_ref: Time | None = None,
    ):
        self._spectral_model = spectral_model
        self._height_km = height_km
        self._geo_signature = geo.signature
        self._t_ref = t_ref

    @property
    def t_ref(self) -> Time | None:
        """Epoch the light curve is measured from.

        Fixed at construction, or adopted from the first observation the
        emitter is prepared for and reused for every later one.
        """
        return self._t_ref

    def _elapsed_hours(self, times: Time) -> np.ndarray:
        """Hours from :attr:`t_ref` to each of *times*, fixing the epoch if unset."""
        if self._t_ref is None:
            self._t_ref = times[0]
        return np.asarray((times - self._t_ref).to_value(u.hour), dtype=float)

    def prepare(self, obs) -> SourceObsData:
        """Precompute van Rhijn weights and observation times.

        ``diffuse_conditions`` comes out ``(nobs, nsky, 2)``: the van Rhijn
        weight per sky pixel, and the hours since :attr:`t_ref`.
        """
        altitudes = obs.geom.lat  # (nsky,) HEALPix altitudes in rad
        vr = _van_rhijn(altitudes, self._height_km)  # (nsky,), fixed in AltAz
        hours = self._elapsed_hours(obs.times)  # (nobs,)

        shape = (obs.nobs, vr.size)
        return SourceObsData(
            diffuse_conditions=jnp.stack(
                [
                    jnp.broadcast_to(jnp.asarray(vr)[None, :], shape),
                    jnp.broadcast_to(jnp.asarray(hours)[:, None], shape),
                ],
                axis=-1,
            ),
            per_obs=("diffuse_conditions",),
        )

    @classmethod
    def from_eso_skycalc(
        cls,
        geo,
        sfu: float = 100.0,
        height_km: float = 90.0,
        drift_order: int = 0,
        t_ref: Time | None = None,
    ) -> "Airglow":
        """Airglow with ESO SkyCalc spectrum + SFU scaling.

        Trainable: the SFU light curve, as polynomial coefficients in the
        hours since *t_ref*::

            sfu(t) = params[0] + params[1] * dt + params[2] * dt**2 + ...

        ``drift_order=0`` (default) leaves ``params`` a single scalar SFU.

        Parameters
        ----------
        geo : Geometry
        sfu : float
            Initial solar flux units, i.e. ``params[0]``; the higher
            coefficients start at zero, so the initial curve is flat.
        height_km : float
            Airglow emission layer height in km.
        drift_order : int
            Polynomial order of the light curve.
        t_ref : astropy.time.Time or None
            Epoch the polynomial is measured from. ``None`` adopts the
            first observation time the emitter is prepared for.

        Examples
        --------
        A linear drift, with a prior keeping it to a physically plausible
        few per cent per hour::

            airglow = Airglow.from_eso_skycalc(geo, drift_order=1, t_ref=times[0])

            def residuals(scene):
                coeffs = scene.airglow.spectral_model.params.value
                return {
                    'data': (scene.render()['CT1'] - target) / sigma,
                    'drift': coeffs[1] / (0.05 * coeffs[0]),
                }
        """
        if drift_order < 0:
            raise ValueError(f"drift_order must be >= 0, got {drift_order}")
        wvls = geo.wvls

        ag_array = np.genfromtxt(ASSETS_PATH + "eso_skycalc_airglow_130sfu.dat")
        wvl_src = ag_array[:, 0] * u.nm
        flx_src = ag_array[:, 1] * u.ph / u.s / u.m**2 / u.micron / u.arcsec**2

        radiance_values = flx_src.to(RADIANCE).value
        base_flux = resample_flux(wvl_src.to(u.nm).value, radiance_values, wvls)

        coeffs = float(sfu) if drift_order == 0 else np.r_[float(sfu), np.zeros(drift_order)]
        spectral_model = ParametricSpectrum(
            params=Parameter.from_value(coeffs),
            _model_fn=_airglow_model_fn(jnp.asarray(base_flux)),
        )
        return cls(geo, spectral_model, height_km, t_ref)
