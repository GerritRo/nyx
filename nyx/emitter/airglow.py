"""Airglow: the upper atmosphere's own emission, after ESO SkyCalc."""

import astropy.units as u
import jax.numpy as jnp
import numpy as np
from astropy.time import Time

from nyx import ASSETS_PATH
from nyx.atmosphere.components import thin_shell_airmass
from nyx.core.parameter import Parameter
from nyx.core.records import SourceObsData
from nyx.core.units import RADIANCE
from nyx.emitter.base import Emitter
from nyx.utils.spectra import ParametricSpectrum, SpectralModel, resample_flux

__all__ = ["Airglow"]


# ---- internals


def _airglow_model_fn(base_spectra):
    """Close over ``base_spectra`` to give SFU(t) x van Rhijn x spectrum.

    Parameters
    ----------
    base_spectra : jax.Array, shape (n_wvl,)

    Returns
    -------
    callable
        ``(coeffs, conditions) -> radiance``, with *coeffs* the SFU
        polynomial lowest order first and *conditions* of shape ``(..., 2)``
        pairing the van Rhijn weight with the hours since the epoch.
    """

    def fn(coeffs, conditions):
        if conditions is None:
            return (0.2 + 0.00614 * jnp.ravel(coeffs)[0]) * base_spectra
        van_rhijn, hours = conditions[..., :1], conditions[..., 1:2]
        sfu = jnp.polyval(jnp.atleast_1d(coeffs)[::-1], hours)
        return (0.2 + 0.00614 * sfu) * van_rhijn * base_spectra

    return fn


def _van_rhijn(altitude, height_km):
    """Van Rhijn weight: path length through a thin shell, relative to vertical.

    Parameters
    ----------
    altitude : array-like
        Radians.
    height_km : float

    Returns
    -------
    numpy.ndarray
    """
    return thin_shell_airmass(np.pi / 2 - np.asarray(altitude), height_km)


class Airglow(Emitter):
    """Airglow source with van Rhijn weighting.

    Alone among nyx's emitters this one takes no ``brightness``: its spectral
    model already evaluates ``0.2 + 0.00614 * sfu(t)``, so a multiplicative
    amplitude on top would be all but degenerate with ``sfu``'s constant term
    and would leave the pair unidentifiable.  Fit the SFU curve instead.

    Parameters
    ----------
    geo : Geometry
        Resolution configuration.
    spectral_model : SpectralModel
    height_km : float
        Emission layer height in km.
    t_ref : astropy.time.Time or None
        Epoch the light curve is measured from; ``None`` adopts the first
        observation time.

    Notes
    -----
    :meth:`prepare` is not pure: when *t_ref* is ``None`` the first
    observation this emitter ever sees fixes the epoch of the light curve,
    for good.  Pass *t_ref* explicitly to pin it.
    """

    def __init__(
        self,
        geo,
        spectral_model: SpectralModel,
        height_km: float = 90.0,
        t_ref: Time | None = None,
    ):
        super().__init__(geo, spectral_model)
        self._height_km = height_km
        self._t_ref = t_ref

    @property
    def t_ref(self) -> Time | None:
        """Epoch the light curve is measured from, adopted on first :meth:`prepare`."""
        return self._t_ref

    def _elapsed_hours(self, times: Time) -> np.ndarray:
        """Hours from :attr:`t_ref` to each of *times*, fixing the epoch if unset."""
        if self._t_ref is None:
            self._t_ref = times[0]
        return np.asarray((times - self._t_ref).to_value(u.hour), dtype=float)

    def _prepare(self, obs) -> SourceObsData:
        """Precompute van Rhijn weights and observation times.

        Parameters
        ----------
        obs : Observation

        Returns
        -------
        SourceObsData
            ``diffuse_conditions`` is ``(nobs, nsky, 2)``: van Rhijn weight
            per sky pixel, and hours since :attr:`t_ref`.
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
            per_obs_fields=("diffuse_conditions",),
        )

    def _repr_parts(self) -> list[str]:
        # An unset t_ref is not worth showing: prepare will adopt one.
        parts = [f"height={self._height_km}"]
        if self._t_ref is not None:
            parts.append(f"t_ref={self._t_ref}")
        return parts

    @classmethod
    def from_eso_skycalc(
        cls,
        geo,
        sfu: float = 100.0,
        height_km: float = 90.0,
        drift_order: int = 0,
        t_ref: Time | None = None,
    ) -> "Airglow":
        """Airglow with an ESO SkyCalc spectrum and a trainable SFU light curve.

        The curve is ``sfu(t) = params[0] + params[1] * dt + ...`` in hours
        since *t_ref*.

        Parameters
        ----------
        geo : Geometry
        sfu : float
            Initial solar flux units, i.e. ``params[0]``; higher coefficients
            start at zero.
        height_km : float
            Emission layer height in km.
        drift_order : int
            Polynomial order of the light curve; 0 leaves a scalar SFU.
        t_ref : astropy.time.Time or None
            Epoch the polynomial is measured from.  ``None`` adopts the first
            observation time the emitter is prepared for.

        Returns
        -------
        Airglow
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
