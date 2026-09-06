import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from nyx import ASSETS_PATH
from nyx.core.parameter import Parameter
from nyx.core.spectral import resample_flux

_R_EARTH_KM = 6378.0

_SEA_LEVEL_HPA = 1013.25


def plane_parallel(Z):
    """
    Plane-parallel airmass. Z is zenith angle in radians.
    """
    return 1 / jnp.maximum(jnp.cos(Z), 0.025)


def kasten_young_1989(Z):
    """
    Kasten & Young (1989) airmass formula. Z is zenith angle in radians.
    """
    Z_safe = jnp.minimum(Z, jnp.pi / 2)
    return 1 / (jnp.cos(Z_safe) + 0.50572 * (96.07995 - jnp.rad2deg(Z_safe)) ** (-1.6364))


def thin_shell_airmass(zenith, height_km):
    """
    Slant path through a thin shell at *height_km*, relative to vertical.
    """
    z = jnp.minimum(jnp.asarray(zenith), jnp.pi / 2)
    ratio = _R_EARTH_KM / (_R_EARTH_KM + height_km)
    return 1.0 / jnp.sqrt(1.0 - ratio**2 * jnp.sin(z) ** 2)


AIRMASS_FUNCTIONS = {
    "plane_parallel": plane_parallel,
    "kasten_young_1989": kasten_young_1989,
}


# Optical-depth formulas


def _rayleigh_tau_sea_level(wavelengths_nm):
    """Sea-level Rayleigh optical depth spectrum, Hansen & Travis (1974)."""
    lam = jnp.asarray(wavelengths_nm) / 1000.0  # micron
    return 0.008569 * lam**-4 * (1 + 0.0113 * lam**-2 + 0.00013 * lam**-4)


def _column_scaling(height_km, pressure_hpa, scale_height_km):
    """Fraction of the sea-level air column that lies above the observer."""
    if pressure_hpa is not None:
        return pressure_hpa / _SEA_LEVEL_HPA
    return jnp.exp(-height_km / scale_height_km)


def tau_rayleigh(wavelengths_nm, height_km, pressure_hpa=None, scale_height_km=8.0):
    """Rayleigh optical depth of the air column above the observer.

    Uses the Hansen & Travis (1974) parameterisation at 1013.25 hPa.

    Parameters
    ----------
    wavelengths_nm : array
        Wavelengths in nm.
    height_km : array or float
        Observer height above sea level, in km.
    pressure_hpa : float or None
        Station pressure.
    scale_height_km : float
        Scale height of the barometric fallback ``exp(-height / H)``.
    """
    return _rayleigh_tau_sea_level(wavelengths_nm) * _column_scaling(
        height_km, pressure_hpa, scale_height_km
    )


def tau_mie(wavelengths_nm, height_km, aod_500, angstrom_exp):
    """Mie (aerosol) optical depth with Angstrom power-law scaling."""
    return aod_500 * (wavelengths_nm / 500) ** (-angstrom_exp) * jnp.exp(-height_km / 1.54)


# Phase functions and gradation


def rayleigh_phase(cos_theta):
    """Rayleigh scattering phase function."""
    return 1 / (4 * jnp.pi) * 3 / 4 * (1 + cos_theta**2)


def henyey_greenstein_phase(cos_theta, g):
    """Safe Henyey-Greenstein scattering phase function."""
    gsq = g**2
    base = 1 + gsq - 2 * g * cos_theta
    safe_base = jnp.maximum(base, jnp.finfo(base.dtype).eps)
    return 1 / (4 * jnp.pi) * (1 - gsq) / safe_base**1.5


def gradation_function(tau, sec_Z, sec_z):
    """Safe atmospheric gradation function for single scattering."""
    scale = jnp.maximum(jnp.abs(sec_z), jnp.abs(sec_Z))
    close = jnp.abs(sec_z - sec_Z) < jnp.sqrt(jnp.finfo(sec_z.dtype).eps) * scale
    safe_denom = jnp.where(close, jnp.ones_like(sec_z), sec_z - sec_Z)
    sec_diff = sec_Z / safe_denom
    exp_diff = jnp.exp(-tau * sec_Z) - jnp.exp(-tau * sec_z)
    return jnp.where(
        ~close,
        sec_diff * exp_diff,
        tau * sec_Z * jnp.exp(-tau * sec_Z),
    )


# Scattering components


class ScatteringComponent(eqx.Module):
    """Atmospheric scattering or absorption component."""

    _rendering_wvls: jax.Array

    @property
    def ssa(self) -> float:
        """
        Single-scattering albedo: the fraction of ``tau`` that scatters.
        """
        return 1.0

    def tau(self, height_km) -> jax.Array:
        raise NotImplementedError

    def phase(self, cos_scattering_angle) -> jax.Array:
        raise NotImplementedError

    def airmass(self, zenith, default_airmass):
        """Relative airmass of this component at zenith angle *zenith*.

        Defaults to the atmosphere's shared airmass formula, which is the
        right one for a species mixed through the troposphere.  A
        component confined to a narrow layer far above it overrides this;
        see :class:`TabulatedAbsorption`.
        """
        return default_airmass(zenith)


class RayleighComponent(ScatteringComponent):
    """Rayleigh molecular scattering."""

    _tau_shape: jax.Array
    _pressure_hpa: float | None = eqx.field(static=True)
    _scale_height_km: float = eqx.field(static=True)

    def __init__(self, rendering_wvls, pressure_hpa=None, scale_height_km=8.0):
        """
        Parameters
        ----------
        rendering_wvls : array
            Wavelength grid in nm.
        pressure_hpa : float or None
            Station pressure.  When given it fixes the air column
            directly and the observer height is ignored; otherwise the
            column falls back to ``exp(-height / scale_height_km)``.
        scale_height_km : float
            Scale height of the barometric fallback.
        """
        self._rendering_wvls = jnp.asarray(rendering_wvls)
        self._tau_shape = _rayleigh_tau_sea_level(self._rendering_wvls)
        self._pressure_hpa = None if pressure_hpa is None else float(pressure_hpa)
        self._scale_height_km = float(scale_height_km)

    def tau(self, height_km):
        return self._tau_shape * _column_scaling(
            height_km, self._pressure_hpa, self._scale_height_km
        )

    def phase(self, cos_scattering_angle):
        return rayleigh_phase(cos_scattering_angle)


class HenyeyGreensteinComponent(ScatteringComponent):
    """
    Mie/aerosol scattering with Henyey-Greenstein phase function.
    """

    aod_500: Parameter
    angstrom_exp: Parameter
    hg_asymmetry: Parameter
    _ssa: float = eqx.field(static=True)

    def __init__(
        self,
        rendering_wvls,
        aod_500=0.1,
        angstrom_exp=1.5,
        hg_asymmetry=0.75,
        hg_ssa=0.9,
    ):
        """
        Parameters
        ----------
        rendering_wvls : array
            Wavelength grid in nm.
        aod_500 : float
            Aerosol optical depth at 500 nm.
        angstrom_exp : float
            Angstrom exponent.
        hg_asymmetry : float
            Henyey-Greenstein asymmetry parameter.
        hg_ssa : float
            Single-scattering albedo.
        """
        self._rendering_wvls = jnp.asarray(rendering_wvls)
        self.aod_500 = Parameter.from_value(aod_500, transform="log")
        self.angstrom_exp = Parameter.from_value(angstrom_exp, transform="softplus")
        self.hg_asymmetry = Parameter.from_value(hg_asymmetry, transform="tanh")
        self._ssa = float(hg_ssa)

    @property
    def ssa(self) -> float:
        """Aerosol single-scattering albedo."""
        return self._ssa

    def tau(self, height_km):
        tau = (
            self.aod_500.value
            * (self._rendering_wvls / 500) ** (-self.angstrom_exp.value)
            * jnp.exp(-height_km / 1.54)
        )
        return tau

    def phase(self, cos_scattering_angle):
        return henyey_greenstein_phase(cos_scattering_angle, self.hg_asymmetry.value)


class TabulatedAbsorption(ScatteringComponent):
    """Pure absorption from a tabulated optical-depth spectrum.

    Pass ``jnp.inf`` for a column that does not vary with observer
      altitude (e.g. stratospheric O₃).
    """

    _tau_shape: jax.Array
    _scale_height_km: float = eqx.field(static=True)
    _layer_height_km: float | None = eqx.field(static=True)

    def __init__(
        self,
        wvl_table,
        transmission_table,
        rendering_wvls,
        scale_height_km,
        layer_height_km=None,
    ):
        self._rendering_wvls = jnp.asarray(rendering_wvls)
        self._tau_shape = -jnp.log(
            resample_flux(
                jnp.asarray(wvl_table),
                jnp.asarray(transmission_table),
                self._rendering_wvls,
                method="conserve",
            )
        )
        self._scale_height_km = float(scale_height_km)
        self._layer_height_km = None if layer_height_km is None else float(layer_height_km)

    @property
    def ssa(self) -> float:
        """Zero: this component absorbs and does not scatter."""
        return 0.0

    def tau(self, height_km):
        return self._tau_shape * jnp.exp(-height_km / self._scale_height_km)

    def phase(self, cos_scattering_angle):
        return jnp.zeros_like(cos_scattering_angle)

    def airmass(self, zenith, default_airmass):
        if self._layer_height_km is None:
            return default_airmass(zenith)
        return thin_shell_airmass(zenith, self._layer_height_km)


# Species factories for TabulatedAbsorption


def tau_ozone(rendering_wvls, layer_height_km=25.0):
    """Ozone absorption from tabulated cross-sections.

    Parameters
    ----------
    rendering_wvls : array
        Wavelength grid in nm.
    layer_height_km : float or None
        Height of the ozone layer.
    """
    o3_table = np.genfromtxt(ASSETS_PATH + "eso_skycalc_ozone_absorption.dat")

    return TabulatedAbsorption(
        jnp.array(o3_table[:, 0]),
        jnp.array(o3_table[:, 1]),
        rendering_wvls,
        scale_height_km=jnp.inf,
        layer_height_km=layer_height_km,
    )
