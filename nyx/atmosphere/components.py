import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from nyx import ASSETS_PATH
from nyx.core.parameter import Parameter
from nyx.core.spectral import resample_flux

# Airmass formulas


def plane_parallel(Z):
    """Plane-parallel airmass. Z is zenith angle in radians."""
    return 1 / jnp.maximum(jnp.cos(Z), 0.025)


def kasten_young_1989(Z):
    """Kasten & Young (1989) airmass formula. Z is zenith angle in radians."""
    Z_safe = jnp.minimum(Z, jnp.deg2rad(95.0))
    return 1 / (jnp.cos(Z_safe) + 0.50572 * (96.07995 - jnp.rad2deg(Z_safe)) ** (-1.6364))


AIRMASS_FUNCTIONS = {
    "plane_parallel": plane_parallel,
    "kasten_young_1989": kasten_young_1989,
}


# Optical-depth formulas


def tau_rayleigh(wavelengths_nm, height_km):
    """Rayleigh optical depth (standard clear-air model)."""
    return 0.00878 * (wavelengths_nm / 1000) ** -4.09 * jnp.exp(-height_km / 8)


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

    def tau(self, height_km) -> jax.Array:
        raise NotImplementedError

    def phase(self, cos_scattering_angle) -> jax.Array:
        raise NotImplementedError


class RayleighComponent(ScatteringComponent):
    """Rayleigh molecular scattering."""

    _tau_shape: jax.Array

    def __init__(self, rendering_wvls):
        self._rendering_wvls = jnp.asarray(rendering_wvls)
        self._tau_shape = 0.00878 * (self._rendering_wvls / 1000) ** -4.09

    def tau(self, height_km):
        return self._tau_shape * jnp.exp(-height_km / 8)

    def phase(self, cos_scattering_angle):
        return rayleigh_phase(cos_scattering_angle)


class HenyeyGreensteinComponent(ScatteringComponent):
    """Mie/aerosol scattering with Henyey-Greenstein phase function."""

    aod_500: Parameter
    angstrom_exp: Parameter
    hg_asymmetry: Parameter
    _fine_wvls: jax.Array

    def __init__(
        self, rendering_wvls, aod_500=0.1, angstrom_exp=1.5, hg_asymmetry=0.75, oversample=10
    ):
        self._rendering_wvls = jnp.asarray(rendering_wvls)
        self._fine_wvls = jnp.linspace(
            self._rendering_wvls[0],
            self._rendering_wvls[-1],
            oversample * len(self._rendering_wvls),
        )
        self.aod_500 = Parameter.from_value(aod_500, scale=0.5)
        self.angstrom_exp = Parameter.from_value(angstrom_exp, scale=1.0)
        self.hg_asymmetry = Parameter.from_value(hg_asymmetry, scale=0.5)

    def tau(self, height_km):
        t_fine = (
            self.aod_500.value
            * (self._fine_wvls / 500) ** (-self.angstrom_exp.value)
            * jnp.exp(-height_km / 1.54)
        )
        return resample_flux(self._fine_wvls, t_fine, self._rendering_wvls, method="conserve")

    def phase(self, cos_scattering_angle):
        return henyey_greenstein_phase(cos_scattering_angle, self.hg_asymmetry.value)


class TabulatedAbsorption(ScatteringComponent):
    """Pure absorption from a tabulated optical-depth spectrum.

    Pass ``jnp.inf`` for a column that does not vary with observer
      altitude (e.g. stratospheric O₃).
    """

    _tau_shape: jax.Array
    _scale_height_km: float = eqx.field(static=True)

    def __init__(self, wvl_table, transmission_table, rendering_wvls, scale_height_km):
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

    def tau(self, height_km):
        return self._tau_shape * jnp.exp(-height_km / self._scale_height_km)

    def phase(self, cos_scattering_angle):
        return jnp.zeros_like(cos_scattering_angle)


# Species factories for TabulatedAbsorption


def tau_ozone(rendering_wvls):
    """Ozone absorption from tabulated cross-sections."""
    o3_table = np.genfromtxt(ASSETS_PATH + "eso_skycalc_ozone_absorption.dat")

    return TabulatedAbsorption(
        jnp.array(o3_table[:, 0]),
        jnp.array(o3_table[:, 1]),
        rendering_wvls,
        scale_height_km=jnp.inf,
    )
