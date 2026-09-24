from __future__ import annotations

import astropy.units as u
import jax
import jax.numpy as jnp
import numpy as np
from astropy.constants import c, h
from numpy.typing import ArrayLike

__all__ = [
    "ANGLE",
    "FLUX",
    "RADIANCE",
    "WAVELENGTH",
    "energy_flux_to_photon_flux",
    "to_angle_rad",
    "to_wavelength_nm",
]

# Internal unit conventions
WAVELENGTH = u.nm
RADIANCE = u.photon / (u.s * u.m**2 * u.nm * u.sr)
FLUX = u.photon / (u.s * u.m**2 * u.nm)
ANGLE = u.rad


def _to_unit(quantity: ArrayLike | u.Quantity, unit: u.UnitBase) -> jax.Array:
    """Raw array of *quantity* in *unit*; a plain array is assumed to be in it.

    Parameters
    ----------
    quantity : array-like or astropy Quantity
    unit : astropy unit

    Returns
    -------
    jax.Array
    """
    if isinstance(quantity, u.Quantity):
        return jnp.asarray(quantity.to(unit).value)
    return jnp.asarray(quantity)


def to_wavelength_nm(quantity: ArrayLike | u.Quantity) -> jax.Array:
    """Convert any wavelength quantity to a raw array in nm.

    Parameters
    ----------
    quantity : array-like or astropy Quantity

    Returns
    -------
    jax.Array
    """
    return _to_unit(quantity, WAVELENGTH)


def to_angle_rad(quantity: ArrayLike | u.Quantity) -> jax.Array:
    """Convert any angle quantity to a raw array in radians.

    Parameters
    ----------
    quantity : array-like or astropy Quantity

    Returns
    -------
    jax.Array
    """
    return _to_unit(quantity, ANGLE)


def energy_flux_to_photon_flux(wvls: ArrayLike, energy_flux: u.Quantity) -> jax.Array:
    """Convert energy flux to photon flux, per steradian or not.

    Parameters
    ----------
    wvls : array-like
        Wavelengths in nm.
    energy_flux : astropy Quantity
        Energy flux, e.g. W/m^2/nm or W/m^2/nm/sr.

    Returns
    -------
    jax.Array
        Photon flux in :data:`FLUX` or :data:`RADIANCE` units.
    """
    wvl_q = np.asarray(wvls) * u.nm
    photon_energy = (h * c / wvl_q).to(u.J)

    # Detect if this is a radiance (has sr in denominator)
    is_radiance = u.sr in energy_flux.unit.bases

    if is_radiance:
        normalized = energy_flux.to(u.W / u.m**2 / u.nm / u.sr)
        photon_radiance = normalized / photon_energy
        return jnp.asarray((photon_radiance * u.ph).to(RADIANCE).value)
    else:
        normalized = energy_flux.to(u.W / u.m**2 / u.nm)
        photon_flux = normalized / photon_energy
        return jnp.asarray((photon_flux * u.ph).to(FLUX).value)
