from __future__ import annotations

import astropy.units as u
import jax
import jax.numpy as jnp
import numpy as np
from astropy.constants import c, h
from numpy.typing import ArrayLike

# Internal unit conventions
WAVELENGTH = u.nm
RADIANCE = u.photon / (u.s * u.m**2 * u.nm * u.sr)
FLUX = u.photon / (u.s * u.m**2 * u.nm)
RATE = u.photon / u.s
ANGLE = u.rad
SOLID_ANGLE = u.sr


# Conversions


def to_wavelength_nm(quantity: ArrayLike | u.Quantity) -> jax.Array:
    """Convert any wavelength quantity to a raw array in nm."""
    if isinstance(quantity, u.Quantity):
        return jnp.asarray(quantity.to(WAVELENGTH).value)
    return jnp.asarray(quantity)  # assume already nm


def to_radiance(quantity: ArrayLike | u.Quantity) -> jax.Array:
    """Convert photon-radiance quantity to raw array in internal units."""
    if isinstance(quantity, u.Quantity):
        return jnp.asarray(quantity.to(RADIANCE).value)
    return jnp.asarray(quantity)


def to_flux(quantity: ArrayLike | u.Quantity) -> jax.Array:
    """Convert photon-flux quantity to raw array in internal units."""
    if isinstance(quantity, u.Quantity):
        return jnp.asarray(quantity.to(FLUX).value)
    return jnp.asarray(quantity)


def to_angle_rad(quantity: ArrayLike | u.Quantity) -> jax.Array:
    """Convert any angle quantity to raw array in radians."""
    if isinstance(quantity, u.Quantity):
        return jnp.asarray(quantity.to(ANGLE).value)
    return jnp.asarray(quantity)


def energy_flux_to_photon_flux(wavelength_nm: ArrayLike, energy_flux: u.Quantity) -> jax.Array:
    """
    Convert energy flux (W/m^2/nm) to photon flux (photon/s/m^2/nm).

    Parameters
    ----------
    wavelength_nm : array
        Wavelengths in nm (raw array).
    energy_flux : astropy Quantity
        Energy flux with units (e.g. W/m^2/nm or W/m^2/nm/sr).

    Returns
    -------
    array
        Photon flux in internal units (raw array).
    """
    wvl_q = np.asarray(wavelength_nm) * u.nm
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
