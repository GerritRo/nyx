import equinox as eqx
import jax.numpy as jnp
import numpy as np
import astropy.units as u
from typing import Union

from .specification import RADIANCE, FLUX, WAVELENGTH, ANGLE


# Type alias for input values
ArrayLike = Union[float, np.ndarray, jnp.ndarray, u.Quantity]


class Wavelength(eqx.Module):
    """
    Wavelength array stored internally in nanometers.
    """
    value: jnp.ndarray

    def __init__(self, input_value: ArrayLike):
        if isinstance(input_value, u.Quantity):
            self.value = jnp.asarray(input_value.to(WAVELENGTH).value)
        else:
            # Assume already in nm if no units
            self.value = jnp.asarray(input_value)

    def to(self, unit: u.Unit) -> u.Quantity:
        """Convert back to astropy Quantity in specified unit."""
        return (np.asarray(self.value) * WAVELENGTH).to(unit)

    def __repr__(self):
        return f"Wavelength({self.value.shape} nm)"


class Angle(eqx.Module):
    """
    Angle array stored internally in radians.
    """
    value: jnp.ndarray

    def __init__(self, input_value: ArrayLike):
        if isinstance(input_value, u.Quantity):
            self.value = jnp.asarray(input_value.to(ANGLE).value)
        else:
            # Assume already in radians if no units
            self.value = jnp.asarray(input_value)

    def to(self, unit: u.Unit) -> u.Quantity:
        """Convert back to astropy Quantity in specified unit."""
        return (np.asarray(self.value) * ANGLE).to(unit)

    def __repr__(self):
        return f"Angle({self.value.shape} rad)"


class Radiance(eqx.Module):
    """
    Diffuse flux (radiance) stored internally in photon/s/m²/nm/sr.
    """
    value: jnp.ndarray

    def __init__(self, input_value: ArrayLike):
        if isinstance(input_value, u.Quantity):
            # Must be in photon units - convert to internal representation
            if u.photon not in input_value.unit.bases:
                raise ValueError(
                    f"Radiance requires photon units, got {input_value.unit}. "
                    "For energy flux conversion, use Spectrum.from_radiance() instead."
                )
            self.value = jnp.asarray(input_value.to(RADIANCE).value)
        else:
            # Assume already in internal units
            self.value = jnp.asarray(input_value)

    def to(self, unit: u.Unit) -> u.Quantity:
        """Convert back to astropy Quantity in specified unit."""
        return (np.asarray(self.value) * RADIANCE).to(unit)

    def __repr__(self):
        return f"Radiance({self.value.shape} ph/s/m²/nm/sr)"


class Flux(eqx.Module):
    """
    Point source flux stored internally in photon/s/m²/nm.
    """
    value: jnp.ndarray

    def __init__(self, input_value: ArrayLike):
        if isinstance(input_value, u.Quantity):
            # Must be in photon units - convert to internal representation
            if u.photon not in input_value.unit.bases:
                raise ValueError(
                    f"Flux requires photon units, got {input_value.unit}. "
                    "For energy flux conversion, use Spectrum.from_energy_flux() instead."
                )
            self.value = jnp.asarray(input_value.to(FLUX).value)
        else:
            # Assume already in internal units
            self.value = jnp.asarray(input_value)

    def to(self, unit: u.Unit) -> u.Quantity:
        """Convert back to astropy Quantity in specified unit."""
        return (np.asarray(self.value) * FLUX).to(unit)

    def __repr__(self):
        return f"Flux({self.value.shape} ph/s/m²/nm)"