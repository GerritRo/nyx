import equinox as eqx
import jax.numpy as jnp
import numpy as np
import astropy.units as u
from astropy.constants import c, h
from typing import Union, Optional

from .specification import RADIANCE, FLUX, WAVELENGTH, ANGLE


# Type alias for input values
ArrayLike = Union[float, np.ndarray, jnp.ndarray, u.Quantity]


class Wavelength(eqx.Module):
    """
    Wavelength array stored internally in nanometers.

    JAX-compatible via Equinox. Can be passed through jit, vmap, etc.

    Examples
    --------
    >>> wvl = Wavelength(500 * u.nm)
    >>> wvl.value  # jnp.ndarray in nm

    >>> wvl = Wavelength([400, 500, 600] * u.nm)
    >>> wvl.to(u.micron)  # Back to astropy Quantity
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

    JAX-compatible via Equinox.

    Examples
    --------
    >>> ang = Angle(45 * u.deg)
    >>> ang.value  # jnp.ndarray in radians
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

    JAX-compatible via Equinox. Handles automatic conversion from
    energy flux units (W/m²/nm/sr) to photon flux if wavelength is provided.

    Examples
    --------
    >>> # Direct photon flux
    >>> rad = Radiance(flux_array * u.ph/u.s/u.m**2/u.nm/u.sr)

    >>> # Energy flux with wavelength for conversion
    >>> rad = Radiance(energy_flux * u.W/u.m**2/u.nm/u.sr, wavelength=wvl)
    """
    value: jnp.ndarray

    def __init__(self, input_value: ArrayLike,
                 wavelength: Optional[Union[Wavelength, u.Quantity]] = None):
        if isinstance(input_value, u.Quantity):
            self.value = jnp.asarray(
                self._convert(input_value, wavelength)
            )
        else:
            # Assume already in internal units
            self.value = jnp.asarray(input_value)

    def _convert(self, quantity: u.Quantity,
                 wavelength: Optional[Union[Wavelength, u.Quantity]]) -> np.ndarray:
        """Convert quantity to internal radiance units."""
        # Check if already in photon units
        if u.photon in quantity.unit.bases:
            return quantity.to(RADIANCE).value

        # Energy flux - needs wavelength for conversion
        if wavelength is None:
            raise ValueError(
                f"Cannot convert {quantity.unit} to photon radiance. "
                "Energy flux requires wavelength parameter for conversion."
            )

        # Get wavelength in nm
        if isinstance(wavelength, Wavelength):
            wl_nm = wavelength.value
        elif isinstance(wavelength, u.Quantity):
            wl_nm = wavelength.to(u.nm).value
        else:
            wl_nm = np.asarray(wavelength)

        # Energy per photon: E = h*c/λ
        wl_quantity = wl_nm * u.nm
        photon_energy = (h * c / wl_quantity).to(u.J)

        # Convert energy flux to photon flux
        energy_flux = quantity.to(u.W / u.m**2 / u.nm / u.sr)
        photon_flux = energy_flux / photon_energy
        result = (photon_flux * u.ph).to(RADIANCE)

        return result.value

    def to(self, unit: u.Unit) -> u.Quantity:
        """Convert back to astropy Quantity in specified unit."""
        return (np.asarray(self.value) * RADIANCE).to(unit)

    def __repr__(self):
        return f"Radiance({self.value.shape} ph/s/m²/nm/sr)"


class Flux(eqx.Module):
    """
    Point source flux stored internally in photon/s/m²/nm.

    JAX-compatible via Equinox. Handles automatic conversion from
    energy flux units (W/m²/nm) to photon flux if wavelength is provided.

    Examples
    --------
    >>> # Direct photon flux
    >>> flx = Flux(flux_array * u.ph/u.s/u.m**2/u.nm)

    >>> # Energy flux with wavelength for conversion
    >>> flx = Flux(energy_flux * u.W/u.m**2/u.nm, wavelength=wvl)
    """
    value: jnp.ndarray

    def __init__(self, input_value: ArrayLike,
                 wavelength: Optional[Union[Wavelength, u.Quantity]] = None):
        if isinstance(input_value, u.Quantity):
            self.value = jnp.asarray(
                self._convert(input_value, wavelength)
            )
        else:
            # Assume already in internal units
            self.value = jnp.asarray(input_value)

    def _convert(self, quantity: u.Quantity,
                 wavelength: Optional[Union[Wavelength, u.Quantity]]) -> np.ndarray:
        """Convert quantity to internal flux units."""
        # Check if already in photon units
        if u.photon in quantity.unit.bases:
            return quantity.to(FLUX).value

        # Energy flux - needs wavelength for conversion
        if wavelength is None:
            raise ValueError(
                f"Cannot convert {quantity.unit} to photon flux. "
                "Energy flux requires wavelength parameter for conversion."
            )

        # Get wavelength in nm
        if isinstance(wavelength, Wavelength):
            wl_nm = wavelength.value
        elif isinstance(wavelength, u.Quantity):
            wl_nm = wavelength.to(u.nm).value
        else:
            wl_nm = np.asarray(wavelength)

        # Energy per photon: E = h*c/λ
        wl_quantity = wl_nm * u.nm
        photon_energy = (h * c / wl_quantity).to(u.J)

        # Convert energy flux to photon flux
        energy_flux = quantity.to(u.W / u.m**2 / u.nm)
        photon_flux = energy_flux / photon_energy
        result = (photon_flux * u.ph).to(FLUX)

        return result.value

    def to(self, unit: u.Unit) -> u.Quantity:
        """Convert back to astropy Quantity in specified unit."""
        return (np.asarray(self.value) * FLUX).to(unit)

    def __repr__(self):
        return f"Flux({self.value.shape} ph/s/m²/nm)"