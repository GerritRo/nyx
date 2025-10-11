"""
Unit Converter for Nyx
"""

import numpy as np
import jax.numpy as jnp
import astropy.units as u
from typing import Union, Optional, Tuple
from astropy.constants import c, h

from .specification import (
    RADIANCE, FLUX, UNIT_REGISTRY
)

class NyxUnit:
    """
    Wraps values with units and handles conversion to internal units.
    """
    
    def __init__(self, value: Union[float, np.ndarray, u.Quantity], 
                 unit: Optional[u.Unit] = None,
                 wavelength: Optional[u.Quantity] = None):
        """
        Parameters
        ----------
        value : float, array, or Quantity
            Value(s) to wrap
        unit : astropy.units.Unit, optional
            Unit of the value. If value is a Quantity, this is ignored.
        wavelength : Quantity, optional
            Wavelength array needed for energy→photon flux conversion
        """
        if isinstance(value, u.Quantity):
            self.value = value.value
            self.unit = value.unit
        else:
            self.value = np.asarray(value)
            self.unit = unit if unit is not None else u.dimensionless_unscaled
        
        self.wavelength = wavelength
    
    def to_internal(self, unit_type: str) -> 'NyxUnit':
        """
        Convert to internal unit system.
        
        Parameters
        ----------
        unit_type : str
            Type of unit (e.g., 'radiance', 'wavelength')
            
        Returns
        -------
        NyxUnit
            New instance with values in internal units
        """
        unit_info = UNIT_REGISTRY[unit_type]
        internal_unit = unit_info['internal']
        
        # Handle energy flux → photon flux conversion specially
        if unit_type in ['radiance', 'flux']:
            converted = self._convert_to_photon_flux(internal_unit, unit_type)
        else:
            quantity = self.value * self.unit
            converted = quantity.to(internal_unit)
        
        return NyxUnit(converted.value, converted.unit, self.wavelength)
    
    def _convert_to_photon_flux(self, target_unit: u.Unit, flux_type: str) -> u.Quantity:
        """
        Convert energy flux to photon flux or between flux types.
        """
        quantity = self.value * self.unit
        
        # Check if already in photon units
        if u.photon in self.unit.bases:
            return quantity.to(target_unit)
        
        # Need wavelength for energy→photon conversion
        if self.wavelength is None:
            raise ValueError(
                f"Cannot convert {self.unit} to {target_unit}. "
                "Energy to photon flux conversion requires wavelength. "
                "Provide wavelength parameter to NyxUnit or use convert_energy_to_photon_flux()."
            )
        
        # Convert wavelength to consistent units
        wl = self.wavelength.to(u.nm) if isinstance(self.wavelength, u.Quantity) else self.wavelength * u.nm
        
        # Energy per photon: E = h*c/λ
        photon_energy = (h * c / wl).to(u.J)
        
        # Determine input flux type and convert
        if quantity.unit.is_equivalent(u.W / u.m**2 / u.nm / u.sr):
            # Diffuse energy flux
            energy_flux = quantity.to(u.W / u.m**2 / u.nm / u.sr)
            # Broadcast if needed
            if photon_energy.ndim > 0 and energy_flux.ndim > 0:
                photon_flux = (energy_flux / photon_energy[..., np.newaxis] if energy_flux.shape[-1] != photon_energy.shape[0] 
                              else energy_flux / photon_energy)
            else:
                photon_flux = energy_flux / photon_energy
            result = (photon_flux*u.ph).to(RADIANCE)
            
        elif quantity.unit.is_equivalent(u.W / u.m**2 / u.nm):
            # Point source energy flux
            energy_flux = quantity.to(u.W / u.m**2 / u.nm)
            if photon_energy.ndim > 0 and energy_flux.ndim > 0:
                photon_flux = (energy_flux / photon_energy[..., np.newaxis] if energy_flux.shape[-1] != photon_energy.shape[0]
                              else energy_flux / photon_energy)
            else:
                photon_flux = energy_flux / photon_energy 
            result = (photon_flux*u.ph).to(FLUX)
            
        else:
            raise u.UnitConversionError(
                f"Cannot convert {quantity.unit} to photon flux. "
                f"Expected energy flux units like W/m²/nm or W/m²/nm/sr"
            )
        
        return result
    
    def to_jax(self) -> jnp.ndarray:
        """Return value as JAX array (units stripped)."""
        return jnp.asarray(self.value)
    
    def to_numpy(self) -> np.ndarray:
        """Return value as NumPy array (units stripped)."""
        return np.asarray(self.value)
    
    def __repr__(self):
        return f"NyxUnit({self.value}, {self.unit})"


def nixify(value: Union[float, np.ndarray, u.Quantity],
                        unit_type: str,
                        current_unit: Optional[u.Unit] = None,
                        wavelength: Optional[u.Quantity] = None,
                        return_jax: bool = True) -> Union[jnp.ndarray, Tuple[jnp.ndarray, u.Unit]]:
    """
    Convert value to internal units and return as JAX array.
    
    Parameters
    ----------
    value : float, array, or Quantity
        Input value
    unit_type : str
        Type of unit (e.g., 'radiance', 'wavelength')
    current_unit : astropy.units.Unit, optional
        Current unit if value is not a Quantity
    wavelength : Quantity, optional
        Wavelength for energy→photon flux conversion
    return_jax : bool, default True
        If True, return JAX array. If False, return (numpy array, unit) tuple
        
    Returns
    -------
    jnp.ndarray or (np.ndarray, u.Unit)
        Converted value as JAX array, or (value, unit) tuple if return_jax=False
        
    Examples
    --------
    >>> # Direct to JAX (default)
    >>> jax_val = convert_to_internal(500*u.nm, 'wavelength')
    >>> 
    >>> # With energy flux (needs wavelength)
    >>> wl = np.linspace(300, 700, 50) * u.nm
    >>> flux = 1e-8 * u.W/u.m**2/u.micron/u.sr
    >>> jax_flux = convert_to_internal(flux, 'radiance', wavelength=wl)
    """
    converter = NyxUnit(value, current_unit, wavelength)
    converted = converter.to_internal(unit_type)
    
    if return_jax:
        return converted.to_jax()
    else:
        return converted.to_numpy(), converted.unit

def strip_units_for_jax(value: Union[float, np.ndarray, u.Quantity]) -> jnp.ndarray:
    """
    Strip units and convert to JAX array.
    
    Parameters
    ----------
    value : float, array, or Quantity
        Input value
        
    Returns
    -------
    jnp.ndarray
        JAX array without units
    """
    if isinstance(value, u.Quantity):
        return jnp.asarray(value.value)
    else:
        return jnp.asarray(value)


def attach_units(value: Union[np.ndarray, jnp.ndarray],
                 unit_type: str) -> u.Quantity:
    """
    Attach internal units to unitless array.
    
    Parameters
    ----------
    value : array
        Unitless value
    unit_type : str
        Type of unit to attach
        
    Returns
    -------
    Quantity
        Value with appropriate units
    """
    unit_info = UNIT_REGISTRY[unit_type]
    internal_unit = unit_info['internal']
    
    if isinstance(value, jnp.ndarray):
        value = np.asarray(value)
    
    return value * internal_unit