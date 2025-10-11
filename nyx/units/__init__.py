"""
Nyx Units Package
"""

from .specification import (
    # Flux units
    RADIANCE,
    FLUX,
    
    # Spectral units
    WAVELENGTH,
    
    # Angular units
    ANGLE,
    SOLID_ANGLE,
    
    # Atmospheric units
    OPTICAL_DEPTH,
    EXTINCTION,
    AIRMASS,
    
    # Instrument units
    TRANSMISSION,
    EFFECTIVE_AREA,
    
    # Output units
    RATE,
    
    # Physical constants
    SPEED_OF_LIGHT,
    PLANCK_CONSTANT,
    
    # Registry
    UNIT_REGISTRY,
    get_unit_info,
    print_unit_specification,
)

from .converter import (
    NyxUnit,
    nixify,
    strip_units_for_jax,
    attach_units,
)

__all__ = [
    # Specification
    'DIFFUSE_FLUX',
    'POINT_FLUX',
    'WAVELENGTH',
    'FREQUENCY',
    'ANGLE',
    'SOLID_ANGLE',
    'OPTICAL_DEPTH',
    'EXTINCTION',
    'AIRMASS',
    'TRANSMISSION',
    'EFFECTIVE_AREA',
    'RATE',
    'COUNT_RATE',
    'SPEED_OF_LIGHT',
    'PLANCK_CONSTANT',
    'UNIT_REGISTRY',
    'get_unit_info',
    'print_unit_specification',
    
    # Converter
    'NyxUnit',
    'nixify',
    'strip_units_for_jax',
    'attach_units',
]