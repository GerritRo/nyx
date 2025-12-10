import astropy.units as u

# =============================================================================
# FLUX UNITS
# =============================================================================

# Diffuse flux (surface brightness)
# Used for: Extended sources like airglow, zodiacal light
RADIANCE = u.photon / (u.s * u.m**2 * u.nm * u.sr)
RADIANCE_NAME = "photon/s/m²/nm/sr"
RADIANCE_DESCRIPTION = "Photon flux per unit area, wavelength, and solid angle"

# Point source flux
# Used for: Stars, catalog sources
FLUX = u.photon / (u.s * u.m**2 * u.nm)
FLUX_NAME = "photon/s/m²/nm"
FLUX_DESCRIPTION = "Photon flux per unit area and wavelength"

# Energy flux
# Used for stars, catalog sources

# =============================================================================
# SPECTRAL UNITS
# =============================================================================

# Wavelength
# Internal representation: nanometers
WAVELENGTH = u.nm
WAVELENGTH_NAME = "nm"
WAVELENGTH_DESCRIPTION = "Wavelength in nanometers"

# =============================================================================
# ANGULAR UNITS
# =============================================================================

# Angle
# Internal representation: radians
ANGLE = u.rad
ANGLE_NAME = "rad"
ANGLE_DESCRIPTION = "Angle in radians"

# Solid angle
SOLID_ANGLE = u.sr
SOLID_ANGLE_NAME = "sr"
SOLID_ANGLE_DESCRIPTION = "Solid angle in steradians"

# =============================================================================
# ATMOSPHERIC UNITS
# =============================================================================

# Optical depth (dimensionless)
OPTICAL_DEPTH = u.dimensionless_unscaled
OPTICAL_DEPTH_NAME = "dimensionless"
OPTICAL_DEPTH_DESCRIPTION = "Atmospheric optical depth (dimensionless)"

# Extinction coefficient (dimensionless)
EXTINCTION = u.dimensionless_unscaled
EXTINCTION_NAME = "dimensionless"
EXTINCTION_DESCRIPTION = "Atmospheric extinction coefficient (dimensionless)"

# Airmass (dimensionless)
AIRMASS = u.dimensionless_unscaled
AIRMASS_NAME = "dimensionless"
AIRMASS_DESCRIPTION = "Atmospheric airmass (dimensionless)"

# =============================================================================
# INSTRUMENT UNITS
# =============================================================================

# Bandpass/transmission (dimensionless)
TRANSMISSION = u.dimensionless_unscaled
TRANSMISSION_NAME = "dimensionless"
TRANSMISSION_DESCRIPTION = "Instrument transmission (dimensionless, 0-1)"

# Effective area
EFFECTIVE_AREA = u.m**2
EFFECTIVE_AREA_NAME = "m²"
EFFECTIVE_AREA_DESCRIPTION = "Effective collection area in square meters"

# =============================================================================
# OUTPUT UNITS
# =============================================================================

# Detection rate
RATE = u.photon / u.s
RATE_NAME = "photon/s"
RATE_DESCRIPTION = "Photon detection rate per second"

# =============================================================================
# PHYSICAL CONSTANTS (for convenience)
# =============================================================================

from astropy.constants import c as SPEED_OF_LIGHT
from astropy.constants import h as PLANCK_CONSTANT

# =============================================================================
# UNIT DOCUMENTATION
# =============================================================================

UNIT_REGISTRY = {
    'radiance': {
        'internal': RADIANCE,
        'name': RADIANCE_NAME,
        'description': RADIANCE_DESCRIPTION,
    },
    'flux': {
        'internal': FLUX,
        'name': FLUX_NAME,
        'description': FLUX_DESCRIPTION,
    },
    'wavelength': {
        'internal': WAVELENGTH,
        'name': WAVELENGTH_NAME,
        'description': WAVELENGTH_DESCRIPTION,
    },
    'angle': {
        'internal': ANGLE,
        'name': ANGLE_NAME,
        'description': ANGLE_DESCRIPTION,
    },
    'solid_angle': {
        'internal': SOLID_ANGLE,
        'name': SOLID_ANGLE_NAME,
        'description': SOLID_ANGLE_DESCRIPTION,
    },
    'airmass': {
        'internal': AIRMASS,
        'name': AIRMASS_NAME,
        'description': AIRMASS_DESCRIPTION,
    },
    'rate': {
        'internal': RATE,
        'name': RATE_NAME,
        'description': RATE_DESCRIPTION,
    },
}


def get_unit_info(unit_type: str) -> dict:
    """
    Get information about a unit type.
    
    Parameters
    ----------
    unit_type : str
        Type of unit (e.g., 'diffuse_flux', 'wavelength')
        
    Returns
    -------
    dict
        Dictionary containing unit information
    """
    if unit_type not in UNIT_REGISTRY:
        raise ValueError(f"Unknown unit type: {unit_type}. "
                        f"Available types: {list(UNIT_REGISTRY.keys())}")
    return UNIT_REGISTRY[unit_type]


def print_unit_specification():
    """Print a formatted summary of the unit specification."""
    print("=" * 70)
    print("Nyx Internal Unit Specification")
    print("=" * 70)
    print()
    
    for unit_type, info in UNIT_REGISTRY.items():
        print(f"{unit_type.upper()}")
        print(f"  Internal: {info['name']}")
        print(f"  Description: {info['description']}")
        print()