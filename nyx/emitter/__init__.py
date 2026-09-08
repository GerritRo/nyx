from .airglow import Airglow
from .base import Emitter
from .catalog import CatalogEmitter
from .moon import Moon
from .point_source import PointSource
from .sources import SpectralSource
from .stars import BrightStars, Stars, gaia_star_field
from .zodiacal import ZodiacalLight

__all__ = [
    # The emitters, which is what you build.
    "Airglow",
    "BrightStars",
    "Moon",
    "PointSource",
    "Stars",
    "ZodiacalLight",
    "gaia_star_field",
    # The two layers, which is what you subclass.
    "CatalogEmitter",
    "Emitter",
    "SpectralSource",
]
