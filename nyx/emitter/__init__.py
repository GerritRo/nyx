"""Emitters: the things in the sky nyx renders.

Each one prepares itself against an :class:`~nyx.core.observation.Observation`
and hands the render loop a :class:`~nyx.core.protocols.SourceObsData`.
"""

from ._base import BaseEmitter
from .airglow import Airglow
from .moon import Moon
from .point_source import PointSource, VariableSource, blackbody_photon_flux
from .stars import BrightStars, Stars, gaia_star_field
from .zodiacal import ZodiacalLight

__all__ = [
    "Airglow",
    "BaseEmitter",
    "BrightStars",
    "Moon",
    "PointSource",
    "Stars",
    "VariableSource",
    "ZodiacalLight",
    "blackbody_photon_flux",
    "gaia_star_field",
]
