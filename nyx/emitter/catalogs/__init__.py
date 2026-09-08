"""Where the star catalogs come from, kept apart from the emitters that render them.

Each module here owns one catalog's file format, photometric system and
spectral model, so a new data release touches one file.  The astrometry and
cone-search helpers they share sit alongside them.
"""

from . import gaia, xhip
from .astrometry import angular_separation_deg, propagate_space_motion, rotate_healpix
from .index import HEALPixIndex

__all__ = [
    "HEALPixIndex",
    "angular_separation_deg",
    "gaia",
    "propagate_space_motion",
    "rotate_healpix",
    "xhip",
]
