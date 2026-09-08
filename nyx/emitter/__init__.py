"""The physical light sources of the night sky.

Every emitter is two objects, and knowing which is which explains most of
this package:

- a **builder** -- :class:`~nyx.emitter.base.Emitter` and its subclasses --
  which holds astropy and numpy state and turns an ``Observation`` into
  arrays, in ``prepare(obs)``.  This is what you construct.
- a **runtime source** -- :class:`~nyx.emitter.sources.SpectralSource` --
  which the builder hands to the scene in ``model()``.  It is a JAX pytree,
  it is traced and differentiated, and it owns every trainable
  ``Parameter``.  You rarely name it.

The builders live in :mod:`nyx.emitter.base` and the modules beside it; the
runtime layer is :mod:`nyx.emitter.sources`; where the star catalogs come
from is :mod:`nyx.emitter.catalogs`.

============================ ========= ======= =================================
Emitter                      Path      Scales? Model
============================ ========= ======= =================================
:class:`Airglow`             diffuse   no      ESO SkyCalc, with an SFU curve
:class:`ZodiacalLight`       diffuse   yes     Leinert et al. (1998)
:class:`Moon`                point     yes     ROLO, after Jones et al. (2013)
:class:`Stars`               both      yes     Gaia DR3 + Pickles (1998)
:class:`BrightStars`         point     yes     XHIP or Gaia, points only
:class:`PointSource`         point     yes     one source, one spectrum
============================ ========= ======= =================================

"Scales" is whether the emitter takes a ``brightness``: a fittable amplitude
on everything it emits.  Four of these have no other free parameter, so it is
the only way to fit their level.  :class:`Airglow` is the exception, and its
docstring says why.

:class:`Stars` and :class:`BrightStars` are both star catalogs, and differ in
what they render: ``Stars`` carries a diffuse all-sky map *and* resolves the
stars in the field of view, while ``BrightStars`` is points only.  Either can
hand one star over to a :class:`PointSource` with ``pop()``, to fit it on its
own.
"""

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
