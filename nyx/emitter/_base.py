from abc import ABC, abstractmethod
from typing import Any

from nyx.core.protocols import SourceModel


class BaseEmitter(ABC):
    """Shared base for EmitterBuilder implementations.

    Provides the ``model()`` method.  Subclasses must set
    ``self._spectral_model`` in their ``__init__`` and implement
    ``prepare(obs)``.
    """

    _spectral_model: Any
    #: Signature of the Geometry this emitter was built against, compared by
    #: :meth:`nyx.core.scene.Scene.build`.  ``None`` on a third-party emitter
    #: that does not record one, which is not an error -- only a missed check.
    _geo_signature: tuple | None = None

    @abstractmethod
    def prepare(self, obs):
        """Precompute per-observation data for the render loop."""

    def model(self):
        """Return the shared source model."""
        return SourceModel(spectral_model=self._spectral_model)

    def __repr__(self) -> str:
        """Name the emitter and whatever sizes it recorded.

        Emitters keep different state, so this reports the fields they
        happen to carry rather than a fixed set -- enough to tell two
        differently-configured emitters apart at a notebook prompt.
        """
        bits = []
        if getattr(self, "_coords", None) is not None:
            bits.append(f"{len(self._coords)} sources")
        if getattr(self, "_sky_map", None) is not None:
            bits.append(f"map nside={self._map_nside}")
        for name, label in (("_height_km", "height"), ("_t_ref", "t_ref")):
            value = getattr(self, name, None)
            if value is not None:
                bits.append(f"{label}={value}")
        model = type(self._spectral_model).__name__ if self._spectral_model is not None else "none"
        bits.append(f"spectrum={model}")
        return f"{type(self).__name__}({', '.join(bits)})"
