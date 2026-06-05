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

    @abstractmethod
    def prepare(self, obs):
        """Precompute per-observation data for the render loop."""

    def model(self):
        """Return the shared source model."""
        return SourceModel(spectral_model=self._spectral_model)
