from __future__ import annotations

import jax

from nyx.core.parameter import Parameter
from nyx.core.protocols import SkySource
from nyx.core.records import PointSourceData, SourceObsData, unwrap
from nyx.spectra import SpectralModel

__all__ = ["SpectralSource"]


class SpectralSource(SkySource):
    """Routes the diffuse and point paths through one ``spectral_model`` call.

    Attributes
    ----------
    spectral_model : SpectralModel
        Maps per-pixel or per-source conditions to spectra.
    brightness : Parameter or None
        Fittable brightness multiplier.
    """

    spectral_model: SpectralModel
    brightness: Parameter | None = None

    def diffuse_radiance(self, obs_data: SourceObsData | None = None) -> jax.Array | None:
        if obs_data is None:
            return None
        conditions = unwrap(obs_data.diffuse_conditions)
        if conditions is None:
            return None
        radiance = obs_data.diffuse_norm * self.spectral_model(conditions)
        if self.brightness is None:
            return radiance
        return self.brightness.value * radiance

    def point_sources(self, obs_data: SourceObsData | None = None) -> PointSourceData | None:
        if obs_data is None:
            return None
        coords = unwrap(obs_data.source_coords)
        if coords is None:
            return None
        spectra = self.spectral_model(unwrap(obs_data.source_conditions))
        if self.brightness is not None:
            spectra = self.brightness.value * spectra
        return PointSourceData(spectra=spectra, coords=coords)
