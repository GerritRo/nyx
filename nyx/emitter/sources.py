from __future__ import annotations

import jax

from nyx.core.parameter import Parameter
from nyx.core.protocols import SkySource
from nyx.core.records import PointSourceData, SourceObsData
from nyx.utils.spectra import SpectralModel

__all__ = ["SpectralSource"]


class SpectralSource(SkySource):
    """Routes the diffuse and point paths through one ``spectral_model`` call.

    Attributes
    ----------
    spectral_model : SpectralModel
        Maps per-pixel or per-source conditions to spectra.
    brightness : Parameter or None
        A fittable multiplier on everything this source emits, on both paths.
        ``None`` -- the default -- leaves the source with no amplitude of its
        own, which is what an emitter whose spectral model already carries one
        wants. Not to be confused with ``SourceObsData.diffuse_norm``, which
        is a fixed geometric factor such as ``1 / pixel_area`` that ``prepare``
        sets and no optimizer touches.
    """

    spectral_model: SpectralModel
    brightness: Parameter | None = None

    def diffuse_radiance(self, obs_data: SourceObsData | None = None) -> jax.Array | None:
        if obs_data is None or obs_data.diffuse_conditions is None:
            return None
        radiance = obs_data.diffuse_norm * self.spectral_model(obs_data.diffuse_conditions)
        if self.brightness is None:
            return radiance
        return self.brightness.value * radiance

    def point_sources(self, obs_data: SourceObsData | None = None) -> PointSourceData | None:
        if obs_data is None or obs_data.source_coords is None:
            return None
        spectra = self.spectral_model(obs_data.source_conditions)
        if self.brightness is not None:
            spectra = self.brightness.value * spectra
        return PointSourceData(spectra=spectra, coords=obs_data.source_coords)
