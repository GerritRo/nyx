"""The runtime layer: the pytrees JAX traces, holding the trainable parameters.

An emitter is two objects.  This module is the second one.  A
:class:`~nyx.emitter.base.Emitter` builds in numpy and astropy and is never
seen by JAX; the :class:`~nyx.core.protocols.SkySource` it hands to the scene
is a pytree, is traced and differentiated, and owns every
:class:`~nyx.core.parameter.Parameter` the optimizer can reach.

One class covers every emitter in nyx.  Return something else from
``Emitter.model()`` only when a source needs physics this does not express.
"""

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
        wants.  Not to be confused with ``SourceObsData.diffuse_norm``, which
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
            # Post-vmap the brightness is already sliced to this frame: a
            # scalar, or ``(n_wvl,)`` for a chromatic curve.  ``spectra`` is
            # ``(n_src, n_wvl)``, so both broadcast without a shape branch.
            spectra = self.brightness.value * spectra
        return PointSourceData(spectra=spectra, coords=obs_data.source_coords)
