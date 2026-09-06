from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from nyx.core.filters import tile_per_obs
from nyx.core.spectral import SpectralModel

if TYPE_CHECKING:
    from nyx.core.observation import Observation, SkyGeometry
    from nyx.core.parameter import Parameter

# Source protocols


class PointSourceData(eqx.Module):
    """Point source data for the render loop (single observation)."""

    spectra: jax.Array  # (n_sources, n_wvl)
    coords: jax.Array  # (n_sources, 2) AltAz (az, alt) in radians


class SkySource(eqx.Module):
    """Source model: shared physics that maps conditions to photons.

    A SkySource is stored at the :class:`Scene` level.  It holds
    trainable parameters (spectral models, scaling factors) and
    implements the physics that maps observation data into radiance
    and point-source spectra.

    The standard implementation is :class:`SourceModel`, which routes
    all computation through a single ``spectral_model(conditions)``
    call for both diffuse and point-source paths.
    """

    def diffuse_radiance(
        self, geometry: SkyGeometry, obs_data: SourceObsData | None = None
    ) -> jax.Array | None:
        """Diffuse radiance ``(..., n_wvl)`` at the geometry positions.

        ``None`` if this source has no diffuse component.
        """
        return None

    def point_sources(self, obs_data: SourceObsData | None = None) -> PointSourceData | None:
        """Point sources, or ``None`` if this source has no point component."""
        return None


# Observation data


class SourceObsData(eqx.Module):
    """Per-instrument observation data for any sky source.

    Every emitter builder returns one of these from ``prepare(obs)``; the
    conditions go through the source model's ``spectral_model`` at render
    time.  Fields are all optional -- a diffuse-only emitter sets
    ``diffuse_conditions``, a point-only one ``source_conditions`` and
    ``source_coords``, a hybrid both.

    Parameters
    ----------
    diffuse_conditions : jax.Array or None
        Per-pixel conditions for the diffuse path.
    diffuse_norm : jax.Array
        Factor applied after spectral evaluation, e.g. ``1 / pixel_area``
        to turn flux into radiance.
    source_conditions : jax.Array or None
        Per-source conditions for the point path.
    source_coords : jax.Array or None
        Point-source AltAz positions ``(nobs, n_src, 2)``.
    source_weights : jax.Array or None
        Per-observation flux factor on point-source spectra -- ``1.0``
        leaves a source alone, ``0.0`` blocks it -- as ``(nobs, n_src)``
        achromatic or ``(nobs, n_src, n_wvl)`` chromatic.  This is how
        light curves (occultations, variable stars) enter.  When set, list
        ``"source_weights"`` in ``per_obs`` so the render vmap slices it.
    direct : bool
        Whether diffuse radiance takes line-of-sight extinction.  True for
        airglow and zodiacal light; False for catalog sources, whose direct
        light comes from their quasi-point sources instead.  Map scattering
        applies either way.
    inscatter : bool
        Whether point sources are in-scattered individually via
        ``scatter_sources``.  True for the moon, which has no map; False
        for star catalogs, where the diffuse map already covers it.
    """

    diffuse_conditions: jax.Array | None = None
    diffuse_norm: jax.Array = eqx.field(default_factory=lambda: jnp.array(1.0))
    source_conditions: jax.Array | None = None
    source_coords: jax.Array | None = None
    source_weights: jax.Array | None = None
    direct: bool = eqx.field(static=True, default=True)
    inscatter: bool = eqx.field(static=True, default=False)
    per_obs: tuple[str, ...] = eqx.field(static=True, default=())

    def __check_init__(self) -> None:
        valid = {f.name for f in dataclasses.fields(self) if f.name != "per_obs"}
        bad = set(self.per_obs) - valid
        if bad:
            raise ValueError(
                f"SourceObsData.per_obs references unknown fields: {bad!r}. Valid fields: {valid!r}"
            )


def set_source_weight(
    weights: jax.Array | None,
    index: int,
    curve: object,
    *,
    nobs: int,
    n_src: int,
    n_wvl: int,
) -> jax.Array:
    """Set source ``index`` of a :attr:`SourceObsData.source_weights` array.

    ``curve`` is ``(nobs,)`` achromatic or ``(nobs, n_wvl)`` chromatic, and
    ``weights`` is ``None`` until some source is modulated.  The result is
    ``(nobs, n_src)`` while every curve is achromatic, and ``(nobs, n_src,
    n_wvl)`` from the first chromatic one on -- existing achromatic entries
    are broadcast across wavelength at that point.

    Shared by :meth:`nyx.emitter.stars.Stars.prepare` (building weights from
    registered light curves) and :meth:`nyx.core.scene.Scene.set_lightcurve`
    (editing them on a built scene).
    """
    curve = np.asarray(curve, dtype=float)
    if not 0 <= index < n_src:
        raise IndexError(f"source index {index} out of range for {n_src} sources")
    chromatic = curve.ndim == 2 or (weights is not None and weights.ndim == 3)
    if weights is None:
        base = np.ones((nobs, n_src, n_wvl) if chromatic else (nobs, n_src))
    else:
        base = np.array(weights, dtype=float)  # writable copy (jax arrays are read-only)
        if chromatic and base.ndim == 2:  # promote existing achromatic entries
            base = np.broadcast_to(base[..., None], (nobs, n_src, n_wvl)).copy()
    if curve.ndim == 1:
        if curve.shape != (nobs,):
            raise ValueError(f"achromatic curve length {curve.shape} != nobs {nobs}")
        base[:, index] = curve[:, None] if base.ndim == 3 else curve
    else:
        if curve.shape != (nobs, n_wvl):
            raise ValueError(
                f"chromatic curve shape {curve.shape} != (nobs, n_wvl)={(nobs, n_wvl)}"
            )
        base[:, index, :] = curve
    return jnp.asarray(base)


# Emitter builder protocol


@runtime_checkable
class EmitterBuilder(Protocol):
    """Protocol for emitter builder classes.

    Emitter builders handle heavyweight I/O (data loading, catalog
    downloads, coordinate transforms) at construction time and expose
    two methods consumed by :meth:`Scene.build`:

    - ``model()`` returns a shared :class:`SourceModel` stored in the
      scene pytree (holds trainable spectral parameters).
    - ``prepare(obs)`` returns a :class:`SourceObsData` with precomputed
      per-observation conditions and rendering flags (``direct``,
      ``inscatter``) for the render loop.
    """

    def model(self) -> SourceModel: ...

    def prepare(self, obs: Observation) -> SourceObsData: ...


# Unified source model


class SourceModel(SkySource):
    """Unified source model for all sky sources.

    Routes both diffuse and point-source computation through a single
    ``spectral_model(conditions)`` call.  Different emitters provide
    different conditions and spectral models, but the interface is
    identical.
    """

    spectral_model: SpectralModel

    def diffuse_radiance(
        self, geometry: SkyGeometry, obs_data: SourceObsData | None = None
    ) -> jax.Array | None:
        if obs_data is None or obs_data.diffuse_conditions is None:
            return None
        return obs_data.diffuse_norm * self.spectral_model(obs_data.diffuse_conditions)

    def point_sources(self, obs_data: SourceObsData | None = None) -> PointSourceData | None:
        if obs_data is not None and obs_data.source_coords is not None:
            spectra = self.spectral_model(obs_data.source_conditions)
            w = obs_data.source_weights
            if w is not None:
                # post-vmap: (n_src,) achromatic -> broadcast over wavelength;
                #            (n_src, n_wvl) chromatic -> elementwise per wavelength.
                spectra = spectra * (w[:, None] if w.ndim == 1 else w)
            return PointSourceData(
                spectra=spectra,
                coords=obs_data.source_coords,
            )
        return None


# Atmosphere protocol


class AtmosphereResult(eqx.Module):
    """Output of atmosphere evaluation.

    Typed intermediate representation consumed by the render loop.
    Internal details (optical depth, airmass) stay inside the atmosphere.

    The ``extinction_hp`` and ``scattering_map`` arrays cover only
    the upper hemisphere (``nsky`` pixels in HEALPix RING ordering).
    ``npix`` stores the full-sphere pixel count so that
    ``apply_extinction`` can return a full HEALPix map.
    """

    extinction_hp: jax.Array  # [nsky, n_wvl]   exp(-tau * sec_z)
    scattering_map: jax.Array  # [grid_lon, grid_lat, nsky, n_wvl]
    npix: int = eqx.field(static=True)  # full-sphere HEALPix pixel count

    @property
    def nsky(self) -> int:
        """Number of hemisphere sky pixels."""
        return self.extinction_hp.shape[0]

    @property
    def n_wvl(self) -> int:
        """Number of wavelength bins."""
        return self.extinction_hp.shape[1]

    def apply_extinction(self, sky_radiance: jax.Array, bandpass: jax.Array) -> jax.Array:
        """Band-integrate radiance ``(nsky, n_wvl)`` through line-of-sight extinction.

        Returns a full-sphere HEALPix array ``(npix,)``, zero below the
        horizon; hemisphere pixels lead it in RING ordering.
        """
        hp_values = jnp.sum(bandpass * sky_radiance * self.extinction_hp, axis=-1)
        return jnp.zeros(self.npix).at[: hp_values.shape[0]].set(hp_values)

    def apply_scattering(self, sky_radiance_obs: jax.Array, bandpass: jax.Array) -> jax.Array:
        """Scatter radiance ``(nsky, n_wvl)`` into the FOV grid ``(n_lon, n_lat)``.

        One observation at a time.
        """
        return jnp.sum(bandpass * sky_radiance_obs * self.scattering_map, axis=(-2, -1))


class AtmosphereModel(eqx.Module):
    """Atmosphere model implementing extinction and scattering.

    Shared at the :class:`Scene` level across all instruments.

    Trainable quantities are any :class:`~nyx.core.parameter.Parameter`
    fields the subclass declares; the fitter discovers them automatically.
    """

    def evaluate(self, sky: SkyGeometry) -> AtmosphereResult:
        """Extinction and scattering for the current trainable parameters."""
        raise NotImplementedError

    def extinct(self, altitudes: jax.Array, spectra: jax.Array, height_km: jax.Array) -> jax.Array:
        """Extinct point-source spectra ``(n_sources, n_wvl)``.

        ``altitudes`` is ``(n_sources, 1)`` in radians and ``height_km`` the
        observer's height above sea level.  The default returns the spectra
        unchanged; override to absorb point sources.
        """
        return spectra

    def scatter_sources(
        self,
        sky: SkyGeometry,
        source_coords: jax.Array,
        source_spectra: jax.Array,
        bp: jax.Array,
    ) -> jax.Array | float:
        """Scatter discrete point sources into the FOV. Default: none."""
        return 0.0


# Instrument protocol


class InstrumentModel(eqx.Module):
    """Sensor that maps sky radiance to detector counts.

    Trainable quantities are :class:`~nyx.core.parameter.Parameter` fields
    the subclass declares; per-observation axes (e.g. ``shift``,
    ``rotation``) are marked by setting ``per_obs=True`` on those
    Parameters, and :func:`nyx.core.filters.tile_per_obs` will broadcast
    them to ``(nobs, ...)`` at prepare time.

    Every concrete instrument provides an ``efficiency`` Parameter (overall
    throughput scale); the render pipeline multiplies pixel rates by it.
    """

    if TYPE_CHECKING:
        # Declared by concrete instruments as an eqx field; part of the
        # contract relied on by nyx.core.pipeline.render.
        efficiency: Parameter

    def prepare(self, obs: Observation) -> InstrumentModel:
        """Batch for multi-observation rendering.

        Default: tiles ``per_obs`` fields to ``(nobs, ...)``.
        """
        return tile_per_obs(self, obs.nobs)

    @property
    def bandpass(self) -> jax.Array:
        """Spectral transmission curve (no efficiency). Shape [n_wvl]."""
        raise NotImplementedError

    def corrected_pm(self, pm: jax.Array) -> jax.Array:
        """Pointing matrix corrected for detector shift and rotation."""
        return pm

    def project_scattered(self, eval_grid_values: jax.Array) -> jax.Array:
        """Scattering eval grid -> pixel rates. Default: no contribution."""
        return jnp.array(0.0)

    def project_diffuse(self, hp_values: jax.Array, pm: jax.Array) -> jax.Array:
        """HEALPix sky values -> pixel rates. Shape [n_pixels]."""
        raise NotImplementedError

    def project_catalog(self, source_coords: jax.Array, source_fluxes: jax.Array) -> jax.Array:
        """Point source (coords, fluxes) -> pixel rates. Default: no contribution."""
        return jnp.array(0.0)
