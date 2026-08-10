from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from nyx.core.filters import _navigate, per_obs_filter
from nyx.core.observation import Observation, RenderGeometry
from nyx.core.parameter import _ParametersTable, _wrap_like, parameters_table
from nyx.core.protocols import (
    AtmosphereModel,
    InstrumentModel,
    SkySource,
    SourceObsData,
    set_source_weight,
)


class _ObsBundle(eqx.Module):
    """Per-instrument observation data.

    Holds the precomputed source data, render geometry, and obs count
    for one instrument.
    """

    obs_data: dict[str, SourceObsData]
    render_geometry: RenderGeometry
    nobs: int = eqx.field(static=True)


class _RenderFrame(eqx.Module):
    """Combined frame for the pipeline: shared sky + per-instrument data.

    Sources are ``(SkySource, SourceObsData)`` pairs.  Each pair's
    ``SourceObsData`` carries ``direct`` and ``inscatter`` flags that
    control the rendering path. No external classification needed.
    """

    atmosphere: AtmosphereModel
    sources: list[tuple[SkySource, SourceObsData]]
    instrument: InstrumentModel
    render_geometry: RenderGeometry
    nobs: int = eqx.field(static=True)


class Scene(eqx.Module):
    """Multi-instrument observation of a shared physical sky.

    The scene cleanly separates the *physical sky* (atmosphere + source
    models with trainable parameters) from *observation data*
    (per-instrument precomputed catalog positions, diffuse maps,
    instrument models, and geometry).

    Atmosphere and source models are shared across all instruments.
    Gradients from all instruments flow to the same shared parameters.
    Each instrument may have a different ``nobs``.

    Instruments and sources are accessed by name::

        scene.CT1.shift                       # instrument parameter
        scene.GaiaDR3.spectral_model.params   # source parameter

    Or directly via the ``instruments`` / ``sources`` dicts::

        scene.instruments['CT1'].shift
        scene.sources['GaiaDR3'].spectral_model.params

    Use :meth:`build` to construct from components, or construct
    directly.

    Parameters
    ----------
    atmosphere : AtmosphereModel
        Shared atmosphere model.
    sources : dict
        ``{name: SkySource}`` shared source models (extracted via
        ``src.model()``).
    instruments : dict
        ``{name: InstrumentModel}`` prepared instrument models.
    _obs_bundles : dict
        ``{name: _ObsBundle}`` per-instrument observation data.
        Keys must match ``instruments``.

    Examples
    --------
    Single instrument::

        scene = Scene.build(instrument, atmosphere, sources, obs)
        rates = scene.render()  # {'instrument': array}

    Multiple instruments with names::

        scene = Scene.build(
            {'CT1': inst_a, 'CT5': inst_b},
            atmosphere,
            {'GaiaDR3': gaia, 'airglow': airglow},
            {'CT1': obs_a, 'CT5': obs_b},
        )
        rates = scene.render()  # {'CT1': array, 'CT5': array}

    Named access to parameters::

        scene.atmosphere.Mie.aod_500   # shared across instruments
        scene.airglow.spectral_model   # source by name
        scene.CT1.shift                # instrument by name
    """

    atmosphere: AtmosphereModel
    sources: dict[str, SkySource]
    instruments: dict[str, InstrumentModel]
    _obs_bundles: dict[str, _ObsBundle]

    def __getattr__(self, name: str) -> Any:
        for d in (self.sources, self.instruments):
            if name in d:
                return d[name]
        raise AttributeError(
            f"'{type(self).__name__}' has no attribute, source, or instrument named '{name}'"
        )

    @property
    def nobs(self) -> dict[str, int]:
        """Dict of observation counts, keyed by instrument name."""
        return {name: b.nobs for name, b in self._obs_bundles.items()}

    @classmethod
    def build(
        cls,
        instruments: InstrumentModel | dict[str, InstrumentModel],
        atmosphere: AtmosphereModel,
        sources: dict[str, Any] | list[Any],
        obs_list: Observation | dict[str, Observation] | list[Observation],
    ) -> Scene:
        """Build a Scene from components and observations.

        All sources are treated uniformly as builders: ``model()``
        returns the shared ``SkySource`` for the scene pytree, and
        ``prepare(obs)`` returns per-instrument observation data (or
        ``None`` for self-contained sources).

        Parameters
        ----------
        instruments : InstrumentModel or dict of {name: InstrumentModel}
            A single instrument is automatically wrapped as
            ``{'instrument': inst}``.
        atmosphere : AtmosphereModel
            Shared physical atmosphere model.
        sources : dict of {name: EmitterBuilder} or list
            Sky sources.  Each must provide ``model()`` and
            ``prepare(obs)``.  A list is auto-named from class names.
        obs_list : Observation or dict of {name: Observation}
            A single Observation is broadcast to all instruments.
            When a dict is given its keys must match ``instruments``.

        Returns
        -------
        Scene
        """
        # Normalize instruments to dict
        if isinstance(instruments, InstrumentModel):
            instruments = {"instrument": instruments}

        # Normalize sources to dict
        if isinstance(sources, (list, tuple)):
            sources = {type(s).__name__: s for s in sources}

        # Normalize obs to dict
        if isinstance(obs_list, Observation):
            obs_list = {name: obs_list for name in instruments}
        elif isinstance(obs_list, (list, tuple)):
            if len(obs_list) != len(instruments):
                raise ValueError(
                    f"len(obs_list)={len(obs_list)} does not match "
                    f"len(instruments)={len(instruments)}"
                )
            obs_list = dict(zip(instruments.keys(), obs_list, strict=True))

        if set(instruments) != set(obs_list):
            raise ValueError(
                f"instrument keys {set(instruments)} do not match obs keys {set(obs_list)}"
            )

        # Extract shared source models
        source_names = list(sources.keys())
        scene_sources = {name: src.model() for name, src in sources.items()}

        # Build per-instrument data
        prepared_instruments = {}
        obs_bundles = {}
        for inst_name, inst in instruments.items():
            obs = obs_list[inst_name]
            geoms = obs.get_render_geometry()
            render_geometry = jax.tree.map(lambda *xs: jnp.stack(xs), *geoms)

            obs_data = {src_name: sources[src_name].prepare(obs) for src_name in source_names}

            prepared_instruments[inst_name] = inst.prepare(obs)
            obs_bundles[inst_name] = _ObsBundle(
                obs_data=obs_data,
                render_geometry=render_geometry,
                nobs=obs.nobs,
            )

        return cls(
            atmosphere=atmosphere,
            sources=scene_sources,
            instruments=prepared_instruments,
            _obs_bundles=obs_bundles,
        )

    def render(self) -> dict[str, jax.Array]:
        """Render all observations to pixel rates.

        Shared atmosphere and source models are combined with
        per-instrument data at render time.

        Returns
        -------
        dict of {name: jax.Array}, each shape (nobs_i, n_pixels_i)
            One array per instrument, keyed by instrument name.
        """
        from nyx.core.pipeline import render as _render_single

        results = {}
        for inst_name, bundle in self._obs_bundles.items():
            od = bundle.obs_data
            frame = _RenderFrame(
                atmosphere=self.atmosphere,
                sources=[(self.sources[n], od[n]) for n in od],
                instrument=self.instruments[inst_name],
                render_geometry=bundle.render_geometry,
                nobs=bundle.nobs,
            )
            filt = per_obs_filter(frame)
            per_obs, shared = eqx.partition(frame, filt)

            def _single(per_obs_i: Any, shared: Any = shared) -> jax.Array:
                f = eqx.combine(shared, per_obs_i)
                return _render_single(f)

            results[inst_name] = jax.vmap(_single)(per_obs)
        return results

    def set(self, path: str, value: Any) -> Scene:
        """Set a field by dotted path.

        Instruments and sources are accessed by name::

            scene.set('atmosphere.Mie.aod_500', 0.3)
            scene.set('airglow.spectral_model.params', 100.0)
            scene.set('CT1.shift', new_shift)
            scene.set('CT1.efficiency', 0.8)

        When the target is a :class:`Parameter`, a raw array-like *value*
        is auto-wrapped, preserving the target's ``scale``, ``per_obs``
        and ``frozen`` metadata.  Pass a :class:`Parameter` explicitly to
        override that metadata.
        """
        return self.set_params({path: value})

    def set_params(self, params: dict[str, Any]) -> Scene:
        """Set multiple fields from a ``{dotted_path: value}`` dict.

        Raw values are auto-wrapped into :class:`Parameter` instances when
        the target is a Parameter (see :meth:`set`).
        """
        if not params:
            return self
        resolved = [tuple(self._resolve_path(k.split("."))) for k in params]
        wrapped = tuple(
            _wrap_like(_navigate(self, r), v)
            for r, v in zip(resolved, params.values(), strict=True)
        )
        return eqx.tree_at(
            lambda s: tuple(_navigate(s, r) for r in resolved),
            self,
            wrapped,
        )

    def set_lightcurve(
        self,
        index: int,
        curve: Any,
        *,
        source: str | None = None,
        instrument: str | None = None,
    ) -> Scene:
        """Set one point source's light curve on a built scene; return a new Scene.

        Unlike rebuilding, this reuses the precomputed per-observation geometry
        (coordinate transforms, diffuse maps), so it is cheap enough to sweep
        many curves over a single scene::

            scene = Scene.build(instrument, atmosphere, {"GaiaDR3": stars}, obs)
            for c in curves:
                rates = scene.set_lightcurve(i, c).render()

        Parameters
        ----------
        index : int
            Source column to modulate, following
            :meth:`~nyx.emitter.stars.Stars.resolved_in_fov` ordering (resolved
            stars first).
        curve : array-like
            ``(nobs,)`` achromatic or ``(nobs, n_wvl)`` wavelength-dependent
            multiplicative factor (``1.0`` unchanged, ``0.0`` fully blocked).
        source : str, optional
            Source name; defaults to the sole point source when unambiguous.
        instrument : str, optional
            Instrument name; defaults to the sole instrument when unambiguous.

        Returns
        -------
        Scene
            A new scene with the light curve applied; the original is unchanged.
        """
        inst = self._resolve_lightcurve_instrument(instrument)
        src = self._resolve_lightcurve_source(source, inst)
        bundle = self._obs_bundles[inst]
        od = bundle.obs_data[src]
        if od.source_coords is None:
            raise ValueError(f"source {src!r} on instrument {inst!r} has no point sources")
        nobs, n_src = int(od.source_coords.shape[0]), int(od.source_coords.shape[1])
        n_wvl = int(self.instruments[inst].bandpass.shape[0])
        weights = set_source_weight(
            od.source_weights, index, curve, nobs=nobs, n_src=n_src, n_wvl=n_wvl
        )
        new_od = SourceObsData(
            diffuse_conditions=od.diffuse_conditions,
            diffuse_norm=od.diffuse_norm,
            source_conditions=od.source_conditions,
            source_coords=od.source_coords,
            source_weights=weights,
            direct=od.direct,
            inscatter=od.inscatter,
            _per_obs=tuple(dict.fromkeys(od._per_obs + ("source_weights",))),
        )
        new_bundle = _ObsBundle(
            obs_data={**bundle.obs_data, src: new_od},
            render_geometry=bundle.render_geometry,
            nobs=bundle.nobs,
        )
        return eqx.tree_at(lambda s: s._obs_bundles[inst], self, new_bundle)

    def _resolve_lightcurve_instrument(self, instrument: str | None) -> str:
        if instrument is not None:
            if instrument not in self._obs_bundles:
                raise KeyError(
                    f"{instrument!r} is not an instrument; choices: {list(self._obs_bundles)}"
                )
            return instrument
        if len(self._obs_bundles) == 1:
            return next(iter(self._obs_bundles))
        raise ValueError(
            f"scene has multiple instruments {list(self._obs_bundles)}; pass instrument=..."
        )

    def _resolve_lightcurve_source(self, source: str | None, instrument: str) -> str:
        obs_data = self._obs_bundles[instrument].obs_data
        if source is not None:
            if source not in obs_data:
                raise KeyError(
                    f"{source!r} is not a source on {instrument!r}; choices: {list(obs_data)}"
                )
            return source
        point = [name for name, od in obs_data.items() if od.source_coords is not None]
        if len(point) == 1:
            return point[0]
        raise ValueError(f"specify source=...; point sources on {instrument!r}: {point}")

    def parameters_table(self) -> _ParametersTable:
        """Return a pretty-printed table of every :class:`Parameter` in
        the scene (including frozen ones)."""
        return parameters_table(self)

    def save(self, path: str | os.PathLike[str], observations: dict[str, Observation]) -> None:
        """Write a full fit bundle (params + observation metadata) to *path*.

        The bundle is a single HDF5 file readable via
        :func:`nyx.core.load_fit` for analysis without reconstructing a
        Scene.

        Parameters
        ----------
        path : str or path-like
            Destination filename.
        observations : dict of {name: Observation}
            The same ``obs_list`` passed to :meth:`Scene.build`.
            Required because Scene only retains the precomputed JAX
            pytrees, not the original astropy objects.

        Examples
        --------
        ::

            scene = Scene.build(instruments, atmosphere, sources, obs_dict)
            fitted, _ = opt.run(scene)
            fitted.save('fit.h5', obs_dict)

            # Later, for analysis:
            from nyx.core import load_fit
            result = load_fit('fit.h5')
            result.params['atmosphere.Mie.aod_500']
            result.observations['CT1'].times
        """
        from nyx.core.io import save_fit

        save_fit(path, self, observations)

    def _resolve_path(self, parts: Sequence[str]) -> list[str | int]:
        """Resolve user-facing path segments to internal pytree path.

        Maps instrument/source names to their internal locations::

            ('CT1', 'shift')  -> ('instruments', 'CT1', 'shift')
            ('airglow', ...)  -> ('sources', 'airglow', ...)
            ('atmosphere', ..) -> ('atmosphere', ..)
        """
        first, *rest = parts
        rest_parts: list[str | int] = [int(p) if p.isdigit() else p for p in rest]
        if first in ("atmosphere", "sources", "instruments", "_obs_bundles"):
            return [first] + rest_parts
        for dict_name in ("sources", "instruments"):
            if first in getattr(self, dict_name):
                return [dict_name, first] + rest_parts
        raise KeyError(
            f"'{first}' is not a Scene field, source name, or "
            f"instrument name. Sources: {list(self.sources.keys())}, "
            f"Instruments: {list(self.instruments.keys())}"
        )

    def __len__(self) -> int:
        """Number of instruments."""
        return len(self.instruments)

    def __getitem__(self, name: str) -> InstrumentModel:
        """Return the instrument model for *name*."""
        return self.instruments[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.instruments)