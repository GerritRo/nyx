from __future__ import annotations

import dataclasses
import os
from collections.abc import Iterator
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from nyx.core.filters import per_obs_filter
from nyx.core.geometry import check_shared_geometry
from nyx.core.observation import Observation, RenderGeometry
from nyx.core.parameter import (
    _iter_parameters,
    _ParametersTable,
    n_trainable,
    parameters_table,
    set_parameters,
)
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
    # Static, so the names never reach vmap as leaves.
    source_names: tuple[str, ...] = eqx.field(static=True, default=())


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
        # Never route dunder lookups (``__deepcopy__``, ``__getstate__``, ...)
        # through the name dicts: during unflattening they may not exist yet,
        # and the lookup would recurse.
        if name.startswith("_"):
            raise AttributeError(name)
        for d in (self.sources, self.instruments):
            if name in d:
                return d[name]
        raise AttributeError(
            f"'{type(self).__name__}' has no attribute, source, or instrument named '{name}'"
        )

    def __repr__(self) -> str:
        """What the scene is, not what it contains.

        See ``parameters_table()`` for the parameters, ``profile_scene()``
        for the arrays.
        """
        counts = ", ".join(f"{name}({n} obs)" for name, n in self.nobs.items())
        n_free = n_trainable(self)
        total = sum(int(np.size(np.asarray(p.factor))) for _, p in _iter_parameters(self))
        return (
            f"Scene(instruments: {counts or '(none)'}; "
            f"sources: {', '.join(self.sources) or '(none)'}; "
            f"atmosphere: {type(self.atmosphere).__name__}; "
            f"{n_free} of {total} parameter values free)"
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

        check_shared_geometry(obs_list, atmosphere, instruments, sources)

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

    def _render_frame(self, inst_name: str) -> _RenderFrame:
        """Combine the shared sky with one instrument's observation data."""
        bundle = self._obs_bundles[inst_name]
        od = bundle.obs_data
        return _RenderFrame(
            atmosphere=self.atmosphere,
            sources=[(self.sources[n], od[n]) for n in od],
            instrument=self.instruments[inst_name],
            render_geometry=bundle.render_geometry,
            nobs=bundle.nobs,
            source_names=tuple(od),
        )

    def render(self, *, per_source: bool = False) -> dict[str, Any]:
        """Render all observations to pixel rates.

        Shared atmosphere and source models are combined with
        per-instrument data at render time.

        Parameters
        ----------
        per_source : bool
            Split each instrument's rates by emitter, to find out what is
            lighting a pixel::

                parts = scene.render(per_source=True)['CT1']

            The parts sum to the whole to float32 rounding.  The
            atmosphere kernel is built once either way.

        Returns
        -------
        dict of {name: jax.Array}, each shape (nobs_i, n_pixels_i)
            One array per instrument, keyed by instrument name; or, with
            *per_source*, ``{instrument: {source: array}}``.
        """
        from nyx.core.pipeline import contributions as _contributions
        from nyx.core.pipeline import render as _render_single

        single = _contributions if per_source else _render_single
        results: dict[str, Any] = {}
        for inst_name in self._obs_bundles:
            frame = self._render_frame(inst_name)
            filt = per_obs_filter(frame)
            per_obs, shared = eqx.partition(frame, filt)

            def _single(per_obs_i: Any, shared: Any = shared, kernel: Any = single) -> Any:
                f = eqx.combine(shared, per_obs_i)
                return kernel(f)

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
        """Set parameters from a ``{name: value}`` dict; see :meth:`set`."""
        return set_parameters(self, params)

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
        inst = self._resolve_instrument(instrument)
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
        # ``replace`` rather than a field-by-field rebuild: a new
        # SourceObsData field would otherwise be silently dropped here.
        new_od = dataclasses.replace(
            od,
            source_weights=weights,
            per_obs=tuple(dict.fromkeys(od.per_obs + ("source_weights",))),
        )
        new_bundle = dataclasses.replace(bundle, obs_data={**bundle.obs_data, src: new_od})
        return eqx.tree_at(lambda s: s._obs_bundles[inst], self, new_bundle)

    def _resolve_instrument(self, instrument: str | None) -> str:
        """Validate an instrument name, or fall back to the only one."""
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

    def __len__(self) -> int:
        """Number of instruments."""
        return len(self.instruments)

    def __getitem__(self, name: str) -> InstrumentModel:
        """Return the instrument model for *name*."""
        return self.instruments[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.instruments)
