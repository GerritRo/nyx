from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from nyx.core.filters import per_obs_filter
from nyx.core.geometry import check_shared_geometry
from nyx.core.io import save_fit
from nyx.core.observation import Observation
from nyx.core.paramtree import (
    ParametersTable,
    hide_path_segments,
    n_trainable,
    parameters_table,
    set_parameters,
)
from nyx.core.pipeline import RenderFrame
from nyx.core.pipeline import contributions as _contributions
from nyx.core.pipeline import render as _render_single
from nyx.core.protocols import AtmosphereModel, InstrumentModel, SkySource
from nyx.core.records import RenderGeometry, SourceObsData

# ``instruments.CT1.shift`` reads better as ``CT1.shift``.
hide_path_segments("instruments", "sources")


class ObsBundle(eqx.Module):
    """Precomputed source data, render geometry and obs count for one instrument."""

    obs_data: dict[str, SourceObsData]
    render_geometry: RenderGeometry
    nobs: int = eqx.field(static=True)


def _as_named_dict[T](
    value: T | dict[str, T] | list[T] | tuple[T, ...],
    single: type | tuple[type, ...],
    default_name: str,
) -> dict[str, T]:
    """Normalise one-or-many into ``{name: value}``.

    Parameters
    ----------
    value : object, dict, list or tuple
    single : type or tuple of type
        What counts as a single item rather than a collection.
    default_name : str
        Name given to a lone item.

    Returns
    -------
    dict
        Sequence entries are named after their class.
    """
    if isinstance(value, single):
        return {default_name: value}
    if isinstance(value, (list, tuple)):
        return {type(v).__name__: v for v in value}
    return dict(value)


def _match_observations(
    obs_list: Observation | dict[str, Observation] | list[Observation],
    instruments: dict[str, InstrumentModel],
) -> dict[str, Observation]:
    """Give every instrument an Observation, keyed the same way.

    Parameters
    ----------
    obs_list : Observation, dict or list
        Broadcast if single, zipped in order if a sequence, matched by key
        if a dict.
    instruments : dict of str to InstrumentModel

    Returns
    -------
    dict of str to Observation

    Raises
    ------
    ValueError
        If a sequence has the wrong length, or a dict has different keys.
    """
    if isinstance(obs_list, Observation):
        return dict.fromkeys(instruments, obs_list)
    if isinstance(obs_list, (list, tuple)):
        if len(obs_list) != len(instruments):
            raise ValueError(
                f"len(obs_list)={len(obs_list)} does not match "
                f"len(instruments)={len(instruments)}"
            )
        return dict(zip(instruments.keys(), obs_list, strict=True))
    if set(instruments) != set(obs_list):
        raise ValueError(
            f"instrument keys {set(instruments)} do not match obs keys {set(obs_list)}"
        )
    return dict(obs_list)


def _prepare_scene_parts(
    instruments: InstrumentModel | dict[str, InstrumentModel],
    atmosphere: AtmosphereModel,
    sources: dict[str, Any] | list[Any],
    obs_list: Observation | dict[str, Observation] | list[Observation],
) -> tuple[dict[str, Any], dict[str, InstrumentModel], dict[str, ObsBundle]]:
    """Normalise, validate, and precompute everything a Scene is made of.

    Parameters
    ----------
    instruments : InstrumentModel or dict of str to InstrumentModel
    atmosphere : AtmosphereModel
    sources : dict of str to EmitterLike, or list
    obs_list : Observation, dict of str to Observation, or list

    Returns
    -------
    scene_sources : dict of str to SkySource
        The shared source models that go in the scene pytree.
    prepared_instruments : dict of str to InstrumentModel
        Instruments batched to their observation count.
    obs_bundles : dict of str to ObsBundle
        Per-instrument precomputed observation data.
    """
    instruments = _as_named_dict(instruments, InstrumentModel, "instrument")
    sources = _as_named_dict(sources, (), "source")
    obs_by_instrument = _match_observations(obs_list, instruments)

    check_shared_geometry(obs_by_instrument, atmosphere, instruments, sources)

    scene_sources = {name: src.model() for name, src in sources.items()}

    prepared_instruments: dict[str, InstrumentModel] = {}
    obs_bundles: dict[str, ObsBundle] = {}
    for inst_name, inst in instruments.items():
        obs = obs_by_instrument[inst_name]
        geoms = obs.get_render_geometry()
        prepared_instruments[inst_name] = inst.prepare(obs)
        obs_bundles[inst_name] = ObsBundle(
            obs_data={name: src.prepare(obs) for name, src in sources.items()},
            render_geometry=jax.tree.map(lambda *xs: jnp.stack(xs), *geoms),
            nobs=obs.nobs,
        )

    return scene_sources, prepared_instruments, obs_bundles


class Scene(eqx.Module):
    """Multi-instrument observation of a shared physical sky.

    Atmosphere and source models are shared across instruments, and every
    instrument's gradients reach the same parameters; each may have its own
    ``nobs``.  Instruments and sources are reachable as attributes by name,
    e.g. ``scene.CT1.shift``.  Build with :meth:`build`.

    Parameters
    ----------
    atmosphere : AtmosphereModel
    sources : dict of str to SkySource
        Shared source models, as returned by each builder's ``model()``.
    instruments : dict of str to InstrumentModel
        Prepared instrument models.
    _obs_bundles : dict of str to ObsBundle
        Per-instrument observation data; keys must match *instruments*.
    """

    atmosphere: AtmosphereModel
    sources: dict[str, SkySource]
    instruments: dict[str, InstrumentModel]
    _obs_bundles: dict[str, ObsBundle]

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
        """What the scene is, not what it contains."""
        counts = ", ".join(f"{name}({n} obs)" for name, n in self.nobs.items())
        n_free = n_trainable(self)
        total = n_trainable(self, include_frozen=True)
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

        Parameters
        ----------
        instruments : InstrumentModel or dict of str to InstrumentModel
            A lone instrument is wrapped as ``{'instrument': inst}``.
        atmosphere : AtmosphereModel
        sources : dict of str to EmitterLike, or list
            Each must provide ``model()`` and ``prepare(obs)``.  A list is
            auto-named from class names.
        obs_list : Observation, or dict of str to Observation
            A lone Observation is broadcast to every instrument; a dict must
            have the same keys as *instruments*.

        Returns
        -------
        Scene
        """
        scene_sources, prepared_instruments, obs_bundles = _prepare_scene_parts(
            instruments, atmosphere, sources, obs_list
        )
        return cls(
            atmosphere=atmosphere,
            sources=scene_sources,
            instruments=prepared_instruments,
            _obs_bundles=obs_bundles,
        )

    def _render_frame(self, inst_name: str) -> RenderFrame:
        """Combine the shared sky with one instrument's observation data."""
        bundle = self._obs_bundles[inst_name]
        od = bundle.obs_data
        return RenderFrame(
            atmosphere=self.atmosphere,
            sources=[(self.sources[n], od[n]) for n in od],
            instrument=self.instruments[inst_name],
            render_geometry=bundle.render_geometry,
            nobs=bundle.nobs,
            source_names=tuple(od),
        )

    def render(self, *, per_source: bool = False) -> dict[str, Any]:
        """Render all observations to pixel rates.

        Parameters
        ----------
        per_source : bool
            Whether to split each instrument's rates by emitter.  The parts
            sum to the whole to float32 rounding.

        Returns
        -------
        dict of str to jax.Array
            One array per instrument, of shape ``(nobs, n_pixels)``; or,
            with *per_source*, ``{instrument: {source: array}}``.
        """
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
        """Set one parameter by its dotted path, e.g. ``'atmosphere.Mie.aod_500'``.

        Parameters
        ----------
        path : str
        value : array-like or Parameter

        Returns
        -------
        Scene
        """
        return self.set_params({path: value})

    def set_params(self, params: dict[str, Any]) -> Scene:
        """Set several parameters at once; see :meth:`set`.

        Parameters
        ----------
        params : dict of str to array-like

        Returns
        -------
        Scene
        """
        return set_parameters(self, params)

    def parameters_table(self) -> ParametersTable:
        """Table of every Parameter in the scene, frozen ones included.

        Returns
        -------
        ParametersTable
        """
        return parameters_table(self)

    def save(self, path: str | os.PathLike[str], observations: dict[str, Observation]) -> None:
        """Write a fit bundle -- parameters plus observation metadata -- to *path*.

        Parameters
        ----------
        path : str or path-like
        observations : dict of str to Observation
            Keyed by instrument name; see :func:`~nyx.core.io.save_fit`.
        """
        save_fit(path, self, observations)

    def __len__(self) -> int:
        """Number of instruments."""
        return len(self.instruments)

    def __getitem__(self, name: str) -> InstrumentModel:
        """Return the instrument model for *name*."""
        return self.instruments[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.instruments)
