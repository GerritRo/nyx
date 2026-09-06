"""Joint fits over several targets."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from nyx.core.parameter import (
    _is_param,
    _ParametersTable,
    dump_params,
    freeze_all,
    n_trainable,
    parameters_table,
    set_parameters,
)

if TYPE_CHECKING:
    from nyx.core.scene import Scene

#: Scene fields that :class:`MultiTargetFit` can link across targets.
_SHAREABLE_FIELDS = ("atmosphere", "sources")


def _signature(tree: Any) -> tuple[Any, tuple[Any, ...]]:
    """Return ``(treedef, leaf shapes)`` with Parameters treated as leaves.

    Two subtrees with equal signatures can be swapped for one another with
    :func:`equinox.tree_at`.
    """
    leaves, treedef = jax.tree_util.tree_flatten(tree, is_leaf=_is_param)
    shapes = tuple(jnp.shape(leaf.factor) if _is_param(leaf) else None for leaf in leaves)
    return treedef, shapes


def _freeze_where[T](tree: T, predicate: Callable[[Any], bool]) -> T:
    """Return *tree* with every Parameter satisfying *predicate* frozen."""

    def at_leaf(x: Any) -> Any:
        if _is_param(x) and predicate(x):
            return x.freeze()
        return x

    return jax.tree.map(at_leaf, tree, is_leaf=_is_param)


class MultiTargetFit(eqx.Module):
    """Joint fit over multiple observation targets with shared instruments.

    Each target has its own :class:`~nyx.core.scene.Scene`.  Instruments
    with the same name across scenes are linked: their shared (non-per-obs)
    Parameters live once in ``canonical_instruments`` and are injected into
    every scene at render time, so JAX sums their gradients over all
    targets.  Per-obs Parameters (``shift``, ``rotation``) stay on the
    individual scenes, and so does per-target scene state (``atmosphere``,
    ``sources``) unless it is listed in *share*.

    When several scenes share an instrument name, the canonical shared
    Parameter values come from the first scene to define that name; the
    canonical values of the fields in *share* likewise come from the first
    scene.

    Fit it with a plain :class:`Optimizer`::

        import optimistix as optx
        mtf = MultiTargetFit({'A': scene_A, 'B': scene_B})
        opt = Optimizer(loss_fn, optx.BFGS(rtol=1e-5, atol=1e-5))

    Fit one common atmosphere over both targets instead of one per
    target::

        mtf = MultiTargetFit({'A': scene_A, 'B': scene_B}, share='atmosphere')
        mtf.parameters_table()   # 'shared.atmosphere.Mie.aod_500', listed once

    Parameters
    ----------
    scenes : dict
        ``{target_name: Scene}``, one pre-built Scene per target.
    share : str or iterable of str, optional
        Scene fields to link across all targets, from ``'atmosphere'`` and
        ``'sources'``.  A linked field lives once in ``canonical_fields``
        and is injected into every scene at render time, so its Parameters
        are fitted jointly.  Linked fields must have the same structure and
        Parameter shapes in every scene.
    """

    canonical_instruments: dict[str, Any]
    canonical_fields: dict[str, Any]
    target_scenes: dict[str, Scene]

    def __init__(
        self,
        scenes: dict[str, Scene],
        share: str | Iterable[str] = (),
    ) -> None:
        share = (share,) if isinstance(share, str) else tuple(share)
        for field in share:
            if field not in _SHAREABLE_FIELDS:
                raise ValueError(
                    f"Cannot share {field!r} across targets; shareable "
                    f"fields are {_SHAREABLE_FIELDS}."
                )

        canonical: dict[str, Any] = {}
        for scene in scenes.values():
            for name, inst in scene.instruments.items():
                if name in canonical:

                    def structure(t: Any) -> Any:
                        return jax.tree_util.tree_structure(
                            t,
                            is_leaf=_is_param,
                        )

                    if structure(canonical[name]) != structure(inst):
                        raise ValueError(
                            f"Instrument {name!r} has a different structure "
                            f"across scenes; instruments linked by name must "
                            f"be the same type with the same parameters."
                        )
                    continue
                # Per-obs Parameters stay on each scene; freeze the canonical
                # copies so the optimizer does not train unused leaves.
                canonical[name] = _freeze_where(inst, lambda p: p.per_obs)

        self.canonical_instruments = canonical

        first = next(iter(scenes.values()))
        for field in share:
            reference = getattr(first, field)
            for t, scene in scenes.items():
                if _signature(getattr(scene, field)) != _signature(reference):
                    raise ValueError(
                        f"Field {field!r} of target {t!r} does not match that "
                        f"of target {next(iter(scenes))!r}; shared fields must "
                        f"have the same structure and Parameter shapes in "
                        f"every scene."
                    )
        self.canonical_fields = {field: getattr(first, field) for field in share}

        # Each scene keeps a frozen shadow of the shared Parameters; the
        # canonical copies are injected over them at render time.
        def _get_instruments(s: Any) -> Any:
            return s.instruments

        target_scenes = {
            t: eqx.tree_at(
                _get_instruments,
                scene,
                _freeze_where(scene.instruments, lambda p: not p.per_obs),
            )
            for t, scene in scenes.items()
        }
        for field in share:
            target_scenes = {
                t: eqx.tree_at(
                    lambda s, _f=field: getattr(s, _f),
                    scene,
                    freeze_all(getattr(scene, field)),
                )
                for t, scene in target_scenes.items()
            }
        self.target_scenes = target_scenes

    def _inject_shared(self, scene: Scene) -> Scene:
        """Replace the shared Parameters in *scene* -- the non-per-obs ones
        in its instruments, plus any linked field -- with their canonical
        counterparts."""

        for field, canonical_field in self.canonical_fields.items():
            scene = eqx.tree_at(
                lambda s, _f=field: getattr(s, _f),
                scene,
                canonical_field,
            )

        def pick(a: Any, b: Any) -> Any:
            return b if (_is_param(b) and not b.per_obs) else a

        for inst_name in scene.instruments:
            if inst_name not in self.canonical_instruments:
                continue
            merged = jax.tree.map(
                pick,
                scene.instruments[inst_name],
                self.canonical_instruments[inst_name],
                is_leaf=_is_param,
            )

            def _get_inst(s: Any, _n: str = inst_name) -> Any:
                return s.instruments[_n]

            scene = eqx.tree_at(_get_inst, scene, merged)
        return scene

    def render(self) -> dict[str, dict[str, jax.Array]]:
        """Render all targets.

        Returns
        -------
        dict of ``{target_name: {inst_name: jax.Array}}``
        """
        return {t: self._inject_shared(s).render() for t, s in self.target_scenes.items()}

    def __repr__(self) -> str:
        """Targets, shared instruments and linked fields -- not N whole Scenes."""
        shared = ", ".join(self.canonical_fields) or "none"
        return (
            f"MultiTargetFit(targets: {', '.join(self.target_names) or '(none)'}; "
            f"instruments: {', '.join(self.instrument_names) or '(none)'}; "
            f"linked fields: {shared}; "
            f"{n_trainable(self)} parameter values free)"
        )

    @property
    def instrument_names(self) -> tuple[str, ...]:
        """Names of the shared instruments."""
        return tuple(self.canonical_instruments.keys())

    @property
    def target_names(self) -> tuple[str, ...]:
        """Names of the fitted targets."""
        return tuple(self.target_scenes.keys())

    def _display_tree(self) -> dict[str, Any]:
        """Pytree view with frozen shadow Parameters removed: shared
        instrument Parameters once, plus each target's own Parameters."""

        def strip(tree: Any, drop: Callable[[Any], bool]) -> Any:
            def _pick(x: Any) -> Any:
                return None if (_is_param(x) and drop(x)) else x

            return jax.tree.map(_pick, tree, is_leaf=_is_param)

        # Keyed exactly as the real tree is, so the names this produces are
        # the ones set_params, freeze and unfreeze accept:
        # 'canonical_instruments' and 'target_scenes' are stripped from
        # displayed paths, giving 'CT1.efficiency' and 'A.CT1.shift'.
        display: dict[str, Any] = {
            "canonical_instruments": strip(self.canonical_instruments, lambda p: p.per_obs)
        }
        display.update(self.canonical_fields)

        per_target_all: dict[str, Any] = {}
        for t, scene in self.target_scenes.items():
            per_target = {
                field: getattr(scene, field)
                for field in _SHAREABLE_FIELDS
                if field not in self.canonical_fields
            }
            per_target["instruments"] = strip(scene.instruments, lambda p: not p.per_obs)
            per_target_all[t] = per_target
        display["target_scenes"] = per_target_all
        return display

    def set(self, path: str, value: Any) -> MultiTargetFit:
        """Set one parameter by name; see :meth:`set_params`."""
        return self.set_params({path: value})

    def set_params(self, params: dict[str, Any]) -> MultiTargetFit:
        """Set parameters by the names :meth:`parameters_table` prints.

        The same spelling :func:`~nyx.core.parameter.freeze` takes, with
        the same shell globbing, so the long chains this class would
        otherwise need (``m.target_scenes[t].instruments['CT1'].shift``)
        are not the only way in::

            mtf = mtf.set_params({'CT1.efficiency': 0.8, 'A.CT1.shift': shifts})
            mtf = mtf.set('*.rotation', 0.0)   # every target

        A raw value is auto-wrapped, preserving the target's ``scale``,
        ``per_obs``, ``frozen`` and ``transform``.
        """
        return set_parameters(self, params)

    def parameters_table(self) -> _ParametersTable:
        """Return a table of every fitted Parameter.

        Shared instrument Parameters are listed once; the frozen shadow
        copies on the individual scenes are omitted.
        """
        return parameters_table(self._display_tree())

    def dump_params(self) -> dict[str, np.ndarray]:
        """Return ``{path: ndarray}`` of every fitted Parameter's value."""
        return dump_params(self._display_tree())
