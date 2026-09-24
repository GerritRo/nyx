from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from nyx.core.parameter import is_parameter
from nyx.core.paramtree import (
    ParametersTable,
    dump_params,
    freeze_all,
    n_trainable,
    parameters_table,
    set_parameters,
)

__all__ = [
    "MultiTargetFit",
]

if TYPE_CHECKING:
    from nyx.core.scene import Scene

# Scene fields that can link across targets.
_SHAREABLE_FIELDS = ("atmosphere", "sources")


def _signature(tree: Any) -> tuple[Any, tuple[Any, ...]]:
    """Return ``(treedef, leaf shapes)`` with Parameters treated as leaves."""
    leaves, treedef = jax.tree_util.tree_flatten(tree, is_leaf=is_parameter)
    shapes = tuple(jnp.shape(leaf.factor) if is_parameter(leaf) else None for leaf in leaves)
    return treedef, shapes


def _freeze_where[T](tree: T, predicate: Callable[[Any], bool]) -> T:
    """Return *tree* with every Parameter satisfying *predicate* frozen."""

    def at_leaf(x: Any) -> Any:
        if is_parameter(x) and predicate(x):
            return x.freeze()
        return x

    return jax.tree.map(at_leaf, tree, is_leaf=is_parameter)


def _field_getter(name: str) -> Callable[[Any], Any]:
    """A ``tree -> tree.<name>`` accessor, for :func:`equinox.tree_at`.

    Parameters
    ----------
    name : str

    Returns
    -------
    callable
    """
    return lambda tree: getattr(tree, name)


class MultiTargetFit(eqx.Module):
    """Joint fit over several targets with shared instruments.

    Instruments of the same name across scenes are linked: their non-per-obs
    Parameters live once in ``canonical_instruments`` and are injected into
    every scene at render time, so their gradients sum over targets.

    Parameters
    ----------
    scenes : dict of str to Scene
        One pre-built Scene per target.
    share : str or iterable of str, optional
        Scene fields to link across targets, from ``'atmosphere'`` and
        ``'sources'``. A linked field must have the same structure and
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
                            is_leaf=is_parameter,
                        )

                    if structure(canonical[name]) != structure(inst):
                        raise ValueError(
                            f"Instrument {name!r} has a different structure "
                            f"across scenes; instruments linked by name must "
                            f"be the same type with the same parameters."
                        )
                    continue
                # Freeze per-obs canonical copies
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
                    _field_getter(field),
                    scene,
                    freeze_all(getattr(scene, field)),
                )
                for t, scene in target_scenes.items()
            }
        self.target_scenes = target_scenes

    def _inject_shared(self, scene: Scene) -> Scene:
        """Replace *scene*'s shared Parameters with their canonical counterparts."""

        for field, canonical_field in self.canonical_fields.items():
            scene = eqx.tree_at(
                _field_getter(field),
                scene,
                canonical_field,
            )

        def pick(a: Any, b: Any) -> Any:
            return b if (is_parameter(b) and not b.per_obs) else a

        for inst_name in scene.instruments:
            if inst_name not in self.canonical_instruments:
                continue
            merged = jax.tree.map(
                pick,
                scene.instruments[inst_name],
                self.canonical_instruments[inst_name],
                is_leaf=is_parameter,
            )

            def _get_inst(s: Any, _n: str = inst_name) -> Any:
                return s.instruments[_n]

            scene = eqx.tree_at(_get_inst, scene, merged)
        return scene

    def render(self) -> dict[str, dict[str, jax.Array]]:
        """Render all targets.

        Returns
        -------
        dict of str to dict of str to jax.Array
            Keyed by target name, then instrument name.
        """
        return {t: self._inject_shared(s).render() for t, s in self.target_scenes.items()}

    def __repr__(self) -> str:
        """Targets, shared instruments and linked fields, not N whole Scenes."""
        shared = ", ".join(self.canonical_fields) or "none"
        return (
            f"MultiTargetFit(targets: {', '.join(self.target_names) or '(none)'}; "
            f"instruments: {', '.join(self.instrument_names) or '(none)'}; "
            f"linked fields: {shared}; "
            f"{n_trainable(self)} parameter values free)"
        )

    @property
    def instrument_names(self) -> tuple[str, ...]:
        """Names of the shared instruments.

        Returns
        -------
        tuple of str
        """
        return tuple(self.canonical_instruments.keys())

    @property
    def target_names(self) -> tuple[str, ...]:
        """Names of the fitted targets.

        Returns
        -------
        tuple of str
        """
        return tuple(self.target_scenes.keys())

    def _display_tree(self) -> dict[str, Any]:
        """Pytree view without the frozen shadow Parameters.

        Shared instrument Parameters appear once, alongside each target's own.
        """

        def strip(tree: Any, drop: Callable[[Any], bool]) -> Any:
            def _pick(x: Any) -> Any:
                return None if (is_parameter(x) and drop(x)) else x

            return jax.tree.map(_pick, tree, is_leaf=is_parameter)

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
        """Set one parameter by name; see :meth:`set_params`.

        Parameters
        ----------
        path : str
        value : array-like

        Returns
        -------
        MultiTargetFit
        """
        return self.set_params({path: value})

    def set_params(self, params: dict[str, Any]) -> MultiTargetFit:
        """Set parameters by the names :meth:`parameters_table` prints.

        Parameters
        ----------
        params : dict of str to array-like
            Names or glob patterns, as :func:`~nyx.core.paramtree.freeze`
            takes. A raw value is wrapped preserving the target's metadata.

        Returns
        -------
        MultiTargetFit
        """
        return set_parameters(self, params)

    def parameters_table(self) -> ParametersTable:
        """Table of every fitted Parameter.

        Returns
        -------
        ParametersTable
        """
        return parameters_table(self._display_tree())

    def dump_params(self) -> dict[str, np.ndarray]:
        """Physical value of every fitted Parameter.

        Returns
        -------
        dict of str to numpy.ndarray
        """
        return dump_params(self._display_tree())
