from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.flatten_util as fu
import jax.numpy as jnp
import numpy as np
import optimistix as optx

from nyx.core.parameter import (
    Parameter,
    _is_param,
    _ParametersTable,
    dump_params,
    freeze_all,
    parameters_table,
)

if TYPE_CHECKING:
    from nyx.core.scene import Scene

__all__ = ["Optimizer", "MultiTargetFit", "parameter_errors", "rescale_from_errors"]


def _is_trainable(x: Any) -> bool:
    return isinstance(x, Parameter) and not x.frozen


def _sum_of_squares(residuals: Any) -> jax.Array:
    """Reduce a residual array or pytree of arrays to ``sum(r ** 2)``."""
    return jax.tree.reduce(
        lambda a, b: a + b,
        jax.tree.map(lambda x: jnp.sum(x**2), residuals),
    )


class Optimizer:
    """Fit any pytree of Parameters with a minimiser or least-squares solver.

    Trains every non-frozen :class:`Parameter` reachable from *model*.
    Per-obs parameters receive independent per-observation gradients;
    global parameters receive gradients summed over observations.

    *fn* depends on the solver type:

    * minimiser (``optx.BFGS`` and similar): *fn* returns a scalar loss.
    * least-squares solver (``optx.LevenbergMarquardt``, ``GaussNewton``,
      ``Dogleg``): *fn* returns a residual array or pytree of arrays.

    A least-squares solver is usually fastest for chi-squared losses
    ``sum(((pred - target) / err) ** 2)``. A non-scalar *fn* passed to a
    minimiser is auto-wrapped with sum-of-squares; a scalar *fn* passed
    to a least-squares solver raises ``TypeError``.

    Parameters
    ----------
    fn : callable
        ``(model) -> scalar`` for a minimiser, or ``(model) -> residuals``
        for a least-squares solver.
    solver : optimistix.AbstractIterativeSolver
        A minimiser or a least-squares solver, e.g.
        ``optx.LevenbergMarquardt(rtol=1e-5, atol=1e-5)``.

    Examples
    --------
    Least-squares fit, run to convergence::

        import optimistix as optx

        def residuals(scene):
            preds = scene.render()['instrument']
            return (preds - targets) / (targets * 0.1)

        opt = Optimizer(residuals, optx.LevenbergMarquardt(rtol=1e-5, atol=1e-5))
        scene, sol = opt.run(scene, max_steps=256)

    Manual stepping, for progress reporting (minimiser solvers only;
    least-squares solver state cannot pass through :func:`jax.jit`)::

        opt = Optimizer(loss_fn, optx.BFGS(rtol=1e-5, atol=1e-5))
        state = opt.init_state(scene)
        step = jax.jit(opt.step)
        for _ in range(200):
            scene, loss, state = step(scene, state)
    """

    def __init__(self, fn: Callable[[Any], Any], solver: Any) -> None:
        self._solver = solver
        self._is_ls = isinstance(solver, optx.AbstractLeastSquaresSolver)
        self._fn_user = fn
        self._fn_is_scalar: bool | None = None  # resolved on first call to _check_fn

    def _check_fn(self, model: Any) -> None:
        """Probe fn output shape and cache _fn_is_scalar. Call with a concrete model."""
        if self._fn_is_scalar is not None:
            return
        probe = jax.eval_shape(lambda: self._fn_user(model))
        fn_is_scalar = isinstance(probe, jax.ShapeDtypeStruct) and probe.shape == ()
        if self._is_ls and fn_is_scalar:
            raise TypeError(
                f"{type(self._solver).__name__} is a least-squares solver; "
                "fn must return a residuals array (or pytree of arrays), "
                "not a scalar."
            )
        self._fn_is_scalar = fn_is_scalar

    def _make_inner(self, model: Any) -> tuple[Callable[..., Any], Any, Any]:
        """Partition *model* and build the inner fn for optimistix."""
        diff, static = eqx.partition(model, _is_trainable, is_leaf=_is_param)
        fn_user = self._fn_user
        user_fn: Callable[[Any], Any]
        if not self._is_ls and not self._fn_is_scalar:

            def user_fn(m: Any) -> Any:
                return _sum_of_squares(fn_user(m))
        else:
            user_fn = fn_user

        def inner(diff: Any, args: Any) -> tuple[Any, None]:
            return user_fn(eqx.combine(diff, static, is_leaf=_is_param)), None

        return inner, diff, static

    def init_state(self, model: Any) -> Any:
        """Compute the initial solver state for manual stepping.

        Parameters
        ----------
        model : pytree
            The starting model.

        Returns
        -------
        state : optimistix solver state
            Pass as *state* to the first :meth:`step` call.
        """
        self._check_fn(model)
        inner, diff, _ = self._make_inner(model)
        f_struct = jax.eval_shape(lambda: inner(diff, None)[0])
        aux_struct = None
        return self._solver.init(inner, diff, None, {}, f_struct, aux_struct, frozenset())

    def step[T](self, model: T, state: Any) -> tuple[T, jax.Array, Any]:
        """One solver step.

        Call :meth:`init_state` with a concrete model before JIT-compiling
        this method so that ``_fn_is_scalar`` is resolved first.

        Parameters
        ----------
        model : pytree
            Current model.
        state : optimistix solver state
            From :meth:`init_state` or a previous :meth:`step` call.

        Returns
        -------
        model : pytree
            Updated model.
        loss : jax.Array
            Scalar loss at *model* before the step was applied. On the
            least-squares path this is ``sum(r ** 2)``.
        state : optimistix solver state
            Updated solver state.
        """
        if self._fn_is_scalar is None:
            self._check_fn(model)
        inner, diff, static = self._make_inner(model)
        if self._fn_is_scalar:
            loss = self._fn_user(model)
        else:
            loss = _sum_of_squares(self._fn_user(model))
        new_diff, new_state, _ = self._solver.step(inner, diff, None, {}, state, frozenset())
        new_model = eqx.combine(new_diff, static, is_leaf=_is_param)
        return new_model, loss, new_state

    def run[T](
        self, model: T, *, max_steps: int = 256, throw: bool = True
    ) -> tuple[T, optx.Solution[Any, Any]]:
        """Run the solver to convergence.

        Dispatches to :func:`optimistix.least_squares` or
        :func:`optimistix.minimise` depending on the solver type. Both
        wrap the iteration in ``lax.while_loop``, so the whole
        convergence compiles once and runs inside XLA.

        Parameters
        ----------
        model : pytree
            Starting model.
        max_steps : int, optional
            Maximum solver iterations (default 256).
        throw : bool, optional
            If True, raise on non-successful termination (default).
            Set False to inspect ``sol.result`` manually.

        Returns
        -------
        model : pytree
            Fitted model.
        sol : optimistix.Solution
            The full solver solution (``sol.value`` is the trainable
            pytree, ``sol.result`` the status, ``sol.stats`` the counts).
        """
        self._check_fn(model)
        inner, diff, static = self._make_inner(model)
        entry = optx.least_squares if self._is_ls else optx.minimise
        sol: optx.Solution[Any, Any] = entry(
            inner,
            self._solver,
            diff,
            args=None,
            has_aux=True,
            max_steps=max_steps,
            throw=throw,
        )
        fitted = eqx.combine(sol.value, static, is_leaf=_is_param)
        return fitted, sol

    def errors[T](
        self,
        fitted: T,
        *,
        batch_size: int = 32,
        reduced_chi2: bool = False,
        rcond: float | None = None,
    ) -> T:
        """1-σ errors on every trainable Parameter at *fitted*.

        Computes the covariance matrix-free (no dense Jacobian
        materialised) via the Moore–Penrose pseudoinverse of ``JᵀJ``
        and returns ``sqrt(diag(cov))`` as a pytree with the same
        structure as the trainable subset of *fitted*.

        Only valid when *fn* returns residuals (least-squares solver, or
        minimiser with a residuals fn that was auto-wrapped). For a
        scalar-loss minimiser fit the J^T J formula does not apply;
        compute a Hessian-based error estimate instead.

        Parameters
        ----------
        fitted : pytree
            Model returned by :meth:`run`.
        batch_size : int, optional
            Columns of ``J^T J`` computed in parallel per chunk.
        reduced_chi2 : bool, optional
            If False (default), assumes residuals are pre-scaled by 1-σ
            uncertainties so ``cov = pinv(J^T J)``.  If True, applies
            the ``(r^T r) / (m - n)`` factor so errors reflect the
            spread consistent with the data.
        rcond : float or None, optional
            Relative singular-value cutoff for the pseudoinverse.  See
            :func:`parameter_errors` for details.

        See :func:`parameter_errors` for the standalone form and full
        docstring.
        """
        if self._fn_is_scalar is None:
            raise RuntimeError(
                "Optimizer.errors() called before run() or init_state(); call opt.run(model) first."
            )
        if self._fn_is_scalar:
            raise TypeError(
                "Optimizer.errors() needs residuals; this Optimizer was "
                "built with a scalar loss fn. Pass a residuals fn to "
                "parameter_errors() directly, or rebuild the Optimizer "
                "with a residuals fn."
            )
        return parameter_errors(
            fitted,
            self._fn_user,
            batch_size=batch_size,
            reduced_chi2=reduced_chi2,
            rcond=rcond,
        )


def parameter_errors[T](
    fitted: T,
    residuals_fn: Callable[[Any], Any],
    *,
    batch_size: int = 1,
    reduced_chi2: bool = False,
    rcond: float | None = None,
) -> T:
    """Get parameter errors via jacobian"""
    diff, static = eqx.partition(fitted, _is_trainable, is_leaf=_is_param)
    flat, unravel = fu.ravel_pytree(diff)
    n = flat.size

    def residuals_of_flat(x: jax.Array) -> jax.Array:
        return fu.ravel_pytree(residuals_fn(eqx.combine(unravel(x), static, is_leaf=_is_param)))[0]

    # Linearise once; jvp_fn is a pure linear operator, no retracing.
    r0, jvp_fn = jax.linearize(residuals_of_flat, flat)
    vjp_fn = jax.linear_transpose(jvp_fn, flat)  # y -> (J^T y,)

    def jtj_col(e: jax.Array) -> jax.Array:
        (JtJe,) = vjp_fn(jvp_fn(e))
        return JtJe

    JtJ = jax.lax.map(jax.jit(jtj_col), jnp.eye(n), batch_size=batch_size)
    JtJ64 = JtJ.astype(jnp.float64)

    _, s, vh = jnp.linalg.svd(JtJ64, hermitian=True)
    smax = s[0]
    tol = (rcond if rcond is not None else n * jnp.finfo(jnp.float64).eps) * smax
    mask = s > tol
    s_inv = jnp.where(mask, 1.0 / s, 0.0)
    cov = (vh.T * s_inv) @ vh

    n_null = int(jnp.sum(~mask))
    if n_null > 0:
        smin = float(s[-1])
        cond = float(smax / smin) if smin > 0 else float("inf")
        warnings.warn(
            f"parameter_errors: J^T J is rank-deficient "
            f"({n_null} singular value(s) <= rcond * s_max; "
            f"condition number {cond:.3e}).  Returned σ are from the "
            f"Moore–Penrose pseudoinverse and describe the identifiable "
            f"projection of each parameter; unidentifiable directions "
            f"get zero error by construction.  A common cause is the "
            f"degeneracy between a global `efficiency` and the overall "
            f"scale of `pixel_efficiency` (flatfield normalisation).  "
            f"Freeze one of the degenerate parameters via "
            f"`nyx.core.parameter.freeze` to obtain a unique MLE.",
            stacklevel=2,
        )

    if reduced_chi2:
        m = r0.size
        cov = cov * (jnp.sum(r0**2) / (m - n))

    sigma_flat = jnp.sqrt(jnp.diag(cov)).astype(flat.dtype)
    return unravel(sigma_flat)


def rescale_from_errors[T](fitted: T, errs: T) -> T:
    """
    Rescale every trainable Parameter using error estimates as new scales.
    This normalises parameter magnitudes across physically disparate 
    quantities and can improve convergence in a subsequent fit.
    
    Parameters
    ----------
    fitted : pytree
        Model as returned by :meth:`Optimizer.run`.
    errs : pytree
        Error estimates as returned by :meth:`Optimizer.errors` or
        :func:`parameter_errors`.  Must share the trainable-parameter
        structure of *fitted*.
    Returns
    -------
    pytree
        Copy of *fitted* with every trainable Parameter rescaled.
    """
    diff, static = eqx.partition(fitted, _is_trainable, is_leaf=_is_param)
    def rescale_leaf(param: Any, err: Any) -> Any:
        if not (_is_param(param) and _is_param(err)):
            return param
        new_scale = float(jnp.max(jnp.abs(err.value)))
        if new_scale == 0.0 or not np.isfinite(new_scale):
            return param
        return Parameter.from_value(
            param.value,
            scale=new_scale,
            per_obs=param.per_obs,
            frozen=param.frozen,
        )
    rescaled = jax.tree.map(rescale_leaf, diff, errs, is_leaf=_is_param)
    return eqx.combine(rescaled, static, is_leaf=_is_param)

    
# Multi-target fitting


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

        shared: dict[str, Any] = {
            # 'instruments' is stripped from displayed paths, so the shared
            # instrument Parameters still read as 'shared.CT1.efficiency'.
            "instruments": strip(self.canonical_instruments, lambda p: p.per_obs),
        }
        shared.update(self.canonical_fields)

        display: dict[str, Any] = {"shared": shared}
        for t, scene in self.target_scenes.items():
            per_target = {
                field: getattr(scene, field)
                for field in _SHAREABLE_FIELDS
                if field not in self.canonical_fields
            }
            per_target["instruments"] = strip(scene.instruments, lambda p: not p.per_obs)
            display[t] = per_target
        return display

    def parameters_table(self) -> _ParametersTable:
        """Return a table of every fitted Parameter.

        Shared instrument Parameters are listed once; the frozen shadow
        copies on the individual scenes are omitted.
        """
        return parameters_table(self._display_tree())

    def dump_params(self) -> dict[str, np.ndarray]:
        """Return ``{path: ndarray}`` of every fitted Parameter's value."""
        return dump_params(self._display_tree())