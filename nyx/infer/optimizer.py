"""The fit driver."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import numpy as np
import optimistix as optx

from nyx.core.parameter import _is_param, freeze, n_trainable, set_parameters
from nyx.infer._common import (
    _is_trainable,
    _non_finite_parameters,
    _sum_of_squares,
    _warn_non_finite,
)
from nyx.infer.profile import ProfileGrid
from nyx.infer.uncertainty import parameter_errors


def _result_text(sol: optx.Solution[Any, Any]) -> str:
    """Solver status as a sentence.

    ``str(sol.result)`` renders as ``optimistix._solution.RESULTS<>``.
    """
    message = str(optx.RESULTS[sol.result]).strip()
    return message or "converged"


@dataclasses.dataclass(frozen=True)
class FitSummary:
    """How a fit came out.

    ``reduced_chi2`` is the number to read: around 1 means the model fits
    to within the stated uncertainties.  It needs the residual and free
    parameter counts together, so both are taken from the model rather
    than re-derived at the call site.  ``n_data``, ``dof`` and
    ``reduced_chi2`` are ``None`` for a scalar loss, which exposes no
    residual count; ``result`` and ``steps`` need a Solution.
    """

    chi2: float
    n_data: int | None
    n_free: int
    dof: int | None
    reduced_chi2: float | None
    result: str | None = None
    steps: int | None = None

    def __repr__(self) -> str:
        lines = [f"chi2         {self.chi2:.6g}"]
        if self.n_data is not None:
            lines.append(f"d.o.f.       {self.dof} = {self.n_data} data - {self.n_free} free")
        else:
            lines.append(f"free params  {self.n_free} (scalar loss: no residual count)")
        if self.reduced_chi2 is not None:
            lines.append(f"chi2/d.o.f.  {self.reduced_chi2:.6g}")
        if self.result is not None:
            lines.append(
                f"solver       {self.result}" + (f", {self.steps} steps" if self.steps else "")
            )
        return "FitSummary(\n  " + "\n  ".join(lines) + "\n)"


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
    least-squares solver state cannot be jitted).  Wrap the step with
    :func:`equinox.filter_jit` rather than :func:`jax.jit`: some solver
    states (``optx.LBFGS``) carry non-array leaves::

        opt = Optimizer(loss_fn, optx.BFGS(rtol=1e-5, atol=1e-5))
        state = opt.init_state(scene)
        step = eqx.filter_jit(opt.step)
        for _ in range(200):
            scene, loss, state = step(scene, state)

    Or let :func:`nyx.infer.convergence.record_fit` drive the same loop
    and record the path the fit takes.
    """

    def __init__(self, fn: Callable[[Any], Any], solver: Any) -> None:
        self._solver = solver
        self._is_ls = isinstance(solver, optx.AbstractLeastSquaresSolver)
        self._fn_user = fn
        self._fn_is_scalar: bool | None = None  # resolved on first call to _check_fn
        self._inner: Callable[..., Any] | None = None  # built once; see _make_inner

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
        """Partition *model* and return the inner fn for optimistix."""
        diff, static = eqx.partition(model, _is_trainable, is_leaf=_is_param)
        if self._inner is None:
            fn_user = self._fn_user
            user_fn: Callable[[Any], Any]
            if not self._is_ls and not self._fn_is_scalar:

                def user_fn(m: Any) -> Any:
                    return _sum_of_squares(fn_user(m))
            else:
                user_fn = fn_user

            def inner(diff: Any, args: Any) -> tuple[Any, None]:
                return user_fn(eqx.combine(diff, args, is_leaf=_is_param)), None

            self._inner = inner
        return self._inner, diff, static

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
        inner, diff, static = self._make_inner(model)
        f_struct = jax.eval_shape(lambda: inner(diff, static)[0])
        aux_struct = None
        return self._solver.init(inner, diff, static, {}, f_struct, aux_struct, frozenset())

    def loss(self, model: Any) -> jax.Array:
        """Scalar loss at *model*, without taking a solver step.

        On the least-squares path -- and for a residuals *fn* handed to a
        minimiser -- this is ``sum(r ** 2)``.

        Parameters
        ----------
        model : pytree
            Model to evaluate.

        Returns
        -------
        jax.Array
            Scalar loss.
        """
        if self._fn_is_scalar is None:
            self._check_fn(model)
        if self._fn_is_scalar:
            return self._fn_user(model)
        return _sum_of_squares(self._fn_user(model))

    def step[T](self, model: T, state: Any) -> tuple[T, jax.Array, Any]:
        """One solver step.

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
        loss = self.loss(model)
        inner, diff, static = self._make_inner(model)
        new_diff, new_state, _ = self._solver.step(inner, diff, static, {}, state, frozenset())
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
            args=static,
            has_aux=True,
            max_steps=max_steps,
            throw=throw,
        )
        _warn_non_finite(_non_finite_parameters(sol.value), "the fitted model")
        fitted = eqx.combine(sol.value, static, is_leaf=_is_param)
        return fitted, sol

    def summary(self, model: Any, sol: optx.Solution[Any, Any] | None = None) -> FitSummary:
        """Goodness of fit at *model*::

            fitted, sol = opt.run(scene)
            print(opt.summary(fitted, sol))

        Counting from the model means the reduced chi-squared cannot
        drift from the parameters actually fitted.  *sol* adds the
        solver's status and step count.
        """
        self._check_fn(model)
        chi2 = float(self.loss(model))
        n_free = n_trainable(model)
        n_data: int | None = None
        if not self._fn_is_scalar:
            n_data = int(
                sum(int(np.size(np.asarray(r))) for r in jax.tree.leaves(self._fn_user(model)))
            )
        dof = None if n_data is None else n_data - n_free
        reduced = chi2 / dof if dof is not None and dof > 0 else None
        return FitSummary(
            chi2=chi2,
            n_data=n_data,
            n_free=n_free,
            dof=dof,
            reduced_chi2=reduced,
            result=None if sol is None else _result_text(sol),
            steps=None if sol is None else int(sol.stats.get("num_steps", 0)) or None,
        )

    def profile(
        self,
        model: Any,
        scan: dict[str, Any],
        *,
        max_steps: int = 256,
        callback: Callable[[tuple[int, ...], dict[str, float], float], None] | None = None,
    ) -> ProfileGrid:
        """Profile likelihood over one or two parameters.

        At each grid point the scanned parameters are held and the rest
        refitted, projecting the nuisances out rather than fixing them at
        their global best.  The grid is indexed in the order given --
        ``chi2[i, j]`` is at ``axes[0][i], axes[1][j]``::

            grid = opt.profile(fitted, {'instrument.sigma_x': sx,
                                        'instrument.sigma_y': sy})
            plt.contour(*grid.axes[::-1], grid.delta_chi2.T, levels=grid.levels)

        *model* needs its nuisances already unfrozen.  *callback* is
        ``(index, point, chi2) -> None``, for a progress bar.
        """
        if not 1 <= len(scan) <= 2:
            raise ValueError(f"scan one or two parameters, got {len(scan)}")
        names = list(scan)
        axes = [np.atleast_1d(np.asarray(v, dtype=float)) for v in scan.values()]
        chi2 = np.full([a.size for a in axes], np.nan)

        for index in np.ndindex(*chi2.shape):
            point = {n: a[i] for n, a, i in zip(names, axes, index, strict=True)}
            # Hold the scanned parameters; everything else is refitted.
            held = freeze(set_parameters(model, point), *names)
            fitted, _ = self.run(held, max_steps=max_steps, throw=False)
            chi2[index] = float(self.loss(fitted))
            if callback is not None:
                callback(index, point, chi2[index])

        return ProfileGrid(names=names, axes=axes, chi2=chi2)

    def errors[T](
        self,
        fitted: T,
        *,
        batch_size: int = 8,
        reduced_chi2: bool = False,
        rcond: float | None = None,
    ) -> T:
        """1-σ errors on every trainable Parameter at *fitted*.

        Returns ``sqrt(diag(pinv(JᵀJ)))`` as a pytree with the same
        structure as the trainable subset of *fitted*.

        Only valid when *fn* returns residuals.  Does not require
        :meth:`run` first: errors can be taken at any model.

        Parameters
        ----------
        fitted : pytree
            Model returned by :meth:`run`.
        batch_size : int, optional
            Jacobian columns evaluated per kernel launch.  Larger is not
            better; see :func:`parameter_errors`.
        reduced_chi2 : bool, optional
            If False (default), assumes residuals are pre-scaled by 1-σ
            uncertainties so ``cov = pinv(J^T J)``.  If True, applies
            the ``(r^T r) / (m - n)`` factor so errors reflect the
            spread consistent with the data.
        rcond : float or None, optional
            Cutoff for the pseudoinverse, relative to the largest
            singular value of ``J``.  See :func:`parameter_errors`.

        See :func:`parameter_errors` for the standalone form and full
        docstring.
        """
        self._check_fn(fitted)
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
