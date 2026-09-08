from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import numpy as np
import optimistix as optx

from nyx.core.parameter import is_parameter
from nyx.core.paramtree import freeze, n_trainable, set_parameters
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

    Returns
    -------
    str
    """
    message = str(optx.RESULTS[sol.result]).strip()
    return message or "converged"


@dataclasses.dataclass(frozen=True)
class FitSummary:
    """How a fit came out.

    ``reduced_chi2`` near 1 means the model fits to within the stated
    uncertainties.  It, ``n_data`` and ``dof`` are ``None`` for a scalar
    loss, which exposes no residual count; ``result`` and ``steps`` need a
    Solution.
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

    Trains every non-frozen :class:`~nyx.core.parameter.Parameter` reachable
    from *model*. Per-obs parameters get independent per-observation
    gradients, global ones the sum over observations.

    Parameters
    ----------
    fn : callable
        ``(model) -> scalar`` for a minimiser, or ``(model) -> residuals``
        for a least-squares solver. A non-scalar *fn* given to a minimiser
        is wrapped with sum-of-squares.
    solver : optimistix.AbstractIterativeSolver
        E.g. ``optx.BFGS`` or ``optx.LevenbergMarquardt``; the latter is
        usually fastest for a chi-squared loss.

    Raises
    ------
    TypeError
        If a scalar *fn* is paired with a least-squares solver.
    """

    def __init__(self, fn: Callable[[Any], Any], solver: Any) -> None:
        self._solver = solver
        self._is_ls = isinstance(solver, optx.AbstractLeastSquaresSolver)
        self._fn_user = fn
        self._fn_is_scalar: bool | None = None  # resolved on first call to _check_fn
        self._inner: Callable[..., Any] | None = None  # built once; see _make_inner

    def _check_fn(self, model: Any) -> None:
        """Probe *fn*'s output shape and cache ``_fn_is_scalar``.

    Raises
    ------
    TypeError
        If a scalar *fn* is paired with a least-squares solver.
    """
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
        diff, static = eqx.partition(model, _is_trainable, is_leaf=is_parameter)
        if self._inner is None:
            fn_user = self._fn_user
            user_fn: Callable[[Any], Any]
            if not self._is_ls and not self._fn_is_scalar:

                def user_fn(m: Any) -> Any:
                    return _sum_of_squares(fn_user(m))
            else:
                user_fn = fn_user

            def inner(diff: Any, args: Any) -> tuple[Any, None]:
                return user_fn(eqx.combine(diff, args, is_leaf=is_parameter)), None

            self._inner = inner
        return self._inner, diff, static

    def init_state(self, model: Any) -> Any:
        """Initial solver state for manual stepping.

        Parameters
        ----------
        model : pytree

        Returns
        -------
        optimistix solver state
            Pass as *state* to the first :meth:`step` call.
        """
        self._check_fn(model)
        inner, diff, static = self._make_inner(model)
        f_struct = jax.eval_shape(lambda: inner(diff, static)[0])
        aux_struct = None
        return self._solver.init(inner, diff, static, {}, f_struct, aux_struct, frozenset())

    def loss(self, model: Any) -> jax.Array:
        """Scalar loss at *model*, without taking a solver step.

        For a residuals *fn* this is ``sum(r ** 2)``.

        Parameters
        ----------
        model : pytree

        Returns
        -------
        jax.Array
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
        state : optimistix solver state
            From :meth:`init_state` or a previous :meth:`step`.

        Returns
        -------
        model : pytree
            Updated model.
        loss : jax.Array
            Scalar loss at *model* before the step was applied.
        state : optimistix solver state
        """
        loss = self.loss(model)
        inner, diff, static = self._make_inner(model)
        new_diff, new_state, _ = self._solver.step(inner, diff, static, {}, state, frozenset())
        new_model = eqx.combine(new_diff, static, is_leaf=is_parameter)
        return new_model, loss, new_state

    def run[T](
        self, model: T, *, max_steps: int = 256, throw: bool = True
    ) -> tuple[T, optx.Solution[Any, Any]]:
        """Run the solver to convergence, inside a single compiled loop.

        Parameters
        ----------
        model : pytree
        max_steps : int, optional
            Maximum solver iterations.
        throw : bool, optional
            Whether to raise on non-successful termination; False leaves
            ``sol.result`` to inspect.

        Returns
        -------
        model : pytree
            Fitted model.
        sol : optimistix.Solution
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
        fitted = eqx.combine(sol.value, static, is_leaf=is_parameter)
        return fitted, sol

    def summary(self, model: Any, sol: optx.Solution[Any, Any] | None = None) -> FitSummary:
        """Goodness of fit at *model*.

        Parameters
        ----------
        model : pytree
        sol : optimistix.Solution, optional
            Adds the solver's status and step count.

        Returns
        -------
        FitSummary
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
        refitted.  The grid is indexed in the order given, so ``chi2[i, j]``
        is at ``axes[0][i], axes[1][j]``.

        Parameters
        ----------
        model : pytree
            With its nuisance parameters already unfrozen.
        scan : dict of str to array-like
            One or two parameter names, each with the values to scan.
        max_steps : int, optional
            Maximum solver iterations per refit.
        callback : callable, optional
            ``(index, point, chi2) -> None``, for a progress bar.

        Returns
        -------
        ProfileGrid
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
        """1-sigma errors on every trainable Parameter at *fitted*.

        Valid only when *fn* returns residuals; :meth:`run` need not have
        been called.

        Parameters
        ----------
        fitted : pytree
        batch_size : int, optional
            Jacobian columns evaluated per kernel launch.
        reduced_chi2 : bool, optional
            Whether to apply the ``(r^T r) / (m - n)`` factor.  False assumes
            residuals pre-scaled by 1-sigma uncertainties.
        rcond : float or None, optional
            Pseudoinverse cutoff, relative to the largest singular value.

        Returns
        -------
        pytree
            ``sqrt(diag(pinv(J^T J)))``, shaped like the trainable subset of
            *fitted*.
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
