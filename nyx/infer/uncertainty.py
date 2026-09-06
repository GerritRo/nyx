"""Parameter uncertainties from the Gauss-Newton covariance."""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.flatten_util as fu
import jax.numpy as jnp
import numpy as np

from nyx import NyxWarning
from nyx.core.parameter import Parameter, _friendly_keypath, _is_param
from nyx.infer._common import (
    _flat_parameter_names,
    _is_trainable,
    _non_finite_parameters,
    _warn_non_finite,
)


def _jacobian_columns(jvp_fn: Callable[[Any], Any], n: int, m: int, chunk: int) -> jax.Array:
    """Build ``J^T`` by pushing basis vectors through a linearised model.

    Forward mode only.

    Parameters
    ----------
    jvp_fn : callable
        Linear map from :func:`jax.linearize`.
    n, m : int
        Parameter and residual counts.
    chunk : int
        Columns evaluated per kernel launch.

    Returns
    -------
    jax.Array, shape (n, m)
    """

    def block(idx: jax.Array) -> jax.Array:
        return jax.vmap(lambda k: jvp_fn(jax.nn.one_hot(k, n)))(idx)

    # Pad the last block to a uniform shape so it compiles once; the
    # duplicated columns are dropped below.
    n_blocks = -(-n // chunk)
    idx = jnp.minimum(jnp.arange(n_blocks * chunk), n - 1).reshape(n_blocks, chunk)
    return jax.lax.map(block, idx).reshape(-1, m)[:n]


#: Bytes to aim for per float64 operand slice of the ``J^T J`` product.
_GRAM_SLICE_BYTES = 64 * 1024 * 1024


def _gram_matrix(jacobian_t: jax.Array) -> np.ndarray:
    """
    ``J^T J`` from ``J^T``, accumulated on the host in float64.
    """
    jt = np.asarray(jacobian_t)  # one device-to-host transfer, still float32
    n, m = jt.shape
    rows = int(np.clip(_GRAM_SLICE_BYTES // (8 * max(m, 1)), 1, n))
    out = np.empty((n, n), dtype=np.float64)
    for i in range(0, n, rows):
        a = jt[i : i + rows].astype(np.float64)
        for j in range(0, i + rows, rows):
            b = a if j == i else jt[j : j + rows].astype(np.float64)
            block = a @ b.T
            out[i : i + rows, j : j + rows] = block
            if j != i:
                out[j : j + rows, i : i + rows] = block.T
    return out


def _null_space_parameters(diff: Any, null_vectors: np.ndarray, top: int = 3) -> str:
    """Name the parameters carrying the discarded directions."""
    if null_vectors.size == 0:
        return "(none)"
    weight = (null_vectors**2).sum(axis=1)
    shares, start = [], 0
    for path, leaf in jax.tree_util.tree_leaves_with_path(diff):
        stop = start + int(np.size(leaf))
        shares.append((float(weight[start:stop].sum()), _friendly_keypath(path)))
        start = stop
    total = sum(s for s, _ in shares) or 1.0
    shares.sort(reverse=True)
    named = [
        f"{name} ({100 * s / total:.0f}%)" if s / total >= 0.01 else f"{name} (<1%)"
        for s, name in shares[:top]
        if s / total > 0.001
    ]
    return ", ".join(named) if named else "(spread over many parameters)"


@dataclasses.dataclass(frozen=True)
class _Decomposition:
    """The eigendecomposition of ``J^T J`` that errors and correlations share."""

    diff: Any
    unravel: Callable[[Any], Any]
    residuals: jax.Array
    eigvals: np.ndarray
    eigvecs: np.ndarray
    keep: np.ndarray
    n: int


def _decompose(
    fitted: Any,
    residuals_fn: Callable[[Any], Any],
    *,
    batch_size: int,
    rcond: float | None,
) -> _Decomposition:
    """Linearise about *fitted* and diagonalise ``J^T J``.

    Shared by :func:`parameter_errors` and :func:`parameter_correlation`,
    which differ only in how much of the inverse they form.
    """
    diff, static = eqx.partition(fitted, _is_trainable, is_leaf=_is_param)
    flat, unravel = fu.ravel_pytree(diff)
    n = flat.size

    def residuals_of_flat(x: jax.Array) -> jax.Array:
        return fu.ravel_pytree(residuals_fn(eqx.combine(unravel(x), static, is_leaf=_is_param)))[0]

    # Linearise once; jvp_fn is a pure linear operator, no retracing.
    r0, jvp_fn = jax.linearize(residuals_of_flat, flat)
    jacobian_t = _jacobian_columns(jvp_fn, n, r0.size, max(int(batch_size), 1))

    # np.linalg.eigh on a NaN Gram matrix reports only "Eigenvalues did not
    # converge", which says nothing about the cause.  Name it here instead.
    finite_rows = np.isfinite(np.asarray(jacobian_t)).all(axis=1)
    if not finite_rows.all():
        names = _flat_parameter_names(diff)
        culprits = sorted({names[i] for i in np.flatnonzero(~finite_rows)})
        raise ValueError(
            f"the Jacobian is non-finite for {int((~finite_rows).sum())} of {n} "
            f"parameter slot(s), in {', '.join(culprits[:5])}"
            + (f" and {len(culprits) - 5} more" if len(culprits) > 5 else "")
            + ". Errors cannot be estimated at a model whose residuals do not "
            "differentiate cleanly -- check that the fit converged before "
            "asking for its uncertainties."
        )

    JtJ = _gram_matrix(jacobian_t)

    # Eigendecomposition in float64 on the host:
    eigvals, eigvecs = np.linalg.eigh(JtJ)
    eigvals = np.clip(eigvals, 0.0, None)
    lam_max = float(eigvals.max()) if eigvals.size else 0.0
    if rcond is None:
        rcond = n * float(np.finfo(jacobian_t.dtype).eps)
    # rcond cuts singular values of J, which are sqrt(eigenvalues of J^T J);
    keep = eigvals > rcond**2 * lam_max

    n_null = int((~keep).sum())
    if n_null > 0:
        lam_min = float(eigvals[keep].min()) if keep.any() else 0.0
        cond = (lam_max / lam_min) ** 0.5 if lam_min > 0 else float("inf")
        culprits = _null_space_parameters(diff, eigvecs[:, ~keep])
        warnings.warn(
            f"J is rank-deficient "
            f"({n_null} of {n} singular value(s) <= rcond * sigma_max, "
            f"rcond = {rcond:.3e}; condition number of the retained block "
            f"{cond:.3e}).  The "
            f"unconstrained direction(s) lie mostly along {culprits}, whose "
            f"returned σ are therefore NOT error bars: they describe the "
            f"identifiable projection only, and are as small as the "
            f"pseudoinverse can make them rather than as large as the true "
            f"(unbounded) uncertainty.  A common cause is the degeneracy "
            f"between a global `efficiency` and the overall scale of "
            f"`pixel_efficiency` (flatfield normalisation).  Freeze one of "
            f"them via `nyx.core.parameter.freeze` to obtain a unique MLE "
            f"and meaningful σ for the rest.",
            NyxWarning,
            stacklevel=3,
        )

    return _Decomposition(
        diff=diff,
        unravel=unravel,
        residuals=r0,
        eigvals=eigvals,
        eigvecs=eigvecs,
        keep=keep,
        n=n,
    )


def parameter_errors[T](
    fitted: T,
    residuals_fn: Callable[[Any], Any],
    *,
    batch_size: int = 8,
    reduced_chi2: bool = False,
    rcond: float | None = None,
) -> T:
    """1-σ errors on every trainable Parameter, from the Gauss-Newton covariance.

    Builds the Jacobian one block of columns at a time, forms ``J^T J``
    in float64, and returns ``sqrt(diag(pinv(J^T J)))`` shaped like the
    trainable subset of *fitted*.

    Parameters
    ----------
    fitted : pytree
        Model to linearise about.
    residuals_fn : callable
        ``(model) -> residuals``, pre-scaled by 1-σ uncertainties.
    batch_size : int, optional
        Jacobian columns per kernel launch.  Larger is not better: past a
        few dozen the per-column intermediates stop fitting in cache and
        throughput collapses.
    reduced_chi2 : bool, optional
        If False (default), assumes residuals are pre-scaled by 1-σ
        uncertainties so ``cov = pinv(J^T J)``.  If True, applies the
        ``(r^T r) / (m - n)`` factor.
    rcond : float or None, optional
        Cutoff for the pseudoinverse, relative to the largest singular
        value of ``J``.  Defaults to ``n * eps`` of the Jacobian's dtype,
        following :func:`numpy.linalg.pinv`: below that a direction is
        indistinguishable from the rounding error in ``J`` itself, so it
        is treated as unidentifiable and given zero error.

    Returns
    -------
    pytree
        1-σ errors, shaped like the trainable subset of *fitted*.
    """
    d = _decompose(fitted, residuals_fn, batch_size=batch_size, rcond=rcond)
    diff, unravel, r0, n = d.diff, d.unravel, d.residuals, d.n
    eigvecs, keep = d.eigvecs, d.keep
    inv = np.where(keep, 1.0 / np.where(keep, d.eigvals, 1.0), 0.0)
    flat_dtype = fu.ravel_pytree(diff)[0].dtype

    # Only the diagonal of the covariance is needed, so contract the
    # eigenvectors directly instead of forming the full n x n inverse.
    variance = (eigvecs**2) @ inv
    if reduced_chi2:
        dof = r0.size - n
        if dof <= 0:
            raise ValueError(
                f"reduced_chi2 needs more residuals than free parameters, got "
                f"{r0.size} residuals for {n} parameters; freeze parameters or "
                f"leave reduced_chi2=False."
            )
        variance = variance * (float(jnp.sum(r0**2)) / dof)

    sigma = unravel(jnp.asarray(np.sqrt(variance), dtype=flat_dtype))

    # `unravel` rebuilds Parameters whose factors are sigmas in the space
    # the optimizer steps in, not physical units.
    def to_physical(fitted_p: Any, sigma_p: Any) -> Any:
        if not (_is_param(fitted_p) and _is_param(sigma_p)):
            return sigma_p
        return Parameter.from_value(
            jnp.abs(fitted_p.dvalue_dfactor) * sigma_p.factor,
            scale=fitted_p.scale,
            per_obs=fitted_p.per_obs,
            frozen=fitted_p.frozen,
            transform=fitted_p.transform,
        )

    return jax.tree.map(to_physical, diff, sigma, is_leaf=_is_param)


@dataclasses.dataclass(frozen=True)
class ParameterCorrelation:
    """How a fit's parameters trade off against one another.

    A degeneracy is the usual reason a fit wanders or returns implausible
    error bars, and the covariance is where it shows.  ``names`` has one
    entry per free scalar in the optimizer's flattening order, so array
    parameters repeat; ``sigma`` is in that same stepping space.
    ``n_unconstrained`` counts directions the pseudoinverse discarded --
    non-zero means the data does not determine some combination at all.
    """

    names: list[str]
    covariance: np.ndarray
    correlation: np.ndarray
    sigma: np.ndarray
    n_unconstrained: int

    def worst_pairs(
        self, count: int = 10, *, threshold: float = 0.0
    ) -> list[tuple[str, str, float]]:
        """The *count* most correlated pairs of distinct parameters.

        Per parameter, not per scalar: for an array parameter the
        strongest correlation in the block is the one that matters, and a
        thousand rows naming the same pair would bury it.  *threshold*
        drops anything weaker.
        """
        unique = list(dict.fromkeys(self.names))
        index = {name: np.flatnonzero(np.asarray(self.names) == name) for name in unique}
        pairs = []
        for i, a in enumerate(unique):
            for b in unique[i + 1 :]:
                block = self.correlation[np.ix_(index[a], index[b])]
                flat = block.ravel()
                worst = flat[np.argmax(np.abs(flat))]
                if abs(worst) >= threshold:
                    pairs.append((a, b, float(worst)))
        pairs.sort(key=lambda p: -abs(p[2]))
        return pairs[:count]

    def __repr__(self) -> str:
        lines = [
            f"ParameterCorrelation({len(dict.fromkeys(self.names))} parameters, "
            f"{len(self.names)} free values"
            + (
                f", {self.n_unconstrained} unconstrained direction(s)"
                if self.n_unconstrained
                else ""
            )
            + ")"
        ]
        worst = self.worst_pairs(5)
        if worst:
            lines.append("  most correlated:")
            width = max(len(a) for a, _, _ in worst)
            lines += [f"    {a:<{width}}  {b:<{width}}  {c:+.4f}" for a, b, c in worst]
        return "\n".join(lines)


def parameter_correlation(
    fitted: Any,
    residuals_fn: Callable[[Any], Any],
    *,
    batch_size: int = 8,
    rcond: float | None = None,
) -> ParameterCorrelation:
    """Full covariance and correlation of the free parameters at *fitted*::

        corr = parameter_correlation(fitted, residuals_fn)
        corr.worst_pairs(threshold=0.9)  # anything nearly degenerate

    :func:`parameter_errors` returns only the diagonal, which is all an
    error bar needs; this forms the whole matrix, which shows *why* one is
    large, and costs an ``n x n`` array to do it.  *residuals_fn*,
    *batch_size* and *rcond* are as there.
    """
    d = _decompose(fitted, residuals_fn, batch_size=batch_size, rcond=rcond)
    inv = np.where(d.keep, 1.0 / np.where(d.keep, d.eigvals, 1.0), 0.0)
    covariance = (d.eigvecs * inv) @ d.eigvecs.T
    sigma = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    safe = np.where(sigma > 0, sigma, 1.0)
    correlation = np.clip(covariance / np.outer(safe, safe), -1.0, 1.0)
    return ParameterCorrelation(
        names=_flat_parameter_names(d.diff),
        covariance=covariance,
        correlation=correlation,
        sigma=sigma,
        n_unconstrained=int((~d.keep).sum()),
    )


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
    _warn_non_finite(_non_finite_parameters(errs), "the error estimate")

    def rescale_leaf(param: Any, err: Any) -> Any:
        if not (_is_param(param) and _is_param(err)):
            return param
        if param.transform is not None:
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
