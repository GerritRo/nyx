"""Watch a fit converge, one solver step at a time.

:meth:`~nyx.core.fitting.Optimizer.run` hands the whole iteration to XLA
and only the endpoint comes back.  :func:`record_fit` drives the same
solver through :meth:`~nyx.core.fitting.Optimizer.step` instead and
evaluates a set of *probes* along the way, so the path the fit takes is
available afterwards -- for a progress plot, a convergence animation, or
just to see which parameter is the slow one::

    trace = record_fit(
        opt,
        scene,
        {
            'shift': lambda s: s.instrument.shift.value,
            'aod_500': lambda s: s.atmosphere.Mie.aod_500.value,
        },
        max_steps=200,
    )
    plt.semilogy(trace.steps, trace.loss)
    plt.plot(trace.steps, trace['aod_500'])

Stepping is only supported for minimiser solvers (``optx.BFGS``,
``optx.LBFGS``, ...); a least-squares solver's state cannot be traced.
A residuals function is fine either way -- it is reduced to
``sum(r ** 2)`` exactly as :meth:`~nyx.core.fitting.Optimizer.run` would.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import TYPE_CHECKING, Any

import equinox as eqx
import numpy as np

if TYPE_CHECKING:
    from nyx.core.fitting import Optimizer

__all__ = ["FitTrace", "record_fit"]


class FitTrace:
    """The path a fit took, sampled once per recorded solver step.

    Returned by :func:`record_fit`; not usually built directly.

    Attributes
    ----------
    steps : np.ndarray, shape (n_frames,)
        Solver step index of each frame.  ``steps[0]`` is 0, the starting
        model, and ``steps[-1]`` the final one.
    loss : np.ndarray, shape (n_frames,)
        Scalar loss at each frame, evaluated *before* that step was
        applied.
    history : dict of {str: np.ndarray}
        One entry per probe, stacked over frames: each has shape
        ``(n_frames,) + probe_output_shape``.
    model : pytree
        The model at the last recorded frame.
    """

    def __init__(
        self,
        steps: Any,
        loss: Any,
        history: Mapping[str, Any],
        model: Any = None,
    ) -> None:
        self.steps = np.asarray(steps, dtype=int)
        self.loss = np.asarray(loss, dtype=float)
        self.history = {name: np.asarray(values) for name, values in history.items()}
        self.model = model

    def __len__(self) -> int:
        """Number of recorded frames."""
        return int(self.steps.size)

    def __getitem__(self, name: str) -> np.ndarray:
        """Recorded values of one probe, shape ``(n_frames,) + probe shape``."""
        return self.history[name]

    def __contains__(self, name: object) -> bool:
        return name in self.history

    def __iter__(self) -> Iterator[str]:
        return iter(self.history)

    def keys(self) -> Iterator[str]:
        """Names of the recorded probes."""
        return iter(self.history)

    def __repr__(self) -> str:
        probes = ", ".join(f"{k}{v.shape[1:]}" for k, v in self.history.items())
        return (
            f"FitTrace({len(self)} frames, steps 0-{int(self.steps[-1]) if len(self) else 0}, "
            f"loss {self.loss[0]:.4g} -> {self.loss[-1]:.4g}"
            f"{', probes: ' + probes if probes else ''})"
        )


def record_fit(
    opt: Optimizer,
    model: Any,
    probes: Mapping[str, Callable[[Any], Any]] | None = None,
    *,
    max_steps: int = 200,
    stride: int = 1,
    jit: bool = True,
    callback: Callable[[int, float, Any], None] | None = None,
) -> FitTrace:
    """Step *opt* over *model*, recording the loss and the probes.

    Every ``stride``-th step is recorded, always including step 0 (the
    starting model) and the final one.  Each frame holds the model *as it
    entered* that step, together with the loss there, so frame 0 is the
    unfitted starting point.

    Parameters
    ----------
    opt : nyx.core.fitting.Optimizer
        Built around a *minimiser* solver, e.g. ``optx.LBFGS``.  A
        least-squares solver raises, because its state cannot be jitted.
    model : pytree
        Starting model, typically a :class:`~nyx.core.scene.Scene`.
    probes : mapping of {str: callable}, optional
        ``(model) -> array`` functions evaluated at every recorded frame.
        They are traced together under one :func:`equinox.filter_jit`, so
        each must return a JAX array of a fixed shape.  A probe that
        re-renders the scene costs one extra render per recorded frame.
    max_steps : int, optional
        Number of solver steps to take (default 200).  Unlike
        :meth:`~nyx.core.fitting.Optimizer.run` this is not a ceiling on
        an early-terminating loop: exactly this many steps are taken.
    stride : int, optional
        Record every ``stride``-th step (default: every one).
    jit : bool, optional
        Compile the solver step and the probes (default True).
    callback : callable, optional
        ``(step, loss, model) -> None``, called at every recorded frame.
        Useful for a progress bar.

    Returns
    -------
    FitTrace
        The recorded path.  ``trace.model`` is the fitted model.

    Examples
    --------
    ::

        opt = Optimizer(residuals_fn, optx.LBFGS(rtol=1e-6, atol=1e-6))
        trace = record_fit(
            opt,
            scene,
            {'image': lambda s: s.render()['instrument'][0]},
            max_steps=150,
            callback=lambda i, loss, _: print(i, loss),
        )
    """
    import optimistix as optx

    if isinstance(opt._solver, optx.AbstractLeastSquaresSolver):
        raise TypeError(
            f"{type(opt._solver).__name__} is a least-squares solver; its state "
            f"cannot be traced, so a fit cannot be stepped and "
            f"recorded.  Use a minimiser (optx.LBFGS, optx.BFGS, ...) with the "
            f"same residuals function instead."
        )
    if max_steps < 0:
        raise ValueError(f"max_steps must be >= 0, got {max_steps}.")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}.")

    probes = dict(probes or {})

    def probe_all(m: Any) -> dict[str, Any]:
        return {name: fn(m) for name, fn in probes.items()}

    step = eqx.filter_jit(opt.step) if jit else opt.step
    evaluate = eqx.filter_jit(probe_all) if (jit and probes) else probe_all
    loss_at = eqx.filter_jit(opt.loss) if jit else opt.loss

    steps: list[int] = []
    losses: list[float] = []
    history: dict[str, list[np.ndarray]] = {name: [] for name in probes}

    def record(i: int, loss: float) -> None:
        steps.append(i)
        losses.append(loss)
        for name, value in evaluate(model).items():
            history[name].append(np.asarray(value))
        if callback is not None:
            callback(i, loss, model)

    state = opt.init_state(model)
    for i in range(max_steps):
        new_model, loss, state = step(model, state)
        if i % stride == 0:
            record(i, float(loss))
        model = new_model
    record(max_steps, float(loss_at(model)))

    return FitTrace(steps, losses, history, model=model)
