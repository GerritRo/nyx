"""Watch a fit converge, one solver step at a time."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator, Mapping
from typing import TYPE_CHECKING, Any

import equinox as eqx
import numpy as np

from nyx import NyxWarning

if TYPE_CHECKING:
    from nyx.infer import Optimizer

__all__ = ["FitTrace", "record_fit"]


class FitTrace:
    """The path a fit took, sampled once per recorded solver step.

    Attributes
    ----------
    steps : numpy.ndarray, shape (n_frames,)
        Solver step index of each frame; ``steps[0]`` is 0.
    loss : numpy.ndarray, shape (n_frames,)
        Scalar loss at each frame, evaluated before that step was applied.
    history : dict of str to numpy.ndarray
        One entry per probe, stacked over frames, each of shape
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
        """Number of recorded frames.

    Returns
    -------
    int
    """
        return int(self.steps.size)

    def __getitem__(self, name: str) -> np.ndarray:
        """Recorded values of one probe, of shape ``(n_frames,) + probe shape``."""
        return self.history[name]

    def __contains__(self, name: object) -> bool:
        return name in self.history

    def __iter__(self) -> Iterator[str]:
        return iter(self.history)

    def keys(self) -> Iterator[str]:
        """Names of the recorded probes.

    Returns
    -------
    KeysView of str
    """
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
    stop_on_diverged: bool = True,
) -> FitTrace:
    """Step *opt* over *model*, recording the loss and the probes.

    Every ``stride``-th step is recorded, always including step 0 and the
    final one.  Each frame holds the model as it entered that step, with the
    loss there.

    Parameters
    ----------
    opt : Optimizer
        Built around a minimiser solver, e.g. ``optx.LBFGS``; a
        least-squares solver raises, its state being untraceable.
    model : pytree
        Starting model, typically a :class:`~nyx.core.scene.Scene`.
    probes : mapping of str to callable, optional
        ``(model) -> array``, evaluated at every recorded frame.  Traced
        together under one :func:`equinox.filter_jit`, so each must return a
        JAX array of fixed shape.
    max_steps : int, optional
        Exact number of solver steps to take, not a ceiling.
    stride : int, optional
        Record every ``stride``-th step.
    jit : bool, optional
        Whether to compile the solver step and the probes.
    callback : callable, optional
        ``(step, loss, model) -> None``, called at every recorded frame.
    stop_on_diverged : bool, optional
        Whether to stop, warn and return early once the loss goes
        non-finite.

    Returns
    -------
    FitTrace
        ``trace.model`` is the fitted model.

    Raises
    ------
    TypeError
        If *opt* uses a least-squares solver.
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
        if stop_on_diverged and not np.isfinite(float(loss)):
            warnings.warn(
                f"the loss went non-finite at step {i}; stopping there. The trace "
                f"holds the path up to that point, which is where to look for the "
                f"parameter that ran away. Pass stop_on_diverged=False to take all "
                f"{max_steps} steps anyway.",
                NyxWarning,
                stacklevel=2,
            )
            return FitTrace(steps, losses, history, model=model)
        model = new_model
    record(max_steps, float(loss_at(model)))

    return FitTrace(steps, losses, history, model=model)
