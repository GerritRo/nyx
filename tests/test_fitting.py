"""Optimizer plumbing: solver reuse and manual stepping."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import optimistix as optx

from nyx.core.fitting import Optimizer
from nyx.core.parameter import Parameter, freeze


class Model(eqx.Module):
    """Toy model: a scale and an offset, one of them freezable."""

    scale: Parameter
    offset: Parameter
    data: jnp.ndarray

    def predict(self):
        return self.scale.value * self.data + self.offset.value


def make_model():
    x = jnp.linspace(0.0, 1.0, 32)
    return Model(
        scale=Parameter.from_value(1.0),
        offset=Parameter.from_value(0.0),
        data=x,
    )


def residuals(model):
    truth = 2.5 * model.data - 0.75
    return model.predict() - truth


def test_run_recovers_parameters():
    opt = Optimizer(residuals, optx.LevenbergMarquardt(rtol=1e-8, atol=1e-8))
    fitted, sol = opt.run(make_model(), max_steps=64)
    assert np.isclose(float(fitted.scale.value), 2.5, atol=1e-4)
    assert np.isclose(float(fitted.offset.value), -0.75, atol=1e-4)


def test_run_reuses_the_compiled_solver():
    """The inner fn handed to optimistix must be stable across calls.

    optimistix jits on the identity of that callable, so rebuilding it
    per call would recompile the entire solver every time.
    """
    opt = Optimizer(residuals, optx.LevenbergMarquardt(rtol=1e-8, atol=1e-8))
    model = make_model()
    opt._check_fn(model)
    first, _, _ = opt._make_inner(model)
    second, _, _ = opt._make_inner(model)
    assert first is second


def test_frozen_model_travels_as_args_not_closure():
    """Changing frozen leaves must not change the traced function."""
    opt = Optimizer(residuals, optx.LevenbergMarquardt(rtol=1e-8, atol=1e-8))
    model = make_model()
    opt._check_fn(model)
    inner_a, _, static_a = opt._make_inner(model)
    other = eqx.tree_at(lambda m: m.data, model, model.data * 2.0)
    inner_b, _, static_b = opt._make_inner(other)
    assert inner_a is inner_b
    # The frozen data must come back out in `static`, so it can be passed
    # through as `args` rather than baked into the jaxpr.
    assert not np.allclose(np.asarray(static_a.data), np.asarray(static_b.data))


def test_manual_stepping_reduces_the_loss():
    opt = Optimizer(residuals, optx.BFGS(rtol=1e-8, atol=1e-8))
    model = make_model()
    state = opt.init_state(model)
    losses = []
    for _ in range(64):
        model, loss, state = opt.step(model, state)
        losses.append(float(loss))
    assert losses[-1] < losses[0]
    assert np.isclose(float(model.scale.value), 2.5, atol=1e-2)


def test_frozen_parameters_are_not_trained():
    model = freeze(make_model(), lambda m: m.offset)
    opt = Optimizer(residuals, optx.LevenbergMarquardt(rtol=1e-8, atol=1e-8))
    fitted, _ = opt.run(model, max_steps=64, throw=False)
    assert float(fitted.offset.value) == 0.0
    assert float(fitted.scale.value) != 1.0
