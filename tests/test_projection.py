"""The lattice projection must agree with the direct one it replaced.

`project_direct` -- evaluating every pixel's tabulated response at every
source position -- is no longer part of nyx, but it remains the definition
of what the projection computes, so it lives on here as the reference
implementation these tests check against.
"""

import astropy.units as u
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import simpson

from nyx import ASSETS_PATH
from nyx.core.geometry import Geometry
from nyx.instrument import EffectiveApertureInstrument
from nyx.instrument._interpolation import PixelLattice as Lattice
from nyx.instrument._interpolation import (
    _bilinear_coeffs,
    integrate_response,
    interpolate_pixel_rates,
    project_lattice,
)

INSTRUMENT = ASSETS_PATH + "HESS_CT1.h5"


def project_direct(grid, values_stack, coords, rates):
    """Reference: bilinear response of every pixel at every source."""

    def one(args):
        grid_item, values = args
        height, width = values.shape
        ystart, ystep = grid_item[0, 0], grid_item[0, 1] - grid_item[0, 0]
        xstart, xstep = grid_item[1, 0], grid_item[1, 1] - grid_item[1, 0]
        y = (coords[:, 0] - ystart) / ystep
        x = (coords[:, 1] - xstart) / xstep
        y0, x0, fy, fx, valid = _bilinear_coeffs(y, x, height, width)
        interpolated = (
            (1 - fx) * (1 - fy) * values[y0, x0]
            + fx * (1 - fy) * values[y0, x0 + 1]
            + (1 - fx) * fy * values[y0 + 1, x0]
            + fx * fy * values[y0 + 1, x0 + 1]
        )
        return jnp.sum(jnp.where(valid, interpolated * rates, 0.0))

    return jax.lax.map(one, (grid, values_stack), batch_size=grid.shape[0])


@pytest.fixture(scope="module")
def geo():
    return Geometry(wvls=jnp.linspace(300, 700, 8) * u.nm, nside=4, ngrid=2, fov=3.5 * u.deg)


@pytest.fixture(scope="module")
def instrument(geo):
    return EffectiveApertureInstrument.load(INSTRUMENT, geo)


@pytest.fixture(scope="module")
def sources(instrument):
    rng = np.random.default_rng(0)
    fov = float(jnp.max(jnp.abs(instrument.centers)))
    # Deliberately wider than the camera, so out-of-field sources are covered.
    coords = jnp.asarray(rng.uniform(-2 * fov, 2 * fov, (2000, 2)), dtype=jnp.float32)
    rates = jnp.asarray(rng.lognormal(0.0, 1.0, 2000), dtype=jnp.float32)
    return coords, rates


def test_lattice_matches_direct_projection(instrument, sources):
    coords, rates = sources
    fast = jax.jit(project_lattice)(instrument.lattice, instrument.pixel_values, coords, rates)
    direct = jax.jit(project_direct)(instrument.grid, instrument.pixel_values, coords, rates)
    assert np.allclose(fast, direct, rtol=2e-3, atol=1e-3 * float(jnp.max(direct)))


def test_lattice_matches_direct_gradients(instrument, sources):
    coords, rates = sources
    weight = jnp.asarray(np.random.default_rng(1).normal(size=instrument.centers.shape[0]))

    def loss(project, c, r):
        return jnp.sum(weight * project(c, r))

    fast = jax.grad(loss, argnums=(1, 2))(
        lambda c, r: project_lattice(instrument.lattice, instrument.pixel_values, c, r),
        coords,
        rates,
    )
    direct = jax.grad(loss, argnums=(1, 2))(
        lambda c, r: project_direct(instrument.grid, instrument.pixel_values, c, r),
        coords,
        rates,
    )
    for a, b in zip(fast, direct, strict=True):
        scale = float(jnp.max(jnp.abs(b)))
        assert np.allclose(a, b, rtol=1e-2, atol=5e-3 * scale)


# Lattice geometry


def test_grid_and_centers_round_trip(instrument):
    """The reconstructed coordinate table must match what a 1.x file stored."""
    lattice = instrument.lattice
    grid = np.asarray(lattice.grid)
    step = np.asarray(lattice.step)
    # Nodes are exactly integer multiples of the step from the origin.
    nodes = (grid - np.asarray(lattice.origin)[None, :, None]) / step[None, :, None]
    assert np.abs(nodes - np.round(nodes)).max() < 1e-4
    assert np.allclose(np.asarray(lattice.centers), grid.mean(axis=2), atol=1e-7)


def test_lattice_rejects_unaligned_grids(instrument):
    """Grids that are not windows onto one lattice must be rejected."""
    grid = np.asarray(instrument.grid, dtype=np.float64)
    values = np.asarray(instrument.pixel_values)
    step = grid[0, 0, 1] - grid[0, 0, 0]
    shifted = grid.copy()
    shifted[::2, 0, :] += 0.37 * step  # half the pixels off-lattice
    with pytest.raises(ValueError, match="miss a common lattice"):
        Lattice.from_grid(shifted, values)


def test_lattice_rejects_nonzero_boundary(instrument):
    """A response that does not vanish at its edge is cut off, not tabulated."""
    values = np.asarray(instrument.pixel_values).copy()
    values[3, 0, 5] = 1.0
    with pytest.raises(ValueError, match="vanish on their boundary"):
        Lattice.from_grid(np.asarray(instrument.grid), values)


def test_lattice_rejects_mismatched_shapes(instrument):
    values = np.asarray(instrument.pixel_values)[:, :-1, :]
    with pytest.raises(ValueError, match="square"):
        Lattice.from_grid(np.asarray(instrument.grid), values)


# Pixel weights


def test_integrate_response_matches_scipy(instrument):
    """The fixed weight vector must reproduce nested Simpson quadrature."""
    grid = np.asarray(instrument.grid, dtype=np.float64)
    values = np.asarray(instrument.pixel_values, dtype=np.float64)
    reference = np.array(
        [simpson(simpson(values[i], x=grid[i][1]), x=grid[i][0]) for i in range(len(values))]
    )
    got = np.asarray(integrate_response(instrument.lattice, instrument.pixel_values))
    assert np.allclose(got, reference, rtol=1e-5)
    assert np.allclose(got, np.asarray(instrument.weight))


# Misalignment instrument


@pytest.fixture(scope="module")
def misaligned(geo, instrument):
    """A small misalignment instrument built from a slice of the camera."""
    from nyx.instrument.effective_aperture import EffectiveApertureMisalignmentInstrument

    n_pix = 64  # a slice keeps the test quick
    values = np.asarray(instrument.pixel_values)[:n_pix]
    lattice = Lattice(
        origin=instrument.lattice.origin,
        step=instrument.lattice.step,
        offset=np.asarray(instrument.lattice.offset)[:n_pix],
        grid_shape=instrument.lattice.grid_shape,
    )
    table = np.stack([np.stack([values, 0.9 * values]), np.stack([1.1 * values, values])])
    return EffectiveApertureMisalignmentInstrument(
        geo=geo,
        bandpass=lambda wvl: np.ones_like(np.asarray(wvl, dtype=float)),
        grid=lattice,
        all_values=table,
        sigma_x_coords=np.array([0.0, 1.0]),
        sigma_y_coords=np.array([0.0, 1.0]),
        sigma_x_init=0.25,
        sigma_y_init=0.5,
    )


def test_misalignment_weight_matches_interpolated_table(misaligned):
    """Integrating the blended response equals blending pre-integrated weights."""
    table = np.asarray(misaligned.all_pixel_values)
    per_corner = np.stack(
        [
            np.stack(
                [np.asarray(integrate_response(misaligned.lattice, table[i, j])) for j in (0, 1)]
            )
            for i in (0, 1)
        ]
    )
    blended = misaligned._interp(jnp.asarray(per_corner))
    assert np.allclose(np.asarray(misaligned.weight), np.asarray(blended), rtol=1e-6)


def test_misalignment_projection_matches_direct(misaligned):
    rng = np.random.default_rng(3)
    scale = float(jnp.max(jnp.abs(misaligned.centers)))
    coords = jnp.asarray(rng.uniform(-scale, scale, (500, 2)), dtype=jnp.float32)
    rates = jnp.asarray(rng.lognormal(0.0, 1.0, 500), dtype=jnp.float32)
    got = eqx.filter_jit(lambda m, c, r: m.project_catalog(c, r))(misaligned, coords, rates)
    want = project_direct(misaligned.grid, misaligned.pixel_values, coords, rates)
    want = want * misaligned.pixel_efficiency.value
    assert np.allclose(got, want, rtol=2e-3, atol=1e-3 * float(jnp.max(want)))


# FOV eval grid


def test_interpolate_pixel_rates_matches_four_corner():
    """The separable form must reproduce plain bilinear interpolation."""
    rng = np.random.default_rng(2)
    grid = jnp.asarray(np.linspace(-0.06, 0.06, 5))
    values = jnp.asarray(rng.normal(size=(5, 5)), dtype=jnp.float32)
    coords = jnp.asarray(rng.uniform(-0.09, 0.09, (500, 2)), dtype=jnp.float32)

    def reference(Xi, Yi, values, coords):
        height, width = values.shape
        y = (coords[:, 0] - Yi[0]) / (Yi[1] - Yi[0])
        x = (coords[:, 1] - Xi[0]) / (Xi[1] - Xi[0])
        y0, x0, fy, fx, valid = _bilinear_coeffs(y, x, height, width)
        interpolated = (
            (1 - fx) * (1 - fy) * values[y0, x0]
            + fx * (1 - fy) * values[y0, x0 + 1]
            + (1 - fx) * fy * values[y0 + 1, x0]
            + fx * fy * values[y0 + 1, x0 + 1]
        )
        return jnp.where(valid, interpolated, 0.0)

    got = interpolate_pixel_rates(grid, grid, values, coords)
    want = reference(grid, grid, values, coords)
    assert np.allclose(got, want, atol=1e-5)
