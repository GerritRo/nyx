"""Sampling geometry of the focal plane.

One convention holds for every angular grid in nyx, and both functions
here rely on it:

- a coordinate pair is ``[lon, lat]`` -- offset-frame longitude first,
  matching ``(az, alt)`` everywhere else;
- a 2-D grid is **lon-major**: axis 0 runs along longitude, axis 1 along
  latitude, so ``values[i, j]`` sits at ``(lon[i], lat[j])`` and axis *k*
  is indexed by column *k* of the coordinate pair.

That is the layout the ray tracer tabulates a pixel response on, and it
is fixed by the on-disk instrument format (``lattice/origin``,
``lattice/step`` and ``lattice/offset`` are all stored lon-first), so
:class:`~nyx.core.geometry.Geometry` builds its FOV evaluation grid the
same way rather than the other way round.  Anything written against
:attr:`~nyx.core.protocols.AtmosphereResult.scattering_map` or
:attr:`~nyx.core.observation.SkyGeometry.fov_altaz_grid` follows it too.
"""

from functools import cache

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import simpson

LATTICE_TOL = 1e-3


def _bilinear_coeffs(y_coords, x_coords, height, width):
    """Corner indices, fractional offsets and in-bounds mask for bilinear lookup."""
    valid_mask = (
        (y_coords >= 0) & (y_coords < height - 1) & (x_coords >= 0) & (x_coords < width - 1)
    )
    y0 = jnp.clip(jnp.floor(y_coords).astype(jnp.int32), 0, height - 2)
    x0 = jnp.clip(jnp.floor(x_coords).astype(jnp.int32), 0, width - 2)
    fy = y_coords - y0
    fx = x_coords - x0
    return y0, x0, fy, fx, valid_mask


# Pixel response lattice


class PixelLattice(eqx.Module):
    """
    The sampling lattice every pixel's response grid is a window onto.

    Parameters
    ----------
    origin : array-like, shape (2,)
        Position of lattice node ``(0, 0)`` as ``[lon, lat]``, in radians.
    step : array-like, shape (2,)
        Node spacing along ``[lon, lat]``, in radians.
    offset : array-like, shape (n_pixels, 2)
        Node index of each pixel's response window corner, ``[lon, lat]``.
    grid_shape : tuple of int
        Response table shape ``(height, width)`` of a single pixel, with
        ``height`` running along longitude and ``width`` along latitude.
    """

    origin: jax.Array
    step: jax.Array
    offset: jax.Array
    grid_shape: tuple[int, int] = eqx.field(static=True)
    shape: tuple[int, int] = eqx.field(static=True)

    def __init__(self, origin, step, offset, grid_shape):
        self.origin = jnp.asarray(origin, dtype=jnp.float32)
        self.step = jnp.asarray(step, dtype=jnp.float32)
        self.offset = jnp.asarray(offset, dtype=jnp.int32)
        self.grid_shape = (int(grid_shape[0]), int(grid_shape[1]))

        corner = np.asarray(offset)
        if corner.ndim != 2 or corner.shape[1] != 2:
            raise ValueError(f"offset must have shape (n_pixels, 2), got {corner.shape}")
        if np.any(corner < 0):
            raise ValueError(f"offsets must be non-negative, got {corner.min()}")
        self.shape = (
            int(corner[:, 0].max()) + self.grid_shape[0],
            int(corner[:, 1].max()) + self.grid_shape[1],
        )

    @property
    def n_pixels(self) -> int:
        """Number of pixels indexed."""
        return self.offset.shape[0]

    @property
    def window_centers(self) -> jax.Array:
        """Centre of each pixel's response window, radians. Shape (n_pixels, 2)."""
        middle = (jnp.asarray(self.grid_shape, dtype=self.step.dtype) - 1.0) / 2.0
        return self.origin + (self.offset + middle) * self.step

    @property
    def grid(self) -> jax.Array:
        """Per-pixel response sample coordinates. Shape (n_pixels, 2, grid_dim).

        The explicit form :meth:`from_grid` consumes, for inspection and
        for comparing against a ray tracer's own coordinates.
        """
        if self.grid_shape[0] != self.grid_shape[1]:
            raise ValueError(
                f"grid carries one sample count for both axes, so it cannot represent "
                f"a {self.grid_shape[0]}x{self.grid_shape[1]} window"
            )
        nodes = jnp.arange(self.grid_shape[0], dtype=self.step.dtype)
        return self.origin[None, :, None] + (
            (self.offset[:, :, None] + nodes[None, None, :]) * self.step[None, :, None]
        )

    @classmethod
    def from_grid(cls, grid, values, tol: float = LATTICE_TOL) -> "PixelLattice":
        """Recover the lattice underlying explicit sample coordinates.

        ``grid`` is ``(n_pixels, 2, grid_dim)`` in radians and ``values``
        ``(..., n_pixels, height, width)``, every leading axis (e.g. a
        misalignment table) checked.  ``tol`` is in units of one step.

        Raises ``ValueError`` unless the responses vanish on their boundary
        rows and columns and the sample coordinates sit on the nodes of a
        common lattice: both make the projection separable, which is what
        :func:`project_lattice` relies on.
        """
        grid = np.asarray(grid, dtype=np.float64)
        values = np.asarray(values)
        height, width = int(values.shape[-2]), int(values.shape[-1])
        if grid.ndim != 3 or grid.shape[1] != 2:
            raise ValueError(f"grid must have shape (n_pixels, 2, grid_dim), got {grid.shape}")
        if grid.shape[2] != height or height != width:
            raise ValueError(
                f"grid gives {grid.shape[2]} samples per axis for a {height}x{width} "
                f"response; the response must be square"
            )
        if not np.all(np.isfinite(grid)):
            raise ValueError("grid contains non-finite sample coordinates")

        edge = max(
            np.abs(values[..., 0, :]).max(),
            np.abs(values[..., -1, :]).max(),
            np.abs(values[..., :, 0]).max(),
            np.abs(values[..., :, -1]).max(),
        )
        if edge != 0:
            raise ValueError(
                f"responses must vanish on their boundary rows and columns (largest "
                f"edge value {edge:.3e}); pad the tables with a zero border"
            )

        origin = np.empty(2)
        step = np.empty(2)
        offset = np.empty((grid.shape[0], 2), dtype=np.int64)
        for axis in (0, 1):
            coords = grid[:, axis, :]
            o, s, residual = _fit_lattice_axis(coords)
            if s == 0:
                raise ValueError(f"grid has a zero step along axis {axis}")
            if residual > tol:
                raise ValueError(
                    f"response grids miss a common lattice by {residual:.2e} of a step "
                    f"on axis {axis} (tolerance {tol:.0e}); re-tabulate them on a "
                    f"shared grid"
                )
            origin[axis], step[axis] = o, s
            offset[:, axis] = np.round((coords[:, 0] - o) / s)

        return cls(origin=origin, step=step, offset=offset, grid_shape=(height, width))


def _fit_lattice_axis(coords_1d, n_iter: int = 8):
    """Least-squares ``(origin, step, residual)`` of the lattice under *coords_1d*.

    ``coords_1d`` is ``(n_pixels, grid_dim)``, one axis of every pixel's
    response grid; ``residual`` is the largest deviation of any sample from
    a node, in steps.
    """
    flat = coords_1d.reshape(-1)
    step = float(np.median(np.diff(coords_1d, axis=1)))
    if step == 0 or not np.isfinite(step):
        return 0.0, 0.0, np.inf
    origin = float(flat.min())
    node = np.round((flat - origin) / step)
    for _ in range(n_iter):
        design = np.stack([np.ones_like(node), node], axis=1)
        origin, step = np.linalg.lstsq(design, flat, rcond=None)[0]
        new_node = np.round((flat - origin) / step)
        if np.array_equal(new_node, node):
            break
        node = new_node
    # Re-base so the lowest sample sits on node 0.
    origin = float(origin + step * node.min())
    node = node - node.min()
    residual = float(np.max(np.abs((flat - origin) / step - node)))
    return origin, float(step), residual


# Point-source projection


def project_lattice(lattice: PixelLattice, values, coords, rates):
    """Project point sources onto pixels via the shared response lattice.

    ``values`` is ``(n_pixels, height, width)``, ``coords`` ``(n_sources,
    2)`` in the detector frame and ``rates`` the ``(n_sources,)``
    band-integrated rates.  Returns ``(n_pixels,)``.
    """
    n_rows, n_cols = lattice.shape
    grid_h, grid_w = lattice.grid_shape

    # Rasterise the sources onto the lattice.
    pos = (coords - lattice.origin) / lattice.step
    y0, x0, fy, fx, valid = _bilinear_coeffs(pos[:, 0], pos[:, 1], n_rows, n_cols)
    masked = jnp.where(valid, rates, 0.0)

    corner = y0 * n_cols + x0
    idx = jnp.stack([corner, corner + 1, corner + n_cols, corner + n_cols + 1], axis=-1)
    weights = (
        jnp.stack([(1 - fy) * (1 - fx), (1 - fy) * fx, fy * (1 - fx), fy * fx], axis=-1)
        * masked[:, None]
    )
    raster = (
        jnp.zeros(n_rows * n_cols, dtype=weights.dtype).at[idx.reshape(-1)].add(weights.reshape(-1))
    )

    # Integrate each pixel's window of the raster against its response.
    window = (jnp.arange(grid_h)[:, None] * n_cols + jnp.arange(grid_w)[None, :]).astype(jnp.int32)
    base = lattice.offset[:, 0] * n_cols + lattice.offset[:, 1]
    patch = raster[base[:, None, None] + window[None]]
    return jnp.sum(values * patch, axis=(1, 2))


@cache
def _simpson_weights(n: int) -> np.ndarray:
    """Quadrature weights reproducing ``scipy`` Simpson on a unit-spaced axis."""
    return simpson(np.eye(n), dx=1.0, axis=-1)


def integrate_response(lattice: PixelLattice, values):
    """Simpson-integrate each pixel's response ``(..., n_pixels, height, width)``.

    Returns a weight per pixel, shape ``values.shape[:-2]``.
    """
    height, width = lattice.grid_shape
    if values.shape[-2:] != (height, width):
        raise ValueError(f"response is {values.shape[-2:]}, lattice window is {(height, width)}")
    wy = jnp.asarray(_simpson_weights(height)) * lattice.step[0]
    wx = jnp.asarray(_simpson_weights(width)) * lattice.step[1]
    return jnp.einsum("...jk,j,k->...", values, wy, wx)


def response_centroid(lattice: PixelLattice, values):
    """Where each pixel looks: the first moment of its response.

    ``values`` is ``(..., n_pixels, height, width)``; the result is
    ``values.shape[:-2] + (2,)``, a field offset ``[lon, lat]`` in radians.
    """
    height, width = lattice.grid_shape
    if values.shape[-2:] != (height, width):
        raise ValueError(f"response is {values.shape[-2:]}, lattice window is {(height, width)}")
    total = jnp.sum(values, axis=(-2, -1))
    rows = jnp.arange(height, dtype=lattice.step.dtype)
    cols = jnp.arange(width, dtype=lattice.step.dtype)
    node = jnp.stack(
        [jnp.einsum("...jk,j->...", values, rows), jnp.einsum("...jk,k->...", values, cols)],
        axis=-1,
    )
    # A dark pixel has no first moment; fall back to the window centre.
    middle = (jnp.asarray(lattice.grid_shape, dtype=lattice.step.dtype) - 1.0) / 2.0
    safe = jnp.where(total > 0, total, 1.0)[..., None]
    node = jnp.where((total > 0)[..., None], node / safe, middle)
    return lattice.origin + (lattice.offset + node) * lattice.step


# Regular-grid interpolation


def interpolate_regular_grid(x, y, x0, x_step, nx, y0, y_step, ny, data):
    """Bilinear interpolation on a regular 2-D grid, clamped at the edges.

    Samples ``data``, whose leading dims are ``(nx, ny, ...)``, at the
    query point ``(x, y)``; the grid is given by its origin, step and count
    along each axis.  Returns shape ``data.shape[2:]``.
    """
    # Fractional indices, clamped to valid range
    fx_raw = jnp.clip((x - x0) / x_step, 0.0, nx - 1.0)
    fy_raw = jnp.clip((y - y0) / y_step, 0.0, ny - 1.0)

    ix0 = jnp.clip(jnp.floor(fx_raw).astype(jnp.int32), 0, max(nx - 2, 0))
    iy0 = jnp.clip(jnp.floor(fy_raw).astype(jnp.int32), 0, max(ny - 2, 0))
    fx = fx_raw - ix0
    fy = fy_raw - iy0

    ix1 = jnp.minimum(ix0 + 1, nx - 1)
    iy1 = jnp.minimum(iy0 + 1, ny - 1)

    return (
        (1 - fx) * (1 - fy) * data[ix0, iy0]
        + fx * (1 - fy) * data[ix1, iy0]
        + (1 - fx) * fy * data[ix0, iy1]
        + fx * fy * data[ix1, iy1]
    )


def interpolate_pixel_rates(grid, values, coords):
    """Bilinear interpolation of the FOV evaluation grid at pixel centres.

    ``values`` is ``(n_lon, n_lat)``, lon-major like every grid in nyx, and
    ``coords`` ``(n_pixels, 2)`` as ``[lon, lat]`` in radians.  ``grid``
    holds the ``(n,)`` node coordinates: the grid is square and both axes
    share them, so there is nothing to put in the wrong order.  Returns one
    value per pixel, zero outside the grid.
    """
    n_lon, n_lat = values.shape
    start, step = grid[0], grid[1] - grid[0]

    # Axis 0 is longitude, so it takes column 0 of the coordinate pair.
    lon_coords = (coords[:, 0] - start) / step
    lat_coords = (coords[:, 1] - start) / step

    valid_mask = (
        (lon_coords >= 0)
        & (lon_coords <= n_lon - 1)
        & (lat_coords >= 0)
        & (lat_coords <= n_lat - 1)
    )

    # Hat weights: max(0, 1 - |c - i|)
    w_lon = jnp.maximum(1.0 - jnp.abs(lon_coords[:, None] - jnp.arange(n_lon)), 0.0)
    w_lat = jnp.maximum(1.0 - jnp.abs(lat_coords[:, None] - jnp.arange(n_lat)), 0.0)
    interpolated = jnp.sum((w_lon @ values) * w_lat, axis=1)
    return jnp.where(valid_mask, interpolated, 0.0)
