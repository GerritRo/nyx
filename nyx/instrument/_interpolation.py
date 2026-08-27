from functools import cache

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import simpson

LATTICE_TOL = 1e-3


def _bilinear_coeffs(y_coords, x_coords, height, width):
    """Compute bilinear interpolation coefficients and validity mask.

    Parameters
    ----------
    y_coords, x_coords : jax.Array
        Fractional grid coordinates.
    height, width : int
        Grid dimensions.

    Returns
    -------
    y0, x0, fy, fx : jax.Array
        Integer indices and fractional offsets.
    valid_mask : jax.Array
        Boolean mask for in-bounds coordinates.
    """
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
        """Centre of each pixel's response window, radians. Shape (n_pixels, 2).
        """
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

        Parameters
        ----------
        grid : array, shape (n_pixels, 2, grid_dim)
            Pixel sub-grid coordinates in radians.
        values : array, shape (..., n_pixels, height, width)
            Pixel response values.  Leading axes (e.g. a misalignment
            table) are all checked.
        tol : float
            Alignment tolerance, in units of one response-grid step.

        Returns
        -------
        PixelLattice

        Raises
        ------
        ValueError
            If the response tables do not vanish on their boundary rows and
            columns, or if the sample coordinates do not sit on the nodes of
            a common lattice to within *tol*.  Both are required for the
            projection to be separable; see :func:`project_lattice`.
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
    """Least-squares (origin, step) of the lattice underlying *coords_1d*.

    Parameters
    ----------
    coords_1d : ndarray, shape (n_pixels, grid_dim)
        Sample positions of every pixel's response grid along one axis.

    Returns
    -------
    origin, step : float
    residual : float
        Largest deviation of any sample from a lattice node, in steps.
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

    Parameters
    ----------
    lattice : PixelLattice
        Focal-plane geometry.
    values : jax.Array, shape (n_pixels, height, width)
        Pixel response values.
    coords : jax.Array, shape (n_sources, 2)
        Source positions in the detector frame, in radians.
    rates : jax.Array, shape (n_sources,)
        Band-integrated source rates.

    Returns
    -------
    jax.Array, shape (n_pixels,)
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
    """Quadrature weights reproducing ``scipy`` Simpson on a unit-spaced axis.
    """
    return simpson(np.eye(n), dx=1.0, axis=-1)


def integrate_response(lattice: PixelLattice, values):
    """Integrate each pixel's response over solid angle.

    Parameters
    ----------
    lattice : PixelLattice
        Supplies the (uniform) node spacing along each axis.
    values : array, shape (..., n_pixels, height, width)
        Pixel response values.

    Returns
    -------
    jax.Array, shape ``values.shape[:-2]``
        Simpson-integrated weight per pixel.
    """
    height, width = lattice.grid_shape
    if values.shape[-2:] != (height, width):
        raise ValueError(f"response is {values.shape[-2:]}, lattice window is {(height, width)}")
    wy = jnp.asarray(_simpson_weights(height)) * lattice.step[0]
    wx = jnp.asarray(_simpson_weights(width)) * lattice.step[1]
    return jnp.einsum("...jk,j,k->...", values, wy, wx)


def response_centroid(lattice: PixelLattice, values):
    """Where each pixel looks.
    
    Parameters
    ----------
    lattice : PixelLattice
        Supplies the node spacing and each window's corner.
    values : array, shape (..., n_pixels, height, width)
        Pixel response values.

    Returns
    -------
    jax.Array, shape ``values.shape[:-2] + (2,)``
        Field offset per pixel as ``[lon, lat]``, in radians.
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
    """Bilinear interpolation on a regular 2-D grid with clamped boundaries.

    Interpolates ``data[ix, iy, ...]`` at fractional position ``(x, y)``
    given a regular grid defined by origin, step, and count along each axis.
    Values outside the grid are clamped to the nearest edge.

    Parameters
    ----------
    x, y : scalar jax arrays
        Query coordinates.
    x0, y0 : float
        Grid origin (first coordinate value) along each axis.
    x_step, y_step : float
        Grid spacing along each axis.
    nx, ny : int
        Number of grid points along each axis.
    data : jax.Array
        Array with leading dims ``(nx, ny, ...)``.

    Returns
    -------
    jax.Array
        Interpolated value with shape ``data.shape[2:]``.
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


def interpolate_pixel_rates(lat_grid, lon_grid, values, coords):
    """Bilinear interpolation of the FOV evaluation grid at pixel centres.
    Parameters
    ----------
    lat_grid, lon_grid : jax.Array
        Node coordinates of the evaluation grid along the row and column
        axis respectively, in radians.
    values : jax.Array, shape (n_lat, n_lon)
        Gridded values to sample.
    coords : jax.Array, shape (n_pixels, 2)
        Pixel centres as ``[lon, lat]``, in radians.

    Returns
    -------
    jax.Array, shape (n_pixels,)
        Interpolated value per pixel; zero outside the grid.
    """
    height, width = values.shape
    lat_start, lat_step = lat_grid[0], lat_grid[1] - lat_grid[0]
    lon_start, lon_step = lon_grid[0], lon_grid[1] - lon_grid[0]

    lat_coords = (coords[:, 1] - lat_start) / lat_step
    lon_coords = (coords[:, 0] - lon_start) / lon_step

    valid_mask = (
        (lat_coords >= 0)
        & (lat_coords <= height - 1)
        & (lon_coords >= 0)
        & (lon_coords <= width - 1)
    )

    # Hat weights: max(0, 1 - |c - i|)
    w_lat = jnp.maximum(1.0 - jnp.abs(lat_coords[:, None] - jnp.arange(height)), 0.0)
    w_lon = jnp.maximum(1.0 - jnp.abs(lon_coords[:, None] - jnp.arange(width)), 0.0)
    interpolated = jnp.sum((w_lat @ values) * w_lon, axis=1)
    return jnp.where(valid_mask, interpolated, 0.0)
