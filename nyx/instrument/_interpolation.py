import jax
import jax.numpy as jnp


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


def _bilinear_sample(values, y0, x0, fy, fx):
    """Sample a 2D grid with bilinear weights."""
    y1 = y0 + 1
    x1 = x0 + 1
    return (
        (1 - fx) * (1 - fy) * values[y0, x0]
        + fx * (1 - fy) * values[y0, x1]
        + (1 - fx) * fy * values[y1, x0]
        + fx * fy * values[y1, x1]
    )


def compute_pixel_weights(centers, values_stack, coords, rates, batch_size=None):
    """Bilinear interpolation of point source rates onto pixel grid.

    Parameters
    ----------
    batch_size : int or None
        Number of pixels processed in parallel per ``lax.map`` iteration.
        ``None`` (default) means fully parallel, equivalent to
        ``jax.vmap`` over pixels, one kernel launch, peak memory
        ``O(n_pixels * n_sources)`` intermediates on the backward pass.
        A smaller value bounds that memory at
        ``O(batch_size * n_sources)`` at the cost of
        ``ceil(n_pixels / batch_size)`` serial kernel launches.
        ``batch_size=1`` is fully serial (minimum memory, maximum
        launch overhead).
    """

    @jax.checkpoint
    def compute_single_weight(args):
        centers_item, values = args
        height, width = values.shape
        ystart, ystep = centers_item[0, 0], centers_item[0, 1] - centers_item[0, 0]
        xstart, xstep = centers_item[1, 0], centers_item[1, 1] - centers_item[1, 0]

        y_coords = (coords[:, 0] - ystart) / ystep
        x_coords = (coords[:, 1] - xstart) / xstep

        y0, x0, fy, fx, valid_mask = _bilinear_coeffs(y_coords, x_coords, height, width)
        interpolated = _bilinear_sample(values, y0, x0, fy, fx)
        return jnp.sum(jnp.where(valid_mask, interpolated * rates, 0.0))

    if batch_size is None:
        batch_size = centers.shape[0]
    return jax.lax.map(
        compute_single_weight,
        (centers, values_stack),
        batch_size=batch_size,
    )


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


def interpolate_pixel_rates(Xi, Yi, values, coords):
    """Bilinear interpolation of gridded values at arbitrary coordinates."""
    height, width = values.shape
    ystart, ystep = Yi[0], Yi[1] - Yi[0]
    xstart, xstep = Xi[0], Xi[1] - Xi[0]

    y_coords = (coords[:, 0] - ystart) / ystep
    x_coords = (coords[:, 1] - xstart) / xstep

    y0, x0, fy, fx, valid_mask = _bilinear_coeffs(y_coords, x_coords, height, width)
    interpolated = _bilinear_sample(values, y0, x0, fy, fx)
    return jnp.where(valid_mask, interpolated, 0.0)
