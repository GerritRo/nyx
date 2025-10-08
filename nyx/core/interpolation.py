import jax.numpy as jnp
from jax import jit, vmap

@jit
def compute_pixel_weights(centers, values_stack, coords, rates):
    def compute_single_weight(centers_item, values):
        height, width = values.shape
        n_points = coords.shape[0]
        
        ystart, ystep = centers_item[0, 0], centers_item[0, 1] - centers_item[0, 0]
        xstart, xstep = centers_item[1, 0], centers_item[1, 1] - centers_item[1, 0]
        
        # Transform coordinates
        y_coords = (coords[:, 0] - ystart) / ystep
        x_coords = (coords[:, 1] - xstart) / xstep
        
        # Create mask for valid coordinates
        valid_mask = (
            (y_coords >= 0) & (y_coords < height - 1) &
            (x_coords >= 0) & (x_coords < width - 1)
        )
        
        # Get integer and fractional parts
        y0 = jnp.floor(y_coords).astype(jnp.int32)
        x0 = jnp.floor(x_coords).astype(jnp.int32)
        y1 = y0 + 1
        x1 = x0 + 1
        fy = y_coords - y0
        fx = x_coords - x0
        
        # Ensure indices are within bounds (for safety)
        y0 = jnp.clip(y0, 0, height - 2)
        x0 = jnp.clip(x0, 0, width - 2)
        y1 = jnp.clip(y1, 0, height - 1)
        x1 = jnp.clip(x1, 0, width - 1)
        
        # Bilinear interpolation
        v00 = values[y0, x0]
        v01 = values[y1, x0]
        v10 = values[y0, x1]
        v11 = values[y1, x1]
        
        interpolated = (
            (1 - fx) * (1 - fy) * v00 +
            fx * (1 - fy) * v10 +
            (1 - fx) * fy * v01 +
            fx * fy * v11
        )
        
        # Apply mask and sum
        return jnp.sum(jnp.where(valid_mask, interpolated*rates, 0.0))
    
    # Vectorize over the first axis (items)
    return vmap(compute_single_weight)(centers, values_stack)

@jit
def interpolate_pixel_rates(Xi, Yi, values, coords):
    """
    """
    height, width = values.shape
    
    ystart, ystep = Yi[0], Yi[1] - Yi[0]
    xstart, xstep = Xi[0], Xi[1] - Xi[0]
    
    # Transform all coordinates at once
    y_coords = (coords[:, 0] - ystart) / ystep
    x_coords = (coords[:, 1] - xstart) / xstep
    
    # Create mask for valid coordinates
    valid_mask = (
        (y_coords >= 0) & (y_coords < height - 1) &
        (x_coords >= 0) & (x_coords < width - 1)
    )
    
    # Get integer and fractional parts
    y0 = jnp.floor(y_coords).astype(jnp.int32)
    x0 = jnp.floor(x_coords).astype(jnp.int32)
    y1 = y0 + 1
    x1 = x0 + 1
    fy = y_coords - y0
    fx = x_coords - x0
    
    # Ensure indices are within bounds
    y0 = jnp.clip(y0, 0, height - 2)
    x0 = jnp.clip(x0, 0, width - 2)
    y1 = jnp.clip(y1, 0, height - 1)
    x1 = jnp.clip(x1, 0, width - 1)
    
    # Bilinear interpolation for all coordinates
    v00 = values[y0, x0]
    v01 = values[y1, x0]
    v10 = values[y0, x1]
    v11 = values[y1, x1]
    
    interpolated = (
        (1 - fx) * (1 - fy) * v00 +
        fx * (1 - fy) * v10 +
        (1 - fx) * fy * v01 +
        fx * fy * v11
    )
    
    # Apply mask - return 0 for invalid coordinates
    return jnp.where(valid_mask, interpolated, 0.0)
