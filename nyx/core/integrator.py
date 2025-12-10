import jax.numpy as jnp
from jax import jit
import jax_healpy as jhp

from .interpolation import interpolate_pixel_rates, compute_pixel_weights

@jit
def render(scene):
    # Sum diffuse maps
    diff_sum = jnp.sum(jnp.stack([x.flux_map for x in scene.diffuse]), axis=0)
    cat_diff = jnp.sum(jnp.stack([x.flux_map for x in scene.catalog]), axis=0)

    # Indirect Scattering contribution
    scat_value = jnp.sum(scene.instrument.bandpass * (diff_sum + cat_diff) * scene.atmosphere.scattering, axis=(-2,-1))
    scat_rates = interpolate_pixel_rates(scene.atmosphere.YiXi[0], scene.atmosphere.YiXi[1], scat_value, scene.instrument.centers)
    scat_rates = scat_rates * scene.instrument.weight

    # Direct diffuse contribution using jax_healpy for differentiable interpolation
    diff_value = jnp.sum(scene.instrument.bandpass * diff_sum * scene.atmosphere.extinction, axis=1)
    diff_rates = jhp.get_interp_val(diff_value, scene.instrument.hp_theta, scene.instrument.hp_phi)
    diff_rates = diff_rates * scene.instrument.weight

    # Direct catalog contribution:
    cat_rates = []
    for cat in scene.catalog:
        cat_value = jnp.sum(cat.flux_values * scene.instrument.bandpass *
                            jnp.exp(-scene.atmosphere.tau[None,:] * cat.sec_Z[:,None]), axis=1)
        cat_rates.append(compute_pixel_weights(scene.instrument.grid, scene.instrument.values, cat.image_coords, cat_value))
    
    return scat_rates + diff_rates + jnp.sum(jnp.stack(cat_rates), axis=0)