from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from nyx.core.coordinates import altaz_to_offset

if TYPE_CHECKING:
    from nyx.core.scene import _RenderFrame


def render(scene: _RenderFrame) -> jax.Array:
    """Render a single-observation scene to pixel rates.

    Each source carries ``direct`` and ``inscatter`` flags on its
    :class:`SourceObsData` that control the rendering path:

    - **Diffuse radiance** is always scattered (map scattering).
      If ``direct=True``, it also goes through line-of-sight extinction.
    - **Point sources** always get extinction + pixel projection.
      If ``inscatter=True``, individual in-scattering is computed via
      ``scatter_sources``.

    Parameters
    ----------
    scene : _RenderFrame or compatible pytree (single obs)

    Returns
    -------
    jax.Array, shape (n_pixels,)
        Photon detection rate per pixel [photon/s].
    """
    inst = scene.instrument
    atmo = scene.atmosphere
    sky = scene.render_geometry.sky
    pm = scene.render_geometry.pointing_matrix
    bp = inst.bandpass

    pm_corr = inst.corrected_pm(pm)
    atmo_result = atmo.evaluate(sky)
    nsky, n_wvl = atmo_result.nsky, atmo_result.n_wvl

    # Accumulate diffuse radiance: all sources scatter, some also extinct
    direct_radiance = jnp.zeros((nsky, n_wvl))
    scatter_radiance = jnp.zeros((nsky, n_wvl))

    for source, obs_data in scene.sources:
        diffuse = source.diffuse_radiance(sky, obs_data)
        if diffuse is not None:
            scatter_radiance = scatter_radiance + diffuse
            if obs_data.direct:
                direct_radiance = direct_radiance + diffuse

    direct = atmo_result.apply_extinction(direct_radiance, bp)
    scattered = atmo_result.apply_scattering(scatter_radiance, bp)

    # Point sources: in-scatter (if flagged) + extinct + project
    all_coords = []
    all_fluxes = []

    for source, obs_data in scene.sources:
        pts = source.point_sources(obs_data)
        if pts is not None:
            if obs_data.inscatter:
                scattered = scattered + atmo.scatter_sources(
                    sky,
                    pts.coords,
                    pts.spectra,
                    bp,
                )
            az, alt = pts.coords[:, 0], pts.coords[:, 1]
            extincted = atmo.extinct(alt[:, None], pts.spectra, sky.height_km)
            flux = jnp.sum(extincted * bp, axis=1)
            lon, lat = altaz_to_offset(az, alt, pm_corr)
            all_coords.append(jnp.stack([lon, lat], axis=-1))
            all_fluxes.append(flux)

    if all_coords:
        coords = jnp.concatenate(all_coords, axis=0)
        fluxes = jnp.concatenate(all_fluxes, axis=0)
    else:
        coords = jnp.zeros((0, 2))
        fluxes = jnp.zeros((0,))

    # Project onto detector pixels
    return inst.efficiency.value * (
        inst.project_diffuse(direct, pm)
        + inst.project_scattered(scattered)
        + inst.project_catalog(coords, fluxes)
    )
