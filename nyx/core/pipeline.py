from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from nyx.core.coordinates import altaz_to_offset
from nyx.core.protocols import AtmosphereModel, InstrumentModel, SkySource
from nyx.core.records import RenderGeometry, SkyGeometry, SourceObsData, unwrap

__all__ = ["RenderFrame", "contributions", "render"]


class RenderFrame(eqx.Module):
    """One instrument's view of the scene."""

    atmosphere: AtmosphereModel
    sources: list[tuple[SkySource, SourceObsData]]
    instrument: InstrumentModel
    render_geometry: RenderGeometry
    nobs: int = eqx.field(static=True)
    # Static
    source_names: tuple[str, ...] = eqx.field(static=True, default=())


class _PointTerm(NamedTuple):
    """One source's point-source contribution, before it is projected."""

    inscatter: jax.Array | None
    coords: jax.Array
    flux: jax.Array


def _point_term(
    source: SkySource,
    obs_data: SourceObsData,
    atmo: AtmosphereModel,
    sky: SkyGeometry,
    bp: jax.Array,
    pm_corr: jax.Array,
) -> _PointTerm | None:
    """Extinct, band-integrate and project point sources.

    Returns
    -------
    _PointTerm or None
        ``None`` if the source has no point component.
    """
    pts = source.point_sources(obs_data)
    if pts is None:
        return None
    inscatter = None
    if obs_data.inscatter:
        inscatter = atmo.scatter_sources(sky, pts.coords, pts.spectra, bp)
    az, alt = pts.coords[:, 0], pts.coords[:, 1]
    extincted = atmo.extinct(alt[:, None], pts.spectra, sky.height_km)
    flux = jnp.sum(extincted * bp, axis=1)
    lon, lat = altaz_to_offset(az, alt, pm_corr)
    return _PointTerm(inscatter, jnp.stack([lon, lat], axis=-1), flux)


def contributions(scene: RenderFrame) -> dict[str, jax.Array]:
    """Pixel rates of each source separately, summing to :func:`render`.

    Parameters
    ----------
    scene : RenderFrame
        Single-observation frame.

    Returns
    -------
    dict of str to jax.Array
        One entry per source name, each of shape ``(n_pixels,)``.
    """
    inst = scene.instrument
    atmo = scene.atmosphere
    sky = unwrap(scene.render_geometry.sky)
    pm = unwrap(scene.render_geometry.pointing_matrix)
    bp = inst.bandpass
    pm_corr = inst.corrected_pm(pm)
    atmo_result = atmo.evaluate(sky)

    out: dict[str, jax.Array] = {}
    for name, (source, obs_data) in zip(scene.source_names, scene.sources, strict=True):
        rate = jnp.zeros(())
        diffuse = source.diffuse_radiance(obs_data)
        if diffuse is not None:
            rate = rate + inst.project_scattered(atmo_result.apply_scattering(diffuse, bp))
            if obs_data.direct:
                rate = rate + inst.project_diffuse(atmo_result.apply_extinction(diffuse, bp), pm)
        term = _point_term(source, obs_data, atmo, sky, bp, pm_corr)
        if term is not None:
            if term.inscatter is not None:
                rate = rate + inst.project_scattered(term.inscatter)
            rate = rate + inst.project_catalog(term.coords, term.flux)
        out[name] = inst.efficiency.value * rate

    # Give every entry the pixel shape so the dict is uniform.
    shaped = [r.shape for r in out.values() if r.ndim]
    if shaped:
        out = {k: jnp.broadcast_to(v, shaped[0]) for k, v in out.items()}
    return out


def render(scene: RenderFrame) -> jax.Array:
    """Render a single-observation scene to pixel rates.

    Diffuse radiance is always map-scattered, and additionally extincted
    along the line of sight when ``direct=True``. Point sources are always
    extincted and projected, and in-scattered individually when
    ``inscatter=True``.

    Parameters
    ----------
    scene : RenderFrame
        Single-observation frame, or a compatible pytree.

    Returns
    -------
    jax.Array, shape (n_pixels,)
        Photon detection rate per pixel, in p.e./s.
    """
    inst = scene.instrument
    atmo = scene.atmosphere
    sky = unwrap(scene.render_geometry.sky)
    pm = unwrap(scene.render_geometry.pointing_matrix)
    bp = inst.bandpass

    pm_corr = inst.corrected_pm(pm)
    atmo_result = atmo.evaluate(sky)
    nsky, n_wvl = atmo_result.nsky, atmo_result.n_wvl

    direct_radiance = jnp.zeros((nsky, n_wvl))
    scatter_radiance = jnp.zeros((nsky, n_wvl))

    for source, obs_data in scene.sources:
        diffuse = source.diffuse_radiance(obs_data)
        if diffuse is not None:
            scatter_radiance = scatter_radiance + diffuse
            if obs_data.direct:
                direct_radiance = direct_radiance + diffuse

    direct = atmo_result.apply_extinction(direct_radiance, bp)
    scattered = atmo_result.apply_scattering(scatter_radiance, bp)

    all_coords = []
    all_fluxes = []

    for source, obs_data in scene.sources:
        term = _point_term(source, obs_data, atmo, sky, bp, pm_corr)
        if term is not None:
            if term.inscatter is not None:
                scattered = scattered + term.inscatter
            all_coords.append(term.coords)
            all_fluxes.append(term.flux)

    if all_coords:
        coords = jnp.concatenate(all_coords, axis=0)
        fluxes = jnp.concatenate(all_fluxes, axis=0)
    else:
        coords = jnp.zeros((0, 2))
        fluxes = jnp.zeros((0,))

    return inst.efficiency.value * (
        inst.project_diffuse(direct, pm)
        + inst.project_scattered(scattered)
        + inst.project_catalog(coords, fluxes)
    )
