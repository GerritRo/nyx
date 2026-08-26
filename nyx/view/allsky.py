from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import equinox as eqx
import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np

from nyx.core.coordinates import offset_to_altaz
from nyx.core.filters import select_obs
from nyx.core.protocols import AtmosphereModel, SkySource, SourceObsData
from nyx.view.response import LUMA, SpectralResponse, to_linear_srgb

if TYPE_CHECKING:
    from nyx.core.observation import Observation, RenderGeometry, SkyGeometry

__all__ = ["PointField", "SkyRender", "render_prepared", "render_sky"]

# Peak element count of one chunk of the (targets, source pixels, wavelength)
# scattering kernel.  32M float32 ~ 128 MB.
_KERNEL_BUDGET = 32_000_000

# Emitters with at most this many point sources keep the exact per-source
# scattering path; larger catalogs are binned into the coarse source map,
# whose pixels are far smaller than the scattering halo they produce.
# The default covers the Moon and a planet or two -- the only point sources
# whose halo is bright enough to be worth resolving.
_EXACT_SCATTER = 32


# Hemisphere plumbing


def _hemisphere_dirs(nside: int) -> tuple[np.ndarray, np.ndarray]:
    """Azimuth and altitude of the above-horizon HEALPix pixels.

    Matches :class:`~nyx.core.geometry.Geometry`: RING ordering, upper
    hemisphere only, which is the contiguous leading block of the map.
    """
    theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    keep = theta < np.pi / 2
    return phi[keep], np.pi / 2 - theta[keep]


def _retarget(sky: SkyGeometry, az: np.ndarray, alt: np.ndarray) -> SkyGeometry:
    """Copy *sky* with its FOV evaluation grid replaced by arbitrary directions.

    The atmosphere computes in-scattering onto ``fov_altaz_grid``; aiming
    that grid at a set of directions turns the existing scattering
    machinery into an all-sky map without touching the atmosphere model.
    Laid out as an ``(n, 1)`` grid so every shape the atmosphere expects
    still holds.
    """
    grid = np.stack([az, alt], axis=-1)[:, None, :]  # (n, 1, 2)
    return eqx.tree_at(lambda s: s.fov_altaz_grid, sky, jnp.asarray(grid))


def _pointing_altaz(geometry: RenderGeometry) -> tuple[float, float]:
    """Telescope ``(az, alt)`` in radians, recovered from the pointing matrix."""
    az, alt = offset_to_altaz(0.0, 0.0, geometry.pointing_matrix)
    return float(az), float(alt)


def _coarse_sky(sky: SkyGeometry, nside_out: int) -> tuple[SkyGeometry, np.ndarray]:
    """A copy of *sky* whose hemisphere is resampled to *nside_out*.

    Returns the coarse geometry and the index of the fine pixel nearest
    each coarse one.  Positions are taken from the fine map by nearest
    neighbour rather than averaged, because the auxiliary frames
    (``icrs_coord``, ``sref_coord``) are angles that must not be mixed;
    the offset is at most half a fine pixel.
    """
    az, alt = _hemisphere_dirs(nside_out)
    nside_in = hp.npix2nside(int(sky.hemisphere_mask.shape[0]))
    idx = hp.ang2pix(nside_in, np.pi / 2 - alt, az)
    take = jnp.asarray(idx)
    mask = np.zeros(hp.nside2npix(nside_out), dtype=bool)
    mask[: az.size] = True

    coarse = eqx.tree_at(
        lambda s: (s.altaz_coord, s.icrs_coord, s.sref_coord, s.hemisphere_mask),
        sky,
        (
            jnp.stack([jnp.asarray(az), jnp.asarray(alt)], axis=-1),
            sky.icrs_coord[take],
            sky.sref_coord[take],
            jnp.asarray(mask),
        ),
    )
    return coarse, idx


def _coarsen_radiance(radiance: jax.Array, nside_in: int, nside_out: int) -> jax.Array:
    """Hemisphere radiance at *nside_in* averaged down to *nside_out*.

    The average runs over the above-horizon children only, so the coarse
    pixels straddling the horizon are not diluted by the empty half of
    the sphere.  The result is then scaled by the pixel-area ratio: the
    atmosphere multiplies the scattering kernel by *its* pixel area, and
    this radiance is summed over that many fewer pixels.
    """
    values = np.asarray(radiance)
    nsky, n_wvl = values.shape
    npix_in = hp.nside2npix(nside_in)
    full = np.zeros((n_wvl, npix_in))
    full[:, :nsky] = values.T

    covered = np.zeros(npix_in)
    covered[:nsky] = 1.0
    fraction = hp.ud_grade(covered, nside_out)
    means = hp.ud_grade(full, nside_out) / np.where(fraction > 0, fraction, 1.0)

    nsky_out = int(np.count_nonzero(fraction > 0.5))
    scale = npix_in / hp.nside2npix(nside_out)
    return jnp.asarray(means[:, :nsky_out].T * scale)


def _upsample(values: np.ndarray, nside_in: int, nside_out: int) -> np.ndarray:
    """Hemisphere values ``(n, 3)`` at *nside_in* to a full map at *nside_out*.

    Bilinear, with the colatitude clipped into the coarse hemisphere so
    the fine pixels straddling the horizon interpolate along the lowest
    coarse ring instead of into the empty half-sphere.  Nearest
    neighbour would be cheaper, but the scattered field is exactly the
    thing that must not come out in facets.
    """
    coarse = np.zeros((hp.nside2npix(nside_in), values.shape[1]))
    coarse[: values.shape[0]] = values
    if nside_in == nside_out:
        return coarse

    theta, phi = hp.pix2ang(nside_out, np.arange(hp.nside2npix(nside_out)))
    theta_max = float(hp.pix2ang(nside_in, values.shape[0] - 1)[0])
    clipped = np.minimum(theta, theta_max)
    fine = np.stack(
        [hp.get_interp_val(coarse[:, c], clipped, phi) for c in range(values.shape[1])], axis=-1
    )
    fine[theta >= np.pi / 2] = 0.0
    return fine


# Jitted kernels


@eqx.filter_jit
def _extinct_channels(
    atmo: AtmosphereModel,
    sky: SkyGeometry,
    radiance: dict[str, jax.Array],
    response: jax.Array,
) -> dict[str, jax.Array]:
    """Each diffuse map through line-of-sight extinction, per channel."""
    extinction = atmo.evaluate(sky).extinction_hp
    return {
        name: jnp.einsum("sw,sw,wc->sc", value, extinction, response)
        for name, value in radiance.items()
    }


@eqx.filter_jit
def _point_channels(
    atmo: AtmosphereModel,
    spectra: jax.Array,
    altitudes: jax.Array,
    height_km: jax.Array,
    response: jax.Array,
) -> jax.Array:
    """Extincted, band-integrated flux of each point source, per channel."""
    return atmo.extinct(altitudes, spectra, height_km) @ response


@eqx.filter_jit
def _scatter_channels(
    atmo: AtmosphereModel,
    sky: SkyGeometry,
    radiance: dict[str, jax.Array],
    response: jax.Array,
) -> dict[str, jax.Array]:
    """In-scattered radiance towards every target direction, per channel.

    One :meth:`~nyx.core.protocols.AtmosphereModel.evaluate` builds the
    kernel; every emitter and all three channels are contracted against
    that one copy.
    """
    kernel = atmo.evaluate(sky).scattering_map[:, 0]  # (targets, sources, n_wvl)
    return {
        name: jnp.einsum("tsw,sw,wc->tc", kernel, value, response)
        for name, value in radiance.items()
    }


@eqx.filter_jit
def _scatter_points(
    atmo: AtmosphereModel,
    sky: SkyGeometry,
    coords: jax.Array,
    spectra: jax.Array,
    response: jax.Array,
) -> jax.Array:
    """In-scattered light of individual point sources, per channel."""
    columns = [
        jnp.broadcast_to(
            jnp.asarray(atmo.scatter_sources(sky, coords, spectra, response[:, c])),
            sky.fov_altaz_grid.shape[:2],
        )[:, 0]
        for c in range(response.shape[1])
    ]
    return jnp.stack(columns, axis=-1)


# Results


@dataclasses.dataclass(frozen=True)
class PointField:
    """Point sources of one emitter, as they reach the telescope.

    Attributes
    ----------
    az, alt : np.ndarray, shape (n,)
        Positions in radians.
    flux : np.ndarray, shape (n, 3)
        Extincted, band-integrated flux per channel,
        ``photon / s / m^2`` weighted by the camera response.
    """

    az: np.ndarray
    alt: np.ndarray
    flux: np.ndarray

    def above_horizon(self) -> PointField:
        """Copy holding only the sources above the horizon."""
        keep = self.alt > 0
        return PointField(self.az[keep], self.alt[keep], self.flux[keep])


@dataclasses.dataclass(frozen=True)
class SkyRender:
    """Per-emitter maps of one sky, ready to be looked at.

    Diffuse maps are full-sphere HEALPix arrays in RING ordering with a
    trailing channel axis ``(npix, k)``, zero below the horizon, holding
    radiance in the units of
    :attr:`~nyx.view.response.SpectralResponse.channels` -- ``photon / s
    / sr`` through a telescope bandpass, tristimulus per steradian
    through :meth:`~nyx.view.response.SpectralResponse.cie`.  Point
    sources are kept apart from the maps, at their exact positions, so a
    camera can give them a point spread function instead of a HEALPix
    pixel; :meth:`binned` puts them into pixels when a map view wants
    them there instead.

    Attributes
    ----------
    direct : dict of {source: np.ndarray}
        Emitter radiance seen through line-of-sight extinction,
        ``(npix, k)``.
    indirect : dict of {source: np.ndarray}
        Light that emitter scatters into each line of sight,
        ``(npix, k)``.
    points : dict of {source: PointField}
        Emitters rendered as individual sources rather than a map.
    nside : int
        HEALPix resolution of the maps.
    response : SpectralResponse
        The response the maps were integrated against.
    pointing : tuple of float
        Telescope ``(az, alt)`` in radians for this observation.
    label : str
        What the render was made for, for figure titles.
    obs_index : int
        Which observation of *obs* was rendered.
    scatter_nside, target_nside, point_nside : int
        Resolutions the scattering integral was evaluated at.
    """

    direct: dict[str, np.ndarray]
    indirect: dict[str, np.ndarray]
    points: dict[str, PointField]
    nside: int
    response: SpectralResponse
    pointing: tuple[float, float] = (0.0, 0.0)
    label: str = "sky"
    obs_index: int = 0
    scatter_nside: int = 0
    target_nside: int = 0
    point_nside: int = 0

    @property
    def unit(self) -> str:
        """Units of the maps, for axis labels."""
        return "photon / s / sr" if self.response.n_channels == 1 else "tristimulus / sr"

    @property
    def sources(self) -> list[str]:
        """Emitter names, in the order they were rendered."""
        return list(dict.fromkeys([*self.direct, *self.indirect, *self.points]))

    @property
    def mask(self) -> np.ndarray:
        """Boolean above-horizon mask of the map pixels."""
        theta, _ = hp.pix2ang(self.nside, np.arange(hp.nside2npix(self.nside)))
        return np.asarray(theta < np.pi / 2)

    @property
    def horizon_theta(self) -> float:
        """Colatitude of the lowest populated ring of the map.

        HEALPix puts a ring exactly on the equator, which the hemisphere
        excludes; interpolating below this angle would blend the sky with
        that empty ring and darken the horizon.
        """
        theta, _ = hp.pix2ang(self.nside, np.arange(hp.nside2npix(self.nside)))
        return float(theta[theta < np.pi / 2].max())

    def _names(self, sources: list[str] | str | None) -> list[str]:
        if sources is None:
            return self.sources
        names = [sources] if isinstance(sources, str) else list(sources)
        unknown = [n for n in names if n not in self.sources]
        if unknown:
            raise KeyError(f"{unknown} not in this sky; sources: {self.sources}")
        return names

    def diffuse(
        self,
        sources: list[str] | str | None = None,
        *,
        direct: bool = True,
        inscatter: bool = True,
    ) -> np.ndarray:
        """Summed diffuse map, ``(npix, 3)``.

        Parameters
        ----------
        sources : list of str or str, optional
            Restrict to these emitters (default: all).
        direct, inscatter : bool
            Which of the two paths to include.
        """
        total = np.zeros((hp.nside2npix(self.nside), 3))
        for name in self._names(sources):
            if direct and name in self.direct:
                total += self.direct[name]
            if inscatter and name in self.indirect:
                total += self.indirect[name]
        return total

    def point_field(self, sources: list[str] | str | None = None) -> PointField:
        """Point sources of the requested emitters, concatenated."""
        fields = [self.points[n] for n in self._names(sources) if n in self.points]
        if not fields:
            return PointField(np.zeros(0), np.zeros(0), np.zeros((0, 3)))
        return PointField(
            np.concatenate([f.az for f in fields]),
            np.concatenate([f.alt for f in fields]),
            np.concatenate([f.flux for f in fields]),
        )

    def binned(self, sources: list[str] | str | None = None) -> np.ndarray:
        """Point sources accumulated into map pixels, ``(npix, k)``.

        A map has nowhere to put a point source but a pixel.  The
        camera path never uses this -- it splats them with a point
        spread function instead -- but a HEALPix panel has to.
        """
        field = self.point_field(sources).above_horizon()
        return _bin_points(field, self.nside) / hp.nside2pixarea(self.nside)

    def direct_map(
        self, sources: list[str] | str | None = None, *, points: bool = True
    ) -> np.ndarray:
        """Direct (extincted) light, ``(npix, k)``, point sources included."""
        total = self.diffuse(sources, inscatter=False)
        return total + self.binned(sources) if points else total

    def indirect_map(self, sources: list[str] | str | None = None) -> np.ndarray:
        """In-scattered light, ``(npix, k)``."""
        return self.diffuse(sources, direct=False)

    def total(self, sources: list[str] | str | None = None, *, points: bool = True) -> np.ndarray:
        """Direct plus in-scattered, ``(npix, k)``."""
        return self.direct_map(sources, points=points) + self.indirect_map(sources)

    def __getitem__(self, name: str) -> np.ndarray:
        """Total map of a single emitter."""
        return self.total([name])

    def masked(self, values: np.ndarray) -> np.ndarray:
        """Copy of *values* with the below-horizon pixels set to NaN.

        HEALPix plotting routines render NaN as the *bad* colour, which
        keeps the horizon clean.  The camera path uses zeros instead.
        """
        mask = self.mask
        return np.where(mask.reshape(mask.shape + (1,) * (values.ndim - 1)), values, np.nan)

    def plot(self, kind: str = "components", **kwargs: Any) -> Any:
        """Draw the hemisphere; see :func:`nyx.view.display.plot_maps`."""
        from nyx.view.display import plot_maps

        return plot_maps(self, kind, **kwargs)

    def scalar(self, values: np.ndarray) -> np.ndarray:
        """Reduce channel values ``(..., k)`` to one number per pixel.

        A map panel and a summary table both need a single brightness.
        A one-channel response already is one; a colorimetric one becomes
        luminance; anything else is summed.
        """
        if self.response.n_channels == 1:
            return np.asarray(values)[..., 0]
        if self.response.to_srgb is not None:
            return to_linear_srgb(values, self.response) @ LUMA
        return np.asarray(values).sum(axis=-1)

    def summary(self) -> str:
        """Hemisphere-mean brightness of every component, as a table."""
        mask = self.mask
        zero = np.zeros((mask.size, self.response.n_channels))

        def mean(values: np.ndarray) -> float:
            return float(np.mean(self.scalar(values)[mask]))

        rows = [
            (
                name,
                mean(self.direct.get(name, zero) + self.binned([name])),
                mean(self.indirect.get(name, zero)),
            )
            for name in self.sources
        ]
        rows.append(("total", sum(r[1] for r in rows), sum(r[2] for r in rows)))

        header = (
            f"SkyRender({self.label!r}, obs={self.obs_index}, nside={self.nside}, "
            f"channels={self.response.n_channels}, scatter_nside={self.scatter_nside}, "
            f"target_nside={self.target_nside})\n"
            f"  hemisphere mean [{self.unit}]"
        )
        return _table(header, rows)

    def __repr__(self) -> str:
        return self.summary()


def _neighbour_smooth(maps: np.ndarray, nside: int, passes: int) -> np.ndarray:
    """Average each map pixel with its HEALPix neighbours, *passes* times.

    A map sampled at half a degree, magnified until one pixel is ten
    image pixels across, shows the HEALPix tessellation itself: bilinear
    interpolation over that grid leaves the diamond pattern the pixels
    are cut in.  It is a sampling artefact, not sky, and one or two
    passes of this take it out at a cost of about one pixel of blur.

    Real space rather than spherical harmonics on purpose: the maps drop
    to zero at the horizon, and a transform of that step would ring
    along the one line of the picture where a ripple would be obvious.
    Below-horizon pixels are excluded from the average instead.

    Parameters
    ----------
    maps : np.ndarray, shape (n, npix, 3)
        Stacked full-sphere maps.
    nside : int
        Their resolution.
    passes : int
        Number of averaging passes.

    Returns
    -------
    np.ndarray, same shape as *maps*
    """
    if passes <= 0 or maps.size == 0:
        return maps
    npix = hp.nside2npix(nside)
    theta, _ = hp.pix2ang(nside, np.arange(npix))
    sky = theta < np.pi / 2

    neighbours = hp.get_all_neighbours(nside, np.arange(npix))  # (8, npix)
    index = np.where(neighbours >= 0, neighbours, 0)
    valid = (neighbours >= 0) & sky[index]
    index = np.where(valid, index, 0)

    # A weight of four on the pixel itself keeps the kernel compact.
    count = (valid.sum(axis=0) + 4.0)[None, :, None]
    out = maps
    for _ in range(passes):
        # One neighbour at a time: gathering all eight at once would build an
        # (n, 8, npix, 3) temporary, which at nside 512 is several gigabytes.
        total = 4.0 * out
        for taken, keep in zip(index, valid, strict=True):
            total += np.where(keep[None, :, None], out[:, taken, :], 0.0)
        out = np.where(sky[None, :, None], total / count, 0.0)
    return out


def _table(header: str, rows: list[tuple[str, float, float]]) -> str:
    """The direct / indirect / total table every view prints."""
    width = max([len(r[0]) for r in rows] + [6])
    lines = [
        header,
        f"  {'source':<{width}}  {'direct':>12}  {'indirect':>12}  {'total':>12}",
        "  " + "-" * (width + 44),
    ]
    for name, direct, indirect in rows:
        lines.append(
            f"  {name:<{width}}  {direct:>12.4g}  {indirect:>12.4g}  {direct + indirect:>12.4g}"
        )
    return "\n".join(lines)


def _bin_points(field: PointField, nside: int) -> np.ndarray:
    """Point fluxes accumulated into HEALPix pixels, ``(npix, 3)``."""
    binned = np.zeros((hp.nside2npix(nside), 3))
    if field.az.size:
        pix = hp.ang2pix(nside, np.pi / 2 - field.alt, field.az)
        np.add.at(binned, pix, field.flux)
    return binned


# Rendering


def render_sky(
    obs: Observation,
    atmosphere: AtmosphereModel,
    emitters: dict[str, Any] | list[Any],
    response: SpectralResponse,
    *,
    obs_index: int = 0,
    label: str = "sky",
    indirect: bool = True,
    scatter_nside: int | None = None,
    target_nside: int | None = None,
    point_nside: int | None = None,
    exact_scatter: int = _EXACT_SCATTER,
    smooth: int = 1,
    chunk: int | None = None,
) -> SkyRender:
    """Render every emitter of a sky into per-channel HEALPix maps.

    This is the expensive half of taking a photograph, and it is done
    once: the result keeps each emitter's direct and in-scattered
    contributions apart, so any combination of them can be exposed
    afterwards for the cost of an array sum.

    Parameters
    ----------
    obs : Observation
        Observation to render.  Its :class:`~nyx.core.geometry.Geometry`
        sets the wavelength grid and the map resolution; the pointing is
        irrelevant here, since the whole hemisphere is rendered.
    atmosphere : AtmosphereModel
        Shared atmosphere, as for :meth:`~nyx.core.scene.Scene.build`.
    emitters : dict of {name: EmitterBuilder} or list
        Sky sources.  A list is auto-named from class names.
    response : SpectralResponse
        Camera response on ``obs.geom.wvls``; see :func:`SpectralResponse.cie`.
    obs_index : int
        Which observation time to render (default: the first).
    label : str
        What this render is of, carried through to figure titles --
        an instrument name, say.
    indirect : bool
        Compute the in-scattered maps.  This is the expensive half of
        the expensive half.
    scatter_nside : int, optional
        Resolution of the sky being scattered *from* (default: 32, or
        the map's own resolution if that is coarser).  The scattering
        kernel integrates the whole sky against a broad phase function,
        so it does not need the detail the direct image does.
    target_nside : int, optional
        Resolution of the directions the sky is scattered *into*
        (default: 32).  The scattered field is smooth; it is
        interpolated back onto the fine map afterwards.  Together with
        ``scatter_nside`` this fixes the cost of the whole scattering
        integral -- dropping both to 16 is sixteen times faster, for a
        percent-level error, which is what a preview wants.
    point_nside : int, optional
        Resolution of the directions *point sources* are scattered into
        (default: 64).  Kept finer than ``target_nside`` because it
        costs almost nothing and because a lunar aureole is the sharpest
        feature a moonlit sky has.
    exact_scatter : int
        Emitters with at most this many point sources scatter through
        :meth:`~nyx.core.protocols.AtmosphereModel.scatter_sources`, one
        source at a time -- what the Moon needs, since its halo is
        structured on scales the source map cannot resolve.  Larger
        catalogs are binned into the coarse source map instead.
    smooth : int
        Passes of a nearest-neighbour average over the finished diffuse
        maps, roughly a pixel of blur each.  Magnified into a
        photograph, an unsmoothed HEALPix map shows its own
        tessellation; one pass takes that out.  Set to 0 to keep the
        maps exactly as rendered.
    chunk : int, optional
        Target directions per batch.  Defaults to a fixed memory budget
        for the scattering kernel.

    Returns
    -------
    SkyRender

    Notes
    -----
    Which path an emitter takes is decided by the same two
    :class:`~nyx.core.protocols.SourceObsData` flags the render loop
    uses, so a view and a render agree by construction.  A diffuse map
    is always rendered.  Point sources are rendered individually only
    when the emitter has no map of its own to carry them -- the Moon, a
    bright-star catalog -- and they scatter individually only when
    ``inscatter`` is set.  A catalog emitter that has both is therefore
    counted once, through its map.

    That last point is why splitting a catalog for an all-sky view takes
    a deliberate step: see
    :func:`~nyx.emitter.stars.gaia_star_field`, which partitions Gaia
    into point sources and the background they leave behind.
    """
    if isinstance(emitters, (list, tuple)):
        emitters = {type(e).__name__: e for e in emitters}
    if not -obs.nobs <= obs_index < obs.nobs:
        # JAX clamps out-of-bounds indices instead of raising, so this would
        # otherwise silently return the last observation.
        raise IndexError(f"obs_index {obs_index} out of range; obs has {obs.nobs} times")
    if response.n_wvl != len(obs.geom.wvls):
        raise ValueError(
            f"response covers {response.n_wvl} wavelengths but the geometry has "
            f"{len(obs.geom.wvls)}; build it on the same grid (geo.wvls)"
        )

    models: dict[str, SkySource] = {}
    obs_data: dict[str, SourceObsData] = {}
    for name, emitter in emitters.items():
        models[name] = emitter.model()
        obs_data[name] = select_obs(emitter.prepare(obs), obs_index)

    return render_prepared(
        atmosphere,
        models,
        obs_data,
        obs.get_render_geometry()[obs_index],
        response,
        label=label,
        obs_index=obs_index,
        indirect=indirect,
        scatter_nside=scatter_nside,
        target_nside=target_nside,
        point_nside=point_nside,
        exact_scatter=exact_scatter,
        smooth=smooth,
        chunk=chunk,
    )


def render_prepared(
    atmosphere: AtmosphereModel,
    models: dict[str, SkySource],
    obs_data: dict[str, SourceObsData],
    geometry: RenderGeometry,
    response: SpectralResponse,
    *,
    label: str = "sky",
    obs_index: int = 0,
    indirect: bool = True,
    scatter_nside: int | None = None,
    target_nside: int | None = None,
    point_nside: int | None = None,
    exact_scatter: int = _EXACT_SCATTER,
    smooth: int = 1,
    chunk: int | None = None,
) -> SkyRender:
    """Render from emitter data that is already prepared for one observation.

    The half of :func:`render_sky` that does the work, split out so that
    a caller holding prepared data rather than emitter builders --
    :meth:`~nyx.core.scene.Scene.sky_view`, which keeps precomputed
    per-observation bundles and no longer has the builders -- can render
    without repeating the preparation.

    Parameters
    ----------
    atmosphere : AtmosphereModel
    models : dict of {name: SkySource}
        Shared source models, as :meth:`nyx.core.protocols.EmitterBuilder.model`
        returns them.
    obs_data : dict of {name: SourceObsData}
        Their per-observation data, **already sliced to one observation**.
    geometry : RenderGeometry
        Single-observation geometry and pointing.
    response, label, obs_index, indirect, scatter_nside, target_nside, point_nside, exact_scatter, smooth, chunk
        As :func:`render_sky`.

    Returns
    -------
    SkyRender
    """
    sky = geometry.sky
    resp = jnp.asarray(response.channels)
    npix = int(sky.hemisphere_mask.shape[0])  # full sphere
    nsky = int(sky.altaz_coord.shape[0])  # above the horizon only
    map_nside = hp.npix2nside(npix)
    pixel_area = hp.nside2pixarea(map_nside)
    scatter_nside = _default_nside(scatter_nside, map_nside, 32)
    target_nside = _default_nside(target_nside, map_nside, 32)
    point_nside = _default_nside(point_nside, map_nside, 64)

    # Split the emitters into the paths they occupy.  The rule is the
    # render loop's, read off the same two SourceObsData flags: a diffuse
    # map is always rendered, point sources are rendered individually only
    # when the emitter has no map of its own to carry them, and they
    # scatter individually only when the emitter asks for it.  A catalog
    # emitter has both a map and points, and must not be counted twice.
    radiance: dict[str, jax.Array] = {}
    direct_points: dict[str, tuple[jax.Array, jax.Array]] = {}
    scatter_points: dict[str, tuple[jax.Array, jax.Array]] = {}
    for name, model in models.items():
        data = obs_data[name]
        diffuse = model.diffuse_radiance(sky, data)
        if diffuse is not None:
            radiance[name] = diffuse
        points = model.point_sources(data)
        if points is not None:
            if data.diffuse_conditions is None:
                direct_points[name] = (points.coords, points.spectra)
            if data.inscatter:
                scatter_points[name] = (points.coords, points.spectra)

    # Direct: extinction does not depend on the FOV grid, so a single
    # target direction keeps the scattering kernel out of the way.
    az_sky = np.asarray(sky.altaz_coord[:, 0])
    alt_sky = np.asarray(sky.altaz_coord[:, 1])
    thin = _retarget(sky, az_sky[:1], alt_sky[:1])

    direct: dict[str, np.ndarray] = {}
    for name, values in _extinct_channels(atmosphere, thin, radiance, resp).items():
        full = np.zeros((npix, response.n_channels))
        full[:nsky] = np.asarray(values)
        direct[name] = full

    points_out: dict[str, PointField] = {}
    for name, (coords, spectra) in direct_points.items():
        flux = _point_channels(atmosphere, spectra, coords[:, 1:2], sky.height_km, resp)
        points_out[name] = PointField(
            np.asarray(coords[:, 0]), np.asarray(coords[:, 1]), np.asarray(flux)
        )

    indirect_out: dict[str, np.ndarray] = {}
    if indirect and (radiance or scatter_points):
        indirect_out = _render_indirect(
            atmosphere,
            sky,
            radiance,
            scatter_points,
            resp,
            map_nside=map_nside,
            pixel_area=pixel_area,
            scatter_nside=scatter_nside,
            target_nside=target_nside,
            point_nside=point_nside,
            exact_scatter=exact_scatter,
            chunk=chunk,
        )

    if smooth:
        keys = [(direct, n) for n in direct] + [(indirect_out, n) for n in indirect_out]
        stacked = _neighbour_smooth(np.stack([store[n] for store, n in keys]), map_nside, smooth)
        for (store, n), values in zip(keys, stacked, strict=True):
            store[n] = values

    ordered = [n for n in models if n in direct or n in indirect_out or n in points_out]
    return SkyRender(
        direct={n: direct[n] for n in ordered if n in direct},
        indirect={n: indirect_out[n] for n in ordered if n in indirect_out},
        points={n: points_out[n] for n in ordered if n in points_out},
        nside=map_nside,
        response=response,
        pointing=_pointing_altaz(geometry),
        label=label,
        obs_index=obs_index,
        scatter_nside=scatter_nside,
        target_nside=target_nside,
        point_nside=point_nside,
    )


def _default_nside(value: int | None, map_nside: int, cap: int) -> int:
    """Validate an explicit nside, or fall back to *cap* capped by the map."""
    if value is None:
        return min(cap, map_nside)
    if not hp.isnsideok(value, nest=True):
        raise ValueError(f"nside must be a power of two, got {value}")
    return value


def _render_indirect(
    atmosphere: AtmosphereModel,
    sky: SkyGeometry,
    radiance: dict[str, jax.Array],
    point_data: dict[str, tuple[jax.Array, jax.Array]],
    resp: jax.Array,
    *,
    map_nside: int,
    pixel_area: float,
    scatter_nside: int,
    target_nside: int,
    point_nside: int,
    exact_scatter: int,
    chunk: int | None,
) -> dict[str, np.ndarray]:
    """In-scattered map of every emitter, resampled back onto the fine map.

    Two passes, because the two source kinds cost differently.  Scattering
    the *sky* costs ``n_targets x n_source_pixels``, so its targets are
    kept coarse.  Scattering a handful of *point sources* costs
    ``n_targets x n_sources``, which is nothing -- and the Moon's aureole
    is the sharpest structure in a moonlit sky, so those targets are kept
    fine.
    """
    coarse, _ = _coarse_sky(sky, scatter_nside)
    sources = {
        name: _coarsen_radiance(values, map_nside, scatter_nside)
        for name, values in radiance.items()
    }

    # A catalog of point sources is far denser than the halo any one of
    # them scatters, so it is folded into the source map; only the sparse
    # emitters keep the exact per-source path.
    exact: dict[str, tuple[jax.Array, jax.Array]] = {}
    n_coarse = int(coarse.altaz_coord.shape[0])
    for name, (coords, spectra) in point_data.items():
        if int(coords.shape[0]) <= exact_scatter:
            exact[name] = (coords, spectra)
            continue
        alt = np.asarray(coords[:, 1])
        keep = alt > 0
        pix = hp.ang2pix(scatter_nside, np.pi / 2 - alt[keep], np.asarray(coords[keep, 0]))
        binned = np.zeros((n_coarse, int(spectra.shape[1])))
        np.add.at(binned, np.minimum(pix, n_coarse - 1), np.asarray(spectra)[keep])
        # _coarsen_radiance carries the same pixel-area convention: the
        # atmosphere multiplies the kernel by the *fine* pixel area.
        sources[name] = jnp.asarray(binned / pixel_area)

    maps: dict[str, np.ndarray] = {}

    if sources:
        budget = chunk or max(1, _KERNEL_BUDGET // (n_coarse * int(resp.shape[0])))
        for name, values in _sweep(
            lambda targets: _scatter_channels(atmosphere, targets, sources, resp),
            coarse,
            target_nside,
            budget,
        ).items():
            maps[name] = _upsample(values, target_nside, map_nside)

    for name, (coords, spectra) in exact.items():
        budget = chunk or max(1, _KERNEL_BUDGET // (int(coords.shape[0]) * int(resp.shape[0])))
        scattered = _sweep(
            lambda targets, c=coords, s=spectra, n=name: {
                n: _scatter_points(atmosphere, targets, c, s, resp)
            },
            coarse,
            point_nside,
            budget,
        )[name]
        values = _upsample(scattered, point_nside, map_nside)
        maps[name] = maps[name] + values if name in maps else values

    return maps


def _sweep(
    kernel: Any,
    sky: SkyGeometry,
    target_nside: int,
    chunk: int,
) -> dict[str, np.ndarray]:
    """Run *kernel* over every above-horizon direction at *target_nside*.

    The directions are fed in batches of at most *chunk*, keeping the
    scattering kernel inside a fixed memory budget however many of them
    there are.
    """
    az, alt = _hemisphere_dirs(target_nside)
    parts: dict[str, list[np.ndarray]] = {}
    for lo in range(0, az.size, max(1, chunk)):
        batch = kernel(_retarget(sky, az[lo : lo + chunk], alt[lo : lo + chunk]))
        for name, values in batch.items():
            parts.setdefault(name, []).append(np.asarray(values))
    return {name: np.concatenate(values) for name, values in parts.items()}
