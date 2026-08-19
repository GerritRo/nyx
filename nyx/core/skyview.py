"""Band-integrated hemisphere maps of a built :class:`~nyx.core.scene.Scene`.

The render loop already evaluates every emitter on a HEALPix hemisphere --
that map is what the atmosphere scatters into the field of view.  This
module band-integrates those same maps through an instrument's passband
and returns them as plain HEALPix arrays, so the sky a telescope sits
under can be inspected directly:

- the **direct** map is the emitter's own radiance seen through
  line-of-sight extinction, and
- the **indirect** map is the light the atmosphere scatters into each
  line of sight from the rest of the sky.

Nothing here changes the forward model.  The indirect map is obtained by
pointing the atmosphere's FOV evaluation grid at the hemisphere itself
(see :func:`_retarget`), so :meth:`~nyx.core.protocols.AtmosphereModel.evaluate`
and :meth:`~nyx.core.protocols.AtmosphereModel.scatter_sources` are reused
verbatim.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import equinox as eqx
import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np

from nyx.core.coordinates import offset_to_altaz
from nyx.core.filters import per_obs_filter
from nyx.core.protocols import AtmosphereModel, PointSourceData

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from nyx.core.observation import SkyGeometry
    from nyx.core.scene import Scene, _RenderFrame

__all__ = ["SkyView", "sky_view"]

# Peak element count of one chunk of the (targets, sky pixels, wavelength)
# scattering kernel.  32M float32 ~ 128 MB, the same order as the render
# loop's own working set at the default resolutions.
_KERNEL_BUDGET = 32_000_000

_UNIT = "photon / s / sr"


# Hemisphere geometry


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
    that grid at the hemisphere turns the existing scattering machinery
    into an all-sky map without touching the atmosphere model.  The
    directions are laid out as a ``(n, 1)`` grid so every shape the
    atmosphere expects still holds, and the scattering angles follow from
    the grid.

    Parameters
    ----------
    sky : SkyGeometry
        Single-observation geometry (from the render frame).
    az, alt : np.ndarray, shape (n,)
        Target directions in radians.

    Returns
    -------
    SkyGeometry
    """
    grid = np.stack([az, alt], axis=-1)[:, None, :]  # (n, 1, 2)
    return eqx.tree_at(lambda s: s.fov_altaz_grid, sky, jnp.asarray(grid))


def _to_full_map(values: np.ndarray, nside_in: int, nside_out: int) -> np.ndarray:
    """Hemisphere values at *nside_in* to a full-sphere map at *nside_out*.

    Upsampling is nearest-neighbour with the colatitude clipped into the
    coarse hemisphere, so the fine pixels straddling the horizon take the
    lowest coarse ring instead of falling into the empty half-sphere.
    """
    coarse = np.zeros(hp.nside2npix(nside_in))
    coarse[: values.shape[0]] = values
    if nside_in == nside_out:
        return coarse

    theta, phi = hp.pix2ang(nside_out, np.arange(hp.nside2npix(nside_out)))
    theta_max = float(hp.pix2ang(nside_in, values.shape[0] - 1)[0])
    fine = coarse[hp.ang2pix(nside_in, np.minimum(theta, theta_max), phi)]
    fine[theta >= np.pi / 2] = 0.0
    return fine


# Jitted kernels


@eqx.filter_jit
def _extinct_maps(
    atmo: AtmosphereModel,
    sky: SkyGeometry,
    radiance: dict[str, jax.Array],
    bp: jax.Array,
) -> dict[str, jax.Array]:
    """Band-integrate each diffuse map through line-of-sight extinction."""
    result = atmo.evaluate(sky)
    return {name: result.apply_extinction(value, bp) for name, value in radiance.items()}


@eqx.filter_jit
def _scatter_maps(
    atmo: AtmosphereModel,
    sky: SkyGeometry,
    radiance: dict[str, jax.Array],
    points: dict[str, tuple[jax.Array, jax.Array]],
    bp: jax.Array,
) -> dict[str, jax.Array]:
    """In-scattered radiance towards every target direction of *sky*."""
    result = atmo.evaluate(sky)
    out = {name: result.apply_scattering(value, bp)[:, 0] for name, value in radiance.items()}
    for name, (coords, spectra) in points.items():
        # broadcast_to also absorbs the ``0.0`` an atmosphere without
        # point-source scattering returns.
        scattered = jnp.broadcast_to(
            jnp.asarray(atmo.scatter_sources(sky, coords, spectra, bp)),
            sky.fov_altaz_grid.shape[:2],
        )[:, 0]
        out[name] = out[name] + scattered if name in out else scattered
    return out


@eqx.filter_jit
def _point_flux(
    atmo: AtmosphereModel,
    sky: SkyGeometry,
    points: PointSourceData,
    bp: jax.Array,
) -> jax.Array:
    """Band-integrated, extincted flux of each point source [photon / s]."""
    extincted = atmo.extinct(points.coords[:, 1:2], points.spectra, sky.height_km)
    return jnp.sum(extincted * bp, axis=1)


# Scene plumbing


def _single_obs(frame: _RenderFrame, index: int) -> _RenderFrame:
    """Slice observation *index* out of a multi-observation render frame.

    The same partition :meth:`~nyx.core.scene.Scene.render` vmaps over,
    indexed instead of mapped.
    """
    per_obs, shared = eqx.partition(frame, per_obs_filter(frame))
    return eqx.combine(shared, jax.tree.map(lambda x: x[index], per_obs))


def _pointing_altaz(frame: _RenderFrame) -> tuple[float, float]:
    """Telescope ``(az, alt)`` in radians, recovered from the pointing matrix."""
    az, alt = offset_to_altaz(0.0, 0.0, frame.render_geometry.pointing_matrix)
    return float(az), float(alt)


# Public API


@dataclasses.dataclass(frozen=True)
class SkyView:
    """Per-emitter hemisphere maps in one instrument's passband.

    Maps are full-sphere HEALPix arrays in RING ordering (zero below the
    horizon) holding the photon rate per steradian that reaches the
    telescope aperture, ``photon / s / sr``: the emitter radiance
    integrated against the instrument bandpass (effective aperture x
    transmission x wavelength bin).  Detector-side factors -- overall
    efficiency, per-pixel efficiency and pixel solid angle -- are *not*
    applied, so the maps describe the sky over the telescope rather than
    any one camera pixel.

    Attributes
    ----------
    direct : dict of {source: np.ndarray}
        Emitter radiance seen through line-of-sight extinction.
    indirect : dict of {source: np.ndarray}
        Light that emitter scatters into each line of sight.
    nside : int
        HEALPix resolution of the maps.
    pointing : tuple of float
        Telescope ``(az, alt)`` in radians for this observation.
    instrument, obs_index : str, int
        What the view was computed for.

    Examples
    --------
    ::

        view = scene.sky_view()
        view.plot()                       # direct | indirect | total
        view.plot('grid')                 # one row per emitter
        view.plot(sources=['moon'])       # rescaled to one emitter
        view['moon']                      # that emitter's total map
        hp.orthview(view.masked(view.total()), rot=[0, 90, 0], half_sky=True)
    """

    direct: dict[str, np.ndarray]
    indirect: dict[str, np.ndarray]
    nside: int
    pointing: tuple[float, float]
    instrument: str = "instrument"
    obs_index: int = 0
    unit: str = _UNIT

    # Accessors

    @property
    def sources(self) -> list[str]:
        """Emitter names, in scene order."""
        return list(dict.fromkeys([*self.direct, *self.indirect]))

    @property
    def mask(self) -> np.ndarray:
        """Boolean above-horizon mask of the map pixels."""
        theta, _ = hp.pix2ang(self.nside, np.arange(hp.nside2npix(self.nside)))
        return theta < np.pi / 2

    def _sum(self, maps: dict[str, np.ndarray], sources: list[str] | None) -> np.ndarray:
        names = self.sources if sources is None else sources
        total = np.zeros(hp.nside2npix(self.nside))
        for name in names:
            if name in maps:
                total = total + maps[name]
        return total

    def direct_map(self, sources: list[str] | None = None) -> np.ndarray:
        """Direct contribution summed over *sources* (default: all)."""
        return self._sum(self.direct, sources)

    def indirect_map(self, sources: list[str] | None = None) -> np.ndarray:
        """Indirect (in-scattered) contribution summed over *sources*."""
        return self._sum(self.indirect, sources)

    def total(self, sources: list[str] | None = None) -> np.ndarray:
        """Direct plus indirect, summed over *sources* (default: all)."""
        return self.direct_map(sources) + self.indirect_map(sources)

    def __getitem__(self, name: str) -> np.ndarray:
        """Total (direct + indirect) map of a single emitter."""
        if name not in self.sources:
            raise KeyError(f"{name!r} is not in this view; sources: {self.sources}")
        return self.total([name])

    def masked(self, values: np.ndarray) -> np.ndarray:
        """Copy of *values* with the below-horizon pixels set to NaN.

        HEALPix plotting routines render NaN as the *bad* colour, which
        keeps the horizon clean.
        """
        return np.where(self.mask, values, np.nan)

    # Reporting

    def summary(self) -> str:
        """Hemisphere-mean brightness of every component as a table."""
        mask = self.mask
        rows = [
            (
                name,
                float(np.mean(self.direct.get(name, np.zeros(mask.size))[mask])),
                float(np.mean(self.indirect.get(name, np.zeros(mask.size))[mask])),
            )
            for name in self.sources
        ]
        rows.append(
            (
                "total",
                float(np.mean(self.direct_map()[mask])),
                float(np.mean(self.indirect_map()[mask])),
            )
        )
        width = max([len(r[0]) for r in rows] + [6])
        header = (
            f"SkyView({self.instrument!r}, obs={self.obs_index}, nside={self.nside}, "
            f"pointing=({np.degrees(self.pointing[0]):.1f}, "
            f"{np.degrees(self.pointing[1]):.1f}) deg)\n"
            f"  hemisphere mean [{self.unit}]\n"
            f"  {'source':<{width}}  {'direct':>12}  {'indirect':>12}  {'total':>12}"
        )
        lines = [header, "  " + "-" * (width + 44)]
        for name, direct, indirect in rows:
            lines.append(
                f"  {name:<{width}}  {direct:>12.4g}  {indirect:>12.4g}  {direct + indirect:>12.4g}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    # Plotting

    def _layout(self, kind: str, sources: list[str] | None) -> tuple[list[tuple[str, Any]], int]:
        """Panel titles, maps and column count for a plot *kind*."""
        names = self.sources if sources is None else sources
        if kind == "components":
            return [
                ("direct", self.direct_map(names)),
                ("indirect", self.indirect_map(names)),
                ("total", self.total(names)),
            ], 3
        if kind == "sources":
            panels = [(name, self.total([name])) for name in names]
            if len(names) > 1:
                panels.append(("total", self.total(names)))
            return panels, min(len(panels), 3)
        if kind == "grid":
            panels = []
            for name in [*names, "total"]:
                subset = names if name == "total" else [name]
                panels += [
                    (f"{name} | direct", self.direct_map(subset)),
                    (f"{name} | indirect", self.indirect_map(subset)),
                    (f"{name} | total", self.total(subset)),
                ]
            return panels, 3
        raise ValueError(f"kind must be 'components', 'sources' or 'grid', not {kind!r}")

    def _limits(
        self, maps: list[np.ndarray], log: bool, vmin: float | None, vmax: float | None
    ) -> tuple[float, float]:
        """Colour limits shared by every panel.

        The upper limit is the 99.9th percentile rather than the maximum:
        a point source binned into a single pixel is orders of magnitude
        above its surroundings and would otherwise flatten the sky.
        """
        values = np.concatenate([m[self.mask] for m in maps])
        values = values[np.isfinite(values)]
        if log:
            values = values[values > 0]
        if values.size == 0:  # nothing to show (e.g. the Moon is down)
            return (vmin if vmin is not None else 0.0), (vmax if vmax is not None else 1.0)
        hi = float(np.percentile(values, 99.9)) if vmax is None else vmax
        lo = float(values.min()) if vmin is None else vmin
        if log:
            lo = max(lo, hi * 1e-5)
        return lo, max(hi, lo * (1 + 1e-6))

    def _annotate(self, compass: bool) -> None:
        """Mark the pointing and the cardinal directions on the current panel."""
        az, alt = self.pointing
        if alt > 0:
            hp.projscatter(
                np.pi / 2 - alt,
                az,
                marker="+",
                s=90,
                linewidths=1.3,
                color="white",
                zorder=10,
            )
        if compass:
            # Orthographic radius is cos(alt), so a label near the rim has
            # to sit well up from the horizon to stay clear of it.
            for label, direction in (("N", 0.0), ("E", 90.0), ("S", 180.0), ("W", 270.0)):
                hp.projtext(
                    np.radians(70.0),
                    np.radians(direction),
                    label,
                    color="white",
                    alpha=0.7,
                    fontsize=9,
                    horizontalalignment="center",
                    verticalalignment="center",
                )

    def plot(
        self,
        kind: str = "components",
        *,
        sources: list[str] | None = None,
        log: bool = True,
        cmap: str = "magma",
        vmin: float | None = None,
        vmax: float | None = None,
        compass: bool = True,
        graticule: bool = True,
        figsize: tuple[float, float] | None = None,
        fig: Figure | None = None,
    ) -> Figure:
        """Draw the hemisphere in orthographic projection, zenith at the centre.

        The horizon is the rim, north is up and east is left -- the sky as
        an observer under it sees it.  All panels share one colour scale
        and one colour bar, so components can be compared by eye.

        Parameters
        ----------
        kind : {'components', 'sources', 'grid'}
            ``'components'`` shows direct, indirect and total summed over
            all emitters; ``'sources'`` one total panel per emitter;
            ``'grid'`` a full emitter x component matrix.
        sources : list of str, optional
            Restrict to these emitters (default: all).  The colour scale
            follows, which is the way to bring out a faint component that
            a bright one flattens.
        log : bool
            Logarithmic colour scale (default).  Sky brightness spans
            decades between the Moon and a dark patch.
        cmap, vmin, vmax : str, float, float
            Colour map and limits.  Limits default to the minimum and the
            99.9th percentile over every panel; values outside are clipped
            rather than dropped.
        compass, graticule : bool
            Draw cardinal-direction labels / an alt-az grid.
        figsize, fig : tuple, matplotlib Figure
            Figure size (default: scaled to the panel count) and an
            existing figure to draw into.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import LogNorm, Normalize

        panels, ncols = self._layout(kind, sources)
        nrows = -(-len(panels) // ncols)
        # Inches reserved for the figure title, the colour bar, the gap
        # between panels and the strip each panel title sits in.
        head, foot, gap, strip = 0.45, 1.0, 0.12, 0.3
        if fig is None:
            fig = plt.figure(figsize=figsize or (4.0 * ncols, 4.0 * nrows + head + foot))
        else:
            plt.figure(fig.number)  # healpy draws into the current figure
        width, height = fig.get_size_inches()

        lo, hi = self._limits([m for _, m in panels], log, vmin, vmax)
        log = log and lo > 0
        top, bottom = 1.0 - head / height, foot / height
        cell_w, cell_h = 1.0 / ncols, (top - bottom) / nrows

        for i, (title, values) in enumerate(panels):
            row, col = divmod(i, ncols)
            axes = fig.add_axes(
                (
                    col * cell_w + gap / width,
                    top - (row + 1) * cell_h,
                    cell_w - 2 * gap / width,
                    cell_h - strip / height,
                )
            )
            axes.set_axis_off()
            hp.orthview(
                self.masked(np.clip(values, lo, hi)),
                hold=True,  # take over the axes just created
                half_sky=True,
                rot=(0, 90, 180),
                flip="geo",  # with the 180 deg roll: north up, east left
                title=title,
                cbar=False,
                cmap=cmap,
                norm="log" if log else None,
                min=lo,
                max=hi,
                badcolor="none",
                notext=True,
            )
            self._annotate(compass)
        if graticule:
            hp.graticule(dpar=30, dmer=45, color="0.7", alpha=0.5, lw=0.4)

        bar = fig.add_axes((0.3, 0.55 * foot / height, 0.4, 0.14 / height))
        fig.colorbar(
            ScalarMappable(norm=LogNorm(lo, hi) if log else Normalize(lo, hi), cmap=cmap),
            cax=bar,
            orientation="horizontal",
            label=self.unit,
        )

        az, alt = np.degrees(self.pointing)
        fig.suptitle(
            f"{self.instrument} | obs {self.obs_index} | pointing az {az:.1f}, alt {alt:.1f} deg",
            fontsize=11,
            y=1.0 - 0.5 * head / height,
            verticalalignment="center",
        )
        return fig


def sky_view(
    scene: Scene,
    instrument: str | None = None,
    obs: int = 0,
    *,
    indirect: bool = True,
    point_sources: bool = True,
    nside: int | None = None,
    chunk: int | None = None,
) -> SkyView:
    """Band-integrated hemisphere maps of every emitter in *scene*.

    Implements :meth:`nyx.core.scene.Scene.sky_view`; see :class:`SkyView`
    for the result.

    Parameters
    ----------
    scene : Scene
        A built scene.
    instrument : str, optional
        Instrument whose passband and pointing are used; defaults to the
        sole instrument.
    obs : int
        Observation index (default: the first).
    indirect : bool
        Compute the in-scattered maps.  This is the expensive half: it
        pairs every target direction with every sky pixel.
    point_sources : bool
        Bin emitters that have no diffuse map (Moon, bright stars) into
        the direct map at their HEALPix pixel.
    nside : int, optional
        Resolution of the *target directions* of the indirect map
        (default: the scene's own).  Lowering it is the way to trade
        detail for speed: the sky being scattered from is always kept at
        full resolution, only the directions scattered into are sampled
        more coarsely, and the map is resampled back to the scene's nside.
        The scattered field is smooth, so halving nside costs a few per
        cent near the horizon and much less higher up.
    chunk : int, optional
        Target directions evaluated per batch.  Defaults to a fixed
        memory budget for the scattering kernel.

    Returns
    -------
    SkyView

    Notes
    -----
    The indirect maps follow the render loop exactly: every diffuse map
    scatters, and point sources scatter when their
    :class:`~nyx.core.protocols.SourceObsData` sets ``inscatter``.

    The direct maps differ in one place.  A catalog emitter such as
    :class:`~nyx.emitter.stars.Stars` carries ``direct=False``, because
    the render loop takes its direct light from quasi-point sources
    inside the field of view rather than from the map.  Over the whole
    hemisphere there are no such point sources, so its diffuse map is
    extincted and shown instead -- the same physics, sampled at the map
    resolution.  Each emitter therefore appears exactly once: through its
    diffuse map if it has one, otherwise through its point sources.
    """
    name = scene._resolve_instrument(instrument)
    nobs = scene._obs_bundles[name].nobs
    if not -nobs <= obs < nobs:
        # JAX clamps out-of-bounds indices instead of raising, so this
        # would otherwise silently return the last observation.
        raise IndexError(f"obs {obs} out of range; {name!r} has {nobs} observations")
    if nside is not None and not hp.isnsideok(nside, nest=True):
        raise ValueError(f"nside must be a power of two, got {nside}")
    frame = _single_obs(scene._render_frame(name), obs)
    source_names = list(scene._obs_bundles[name].obs_data)

    sky = frame.render_geometry.sky
    atmo = frame.atmosphere
    bp = frame.instrument.bandpass

    npix = int(sky.hemisphere_mask.shape[0])
    map_nside = hp.npix2nside(npix)
    pixel_area = hp.nside2pixarea(map_nside)
    az_sky = np.asarray(sky.altaz_coord[:, 0])
    alt_sky = np.asarray(sky.altaz_coord[:, 1])

    # Split the emitters into the paths they occupy on the hemisphere.
    radiance: dict[str, jax.Array] = {}
    direct_points: dict[str, PointSourceData] = {}
    scatter_points: dict[str, tuple[jax.Array, jax.Array]] = {}
    for src_name, (source, obs_data) in zip(source_names, frame.sources, strict=True):
        diffuse = source.diffuse_radiance(sky, obs_data)
        if diffuse is not None:
            radiance[src_name] = diffuse
        points = source.point_sources(obs_data)
        if points is not None:
            if obs_data.diffuse_conditions is None:
                direct_points[src_name] = points
            if obs_data.inscatter:
                scatter_points[src_name] = (points.coords, points.spectra)

    # Direct: extinction does not depend on the FOV grid, so a single
    # target direction keeps the scattering kernel out of the way.
    direct: dict[str, np.ndarray] = {
        src_name: np.asarray(values)
        for src_name, values in _extinct_maps(
            atmo, _retarget(sky, az_sky[:1], alt_sky[:1]), radiance, bp
        ).items()
    }

    if point_sources:
        for src_name, points in direct_points.items():
            flux = np.asarray(_point_flux(atmo, sky, points, bp))
            coords = np.asarray(points.coords)
            above = coords[:, 1] > 0
            binned = np.zeros(npix)
            np.add.at(
                binned,
                hp.ang2pix(map_nside, np.pi / 2 - coords[above, 1], coords[above, 0]),
                flux[above] / pixel_area,
            )
            direct[src_name] = direct.get(src_name, np.zeros(npix)) + binned

    indirect_maps: dict[str, np.ndarray] = {}
    if indirect and (radiance or scatter_points):
        target_nside = map_nside if nside is None else nside
        if target_nside == map_nside:
            az_t, alt_t = az_sky, alt_sky
        else:
            az_t, alt_t = _hemisphere_dirs(target_nside)
        if chunk is None:
            chunk = max(1, _KERNEL_BUDGET // (az_sky.size * int(bp.shape[0])))

        parts: dict[str, list[np.ndarray]] = {}
        for lo in range(0, az_t.size, chunk):
            hi = min(lo + chunk, az_t.size)
            chunk_maps = _scatter_maps(
                atmo,
                _retarget(sky, az_t[lo:hi], alt_t[lo:hi]),
                radiance,
                scatter_points,
                bp,
            )
            for src_name, values in chunk_maps.items():
                parts.setdefault(src_name, []).append(np.asarray(values))

        indirect_maps = {
            src_name: _to_full_map(np.concatenate(values), target_nside, map_nside)
            for src_name, values in parts.items()
        }

    pointing = _pointing_altaz(frame)
    ordered = [n for n in source_names if n in direct or n in indirect_maps]
    return SkyView(
        direct={n: direct[n] for n in ordered if n in direct},
        indirect={n: indirect_maps[n] for n in ordered if n in indirect_maps},
        nside=map_nside,
        pointing=pointing,
        instrument=name,
        obs_index=obs,
    )