"""The per-emitter maps a render produces, and how to read them."""

from __future__ import annotations

import dataclasses
from typing import Any

import healpy as hp
import numpy as np

from nyx.view.response import LUMA, SpectralResponse, to_linear_srgb

__all__ = ["PointField", "SkyRender"]


@dataclasses.dataclass(frozen=True)
class PointField:
    """Point sources of one emitter, as they reach the telescope.

    Attributes
    ----------
    az, alt : np.ndarray, shape (n,)
        Positions in radians.
    flux : np.ndarray, shape (n, k)
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
        if self.response.n_channels == 1:
            return "photon / s / sr"
        # Only a colorimetric response actually carries tristimulus values;
        # a set of measured filter curves carries its own.
        return "tristimulus / sr" if self.response.to_srgb is not None else "response / sr"

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
        """Summed diffuse map, ``(npix, k)``.

        Parameters
        ----------
        sources : list of str or str, optional
            Restrict to these emitters (default: all).
        direct, inscatter : bool
            Which of the two paths to include.
        """
        total = np.zeros((hp.nside2npix(self.nside), self.response.n_channels))
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
            return PointField(np.zeros(0), np.zeros(0), np.zeros((0, self.response.n_channels)))
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
        binned = _bin_points(field, self.nside, self.response.n_channels)
        return binned / hp.nside2pixarea(self.nside)

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
        A one-channel response already is one; anything that can be shown
        as colour becomes luminance; anything else is summed.
        """
        if self.response.n_channels == 1:
            return np.asarray(values)[..., 0]
        # Follow SpectralResponse.colour, so a three-channel response that
        # is already sRGB-like (``to_srgb=None``) is reduced to luminance
        # rather than to a meaningless channel sum.
        if self.response.colour:
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


def _bin_points(field: PointField, nside: int, n_channels: int) -> np.ndarray:
    """Point fluxes accumulated into HEALPix pixels, ``(npix, k)``."""
    binned = np.zeros((hp.nside2npix(nside), n_channels))
    if field.az.size:
        pix = hp.ang2pix(nside, np.pi / 2 - field.alt, field.az)
        np.add.at(binned, pix, field.flux)
    return binned
