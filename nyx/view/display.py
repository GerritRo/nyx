from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import healpy as hp
import numpy as np

from nyx.view.response import LUMA, srgb_decode, srgb_encode

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from nyx.view.allsky import SkyRender

__all__ = ["ToneCurve", "add_noise", "plot_maps", "tonemap"]


# Tone mapping


@dataclasses.dataclass(frozen=True)
class ToneCurve:
    """How a linear image is turned into something a screen can show.

    A night sky spans many decades between a dark patch of sky and the
    Moon, so a linear image shows either the background or the stars and
    never both.  The curve is applied to luminance and all three
    channels are scaled by the same factor, which compresses the range
    without draining the colour out of it.

    The point of it being an object rather than a function is that one
    curve can be fitted once and applied to every frame of a series.
    Re-fitting each partial image would rescale every one of them to
    itself, and a sky with one component in it would look exactly as
    bright as a sky with five -- which is the one thing a
    component-by-component figure must not do.

    Parameters
    ----------
    gain : float
        Exposure: what the linear image is multiplied by before the
        curve.  :meth:`fit` solves for it.
    stretch : {'asinh', 'reinhard', 'linear'}
        ``'asinh'`` is the usual astronomical stretch -- linear where
        the signal is small, logarithmic where it is large.
        ``'reinhard'`` is the photographic ``x / (1 + x)``.
    softness : float
        Where the asinh stretch turns over, in units of the clipping
        level.  Smaller lifts the faint sky harder relative to the
        stars.
    saturation : float
        Colour saturation multiplier.  Night photographs are usually
        pushed a little above 1.
    black : float
        Level subtracted after the gain: lifts a washed-out background
        off the floor at the cost of clipping the darkest sky.
    gamma : bool
        Apply the sRGB transfer function (default).  Turn it off only
        when the consumer does its own encoding.

    Examples
    --------
    ::

        full = cam.expose(sky)
        curve = ToneCurve.fit(full, level=0.15, saturation=1.3)
        plt.imshow(curve(cam.expose(sky, ['airglow'])))
        plt.imshow(curve(full))
    """

    gain: float = 1.0
    stretch: str = "asinh"
    softness: float = 0.06
    saturation: float = 1.0
    black: float = 0.0
    gamma: bool = True

    def __post_init__(self) -> None:
        if self.stretch not in ("asinh", "reinhard", "linear"):
            raise ValueError(
                f"stretch must be 'asinh', 'reinhard' or 'linear', not {self.stretch!r}"
            )

    @classmethod
    def fit(
        cls,
        image: np.ndarray,
        *,
        level: float = 0.15,
        percentile: float = 50.0,
        **kwargs: Any,
    ) -> ToneCurve:
        """Solve for the gain that puts *image* where you want it.

        Parameters
        ----------
        image : np.ndarray
            Linear image to anchor on -- normally the complete one, so
            that every partial frame is measured against the full sky.
        level : float
            Display value, 0 to 1, the anchor should come out at.  0.15
            is a night sky that reads as night: dark, but not black.
        percentile : float
            Percentile of the non-zero luminance to anchor.  The default
            of 50 anchors the sky background itself; stars are a
            fraction of a per cent of the pixels and do not move it.
        **kwargs
            Any other field of :class:`ToneCurve`.  They are part of the
            fit -- the gain depends on the stretch it is solved against.

        Returns
        -------
        ToneCurve
        """
        curve = cls(**kwargs)
        luma = np.asarray(image, dtype=float) @ LUMA
        values = luma[luma > 0]
        if values.size == 0:
            return curve
        reference = float(np.percentile(values, percentile))
        if reference <= 0:
            return curve
        target = srgb_decode(level) if curve.gamma else level
        gain = (curve._unstretch(target) + curve.black) / reference
        return dataclasses.replace(curve, gain=float(gain))

    def _stretch(self, luma: np.ndarray) -> np.ndarray:
        if self.stretch == "asinh":
            return np.asarray(np.arcsinh(luma / self.softness) / np.arcsinh(1.0 / self.softness))
        if self.stretch == "reinhard":
            return np.asarray(luma / (1.0 + luma))
        return luma

    def _unstretch(self, value: float) -> float:
        if self.stretch == "asinh":
            return float(self.softness * np.sinh(value * np.arcsinh(1.0 / self.softness)))
        if self.stretch == "reinhard":
            return value / (1.0 - value) if value < 1.0 else np.inf
        return value

    def __call__(self, image: np.ndarray, *, alpha: np.ndarray | None = None) -> np.ndarray:
        """Apply the curve.

        Parameters
        ----------
        image : np.ndarray, shape (h, w, 3)
            Linear sRGB from :meth:`Camera.expose`.
        alpha : np.ndarray of bool, optional
            Opacity mask, e.g. :meth:`Camera.horizon`.  When given, an
            RGBA image comes back -- ready to composite behind a
            foreground.

        Returns
        -------
        np.ndarray of uint8, shape (h, w, 3) or (h, w, 4)
        """
        scaled = np.asarray(image, dtype=float) * self.gain
        luma = np.clip(scaled @ LUMA - self.black, 0.0, None)
        factor = np.divide(self._stretch(luma), luma, out=np.zeros_like(luma), where=luma > 0)
        rgb = scaled * factor[..., None]
        if self.saturation != 1.0:
            grey = (rgb @ LUMA)[..., None]
            rgb = grey + (rgb - grey) * self.saturation
        rgb = np.clip(rgb, 0.0, 1.0)
        if self.gamma:
            rgb = srgb_encode(rgb)

        out = np.clip(np.rint(rgb * 255), 0, 255).astype(np.uint8)
        if alpha is None:
            return out
        opacity = np.rint(np.asarray(alpha, dtype=float) * 255).astype(np.uint8)
        return np.concatenate([out, opacity[..., None]], axis=-1)


def tonemap(image: np.ndarray, *, alpha: np.ndarray | None = None, **kwargs: Any) -> np.ndarray:
    """Fit a :class:`ToneCurve` to *image* and apply it, in one call.

    Convenient for a single look at a single frame.  For a series --
    the same sky with one component after another added to it -- fit one
    curve with :meth:`ToneCurve.fit` and reuse it, or the frames will
    not be comparable.

    Parameters
    ----------
    image : np.ndarray, shape (h, w, 3)
        Linear sRGB from :meth:`Camera.expose`.
    alpha : np.ndarray of bool, optional
        Opacity mask; see :meth:`ToneCurve.__call__`.
    **kwargs
        Passed to :meth:`ToneCurve.fit` (``level``, ``percentile``, and
        any :class:`ToneCurve` field).

    Returns
    -------
    np.ndarray of uint8
    """
    return ToneCurve.fit(image, **kwargs)(image, alpha=alpha)


def add_noise(
    image: np.ndarray,
    *,
    snr: float = 25.0,
    read: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """Shot and read noise, for a sensor that is not infinitely deep.

    A rendered sky is perfectly smooth, which is the one thing that
    gives it away as a render; this puts the grain of a real exposure
    back.  The noise is drawn on luminance and the colour of each pixel
    is left alone, which is what a demosaiced photograph looks like --
    per-channel noise would come out as coloured confetti.

    Parameters
    ----------
    image : np.ndarray
        Linear image from :meth:`Camera.expose`.
    snr : float
        Signal-to-noise of the sky background in a single pixel.  This
        is the exposure written as something you can judge by eye: 30 or
        more is a clean tracked exposure, 10 is a short hand-held one.
    read : float
        Gaussian read noise, as a fraction of the sky background.
    seed : int, optional
        Seed, for grain that does not change between runs.

    Returns
    -------
    np.ndarray
        Linear image with noise, same shape and scale as the input.
    """
    rng = np.random.default_rng(seed)
    values = np.clip(np.asarray(image, dtype=float), 0.0, None)
    luma = values @ LUMA
    lit = luma[luma > 0]
    if lit.size == 0 or snr <= 0:
        return values
    background = float(np.median(lit))
    # Poisson counts scaled so that the background lands at the wanted S/N.
    scale = snr**2 / background
    noisy = rng.poisson(luma * scale) / scale
    if read > 0:
        noisy = noisy + rng.normal(0.0, read * background, size=noisy.shape)
    factor = np.divide(noisy, luma, out=np.ones_like(luma), where=luma > 0)
    return values * np.clip(factor, 0.0, None)[..., None]


# Map plotting


def _panels(
    render: SkyRender, kind: str, sources: list[str] | None
) -> tuple[list[tuple[str, np.ndarray]], int]:
    """Panel titles, maps and column count for a plot *kind*."""
    names = render.sources if sources is None else sources
    scalar = render.scalar
    if kind == "components":
        return [
            ("direct", scalar(render.direct_map(names))),
            ("indirect", scalar(render.indirect_map(names))),
            ("total", scalar(render.total(names))),
        ], 3
    if kind == "sources":
        panels = [(name, scalar(render.total([name]))) for name in names]
        if len(names) > 1:
            panels.append(("total", scalar(render.total(names))))
        return panels, min(len(panels), 3)
    if kind == "grid":
        panels = []
        for name in [*names, "total"]:
            subset = names if name == "total" else [name]
            panels += [
                (f"{name} | direct", scalar(render.direct_map(subset))),
                (f"{name} | indirect", scalar(render.indirect_map(subset))),
                (f"{name} | total", scalar(render.total(subset))),
            ]
        return panels, 3
    raise ValueError(f"kind must be 'components', 'sources' or 'grid', not {kind!r}")


def _limits(
    render: SkyRender, maps: list[np.ndarray], log: bool, vmin: float | None, vmax: float | None
) -> tuple[float, float]:
    """Colour limits shared by every panel.

    The upper limit is the 99.9th percentile rather than the maximum: a
    point source binned into a single pixel is orders of magnitude above
    its surroundings and would otherwise flatten the sky.
    """
    values = np.concatenate([m[render.mask] for m in maps])
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


def _annotate(render: SkyRender, compass: bool) -> None:
    """Mark the pointing and the cardinal directions on the current panel."""
    az, alt = render.pointing
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
        # Orthographic radius is cos(alt), so a label near the rim has to
        # sit well up from the horizon to stay clear of it.
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


def plot_maps(
    render: SkyRender,
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

    The horizon is the rim, north is up and east is left -- the sky as an
    observer under it sees it.  All panels share one colour scale and one
    colour bar, so components can be compared by eye.

    A render of any width can be drawn: one channel is plotted directly,
    a colorimetric one as luminance (see
    :meth:`~nyx.view.allsky.SkyRender.scalar`).  Point sources are binned
    into map pixels, since a map has nowhere else to put them.

    Parameters
    ----------
    render : SkyRender
        Rendered sky, from :func:`~nyx.view.allsky.render_sky`.
    kind : {'components', 'sources', 'grid'}
        ``'components'`` shows direct, indirect and total summed over all
        emitters; ``'sources'`` one total panel per emitter; ``'grid'`` a
        full emitter x component matrix.
    sources : list of str, optional
        Restrict to these emitters (default: all).  The colour scale
        follows, which is the way to bring out a faint component that a
        bright one flattens.
    log : bool
        Logarithmic colour scale (default).  Sky brightness spans decades
        between the Moon and a dark patch.
    cmap, vmin, vmax : str, float, float
        Colour map and limits.  Limits default to the minimum and the
        99.9th percentile over every panel; values outside are clipped
        rather than dropped.
    compass, graticule : bool
        Draw cardinal-direction labels / an alt-az grid.
    figsize, fig : tuple, matplotlib Figure
        Figure size (default: scaled to the panel count) and an existing
        figure to draw into.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LogNorm, Normalize

    panels, ncols = _panels(render, kind, sources)
    nrows = -(-len(panels) // ncols)
    # Inches reserved for the figure title, the colour bar, the gap between
    # panels and the strip each panel title sits in.
    head, foot, gap, strip = 0.45, 1.0, 0.12, 0.3
    if fig is None:
        fig = plt.figure(figsize=figsize or (4.0 * ncols, 4.0 * nrows + head + foot))
    else:
        plt.figure(fig.number)  # healpy draws into the current figure
    width, height = fig.get_size_inches()

    lo, hi = _limits(render, [m for _, m in panels], log, vmin, vmax)
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
            render.masked(np.clip(values, lo, hi)),
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
        _annotate(render, compass)
        if graticule:
            # Inside the loop: healpy draws into the current axes, so one
            # call after it would rule only whichever panel came last.
            hp.graticule(dpar=30, dmer=45, color="0.7", alpha=0.5, lw=0.4)

    bar = fig.add_axes((0.3, 0.55 * foot / height, 0.4, 0.14 / height))
    fig.colorbar(
        ScalarMappable(norm=LogNorm(lo, hi) if log else Normalize(lo, hi), cmap=cmap),
        cax=bar,
        orientation="horizontal",
        label=render.unit,
    )

    az, alt = np.degrees(render.pointing)
    fig.suptitle(
        f"{render.label} | obs {render.obs_index} | pointing az {az:.1f}, alt {alt:.1f} deg",
        fontsize=11,
        y=1.0 - 0.5 * head / height,
        verticalalignment="center",
    )
    return fig
