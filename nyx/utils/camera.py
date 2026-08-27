"""Camera-plane display of per-pixel quantities.

A thin matplotlib helper that draws one value per instrument pixel as a
hexagon at that pixel's position in the field of view.  It is the natural
view for anything shaped like ``instrument.pixel_efficiency`` or a single
row of ``scene.render()``::

    coll = camera_image(ax, scene.instrument.centers, rates[0])
    fig.colorbar(coll, ax=ax)

The returned collection is reusable: ``coll.set_array(new_values)``
redraws it without rebuilding the geometry, which is what
:func:`nyx.utils.convergence.record_fit`-driven animations rely on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import PolyCollection

__all__ = ["camera_image"]

#: Multiplier from radians to each supported display unit.
_UNIT_SCALE = {
    "rad": 1.0,
    "deg": 180.0 / np.pi,
    "arcmin": 60.0 * 180.0 / np.pi,
    "arcsec": 3600.0 * 180.0 / np.pi,
}


def _hex_lattice(xy: np.ndarray) -> tuple[float, float]:
    """Pitch and orientation of the hexagonal lattice *xy* lies on.

    Parameters
    ----------
    xy : np.ndarray, shape (n_pix, 2)
        Pixel centres.

    Returns
    -------
    pitch : float
        Median centre-to-centre distance between neighbouring pixels.
    angle : float
        Direction of the lattice rows, in radians, folded into
        ``[-pi/6, pi/6)`` by the six-fold symmetry of the lattice.
    """
    from scipy.spatial import cKDTree

    if len(xy) < 2:
        return 1.0, 0.0
    distance, index = cKDTree(xy).query(xy, k=2)
    pitch = float(np.median(distance[:, 1]))
    step = xy[index[:, 1]] - xy
    theta = np.arctan2(step[:, 1], step[:, 0])
    # Average the directions modulo 60 degrees: a hexagonal lattice has
    # six neighbours, and which one happens to be nearest is arbitrary.
    angle = float(np.angle(np.mean(np.exp(6j * theta))) / 6.0)
    return pitch, angle


def camera_image(
    ax: Axes,
    centers: ArrayLike,
    values: ArrayLike | None = None,
    *,
    unit: str = "deg",
    radius: float | None = None,
    pad: float = 0.03,
    vmin: float | None = None,
    vmax: float | None = None,
    **kwargs: Any,
) -> PolyCollection:
    """Draw a camera as one hexagon per pixel, coloured by *values*.

    The hexagon size and orientation are read off the pixel positions
    themselves, so any hexagonally packed camera comes out tiled with no
    gaps and no overlaps.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw into.  Its aspect ratio is set to equal and its
        limits to the camera extent.
    centers : array-like, shape (n_pix, 2)
        Pixel centres in the offset frame, in radians, as given by
        ``instrument.centers``.
    values : array-like, shape (n_pix,), optional
        Per-pixel value to colour by.  When omitted the collection is
        left unmapped, ready for a later ``set_array``.
    unit : {'deg', 'rad', 'arcmin', 'arcsec'}, optional
        Display unit for the axes (default degrees).
    radius : float, optional
        Hexagon centre-to-vertex distance, in *unit*.  Defaults to the
        lattice pitch over ``sqrt(3)``, which tiles the plane exactly.
    pad : float, optional
        Fraction of the camera width left as a margin (default 3%).
    vmin, vmax : float, optional
        Colour limits.  Fixing them keeps the scale steady while the
        values behind an animation change.
    **kwargs
        Forwarded to :class:`matplotlib.collections.PolyCollection`
        (``cmap``, ``norm``, ``edgecolors``, ...).

    Returns
    -------
    matplotlib.collections.PolyCollection
        The collection added to *ax*.  Call ``set_array`` on it to show
        new values without rebuilding the hexagons.

    Examples
    --------
    ::

        fig, ax = plt.subplots()
        coll = camera_image(ax, scene.instrument.centers, rates[0], cmap='inferno')
        fig.colorbar(coll, ax=ax, label='rate [1/s]')
    """
    from matplotlib.collections import PolyCollection

    if unit not in _UNIT_SCALE:
        raise ValueError(f"unknown unit {unit!r}; choices are {sorted(_UNIT_SCALE)}.")

    xy = np.asarray(centers, dtype=float) * _UNIT_SCALE[unit]
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"centers must have shape (n_pix, 2), got {xy.shape}.")

    pitch, angle = _hex_lattice(xy)
    if radius is None:
        radius = pitch / np.sqrt(3.0)

    # Vertices sit half a sector away from the lattice rows, so each
    # hexagon presents a flat edge to its neighbour.
    corners = angle + np.pi / 6 + np.arange(6) * np.pi / 3
    offsets = radius * np.stack([np.cos(corners), np.sin(corners)], axis=-1)
    collection = PolyCollection(xy[:, None, :] + offsets[None, :, :], **kwargs)

    if values is not None:
        v = np.asarray(values, dtype=float).ravel()
        if v.size != xy.shape[0]:
            raise ValueError(f"values has {v.size} entries but there are {xy.shape[0]} pixels.")
        collection.set_array(v)
    if vmin is not None or vmax is not None:
        collection.set_clim(vmin, vmax)

    ax.add_collection(collection)
    ax.set_aspect("equal")
    lo, hi = xy.min(axis=0) - radius, xy.max(axis=0) + radius
    mid = (lo + hi) / 2
    half = float(np.max(hi - lo)) / 2 * (1.0 + pad)
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    return collection
