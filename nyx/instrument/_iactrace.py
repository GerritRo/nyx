from __future__ import annotations

import warnings

import numpy as np

from nyx import NyxWarning
from nyx.instrument._interpolation import PixelLattice, response_centroid
from nyx.instrument.io import tabulated_bandpass

_MISSING = (
    "from_iactrace needs the iactrace ray tracer, which nyx does not require by "
    'default.  Install it with `pip install "nyx[iactrace]"`.'
)


def build_from_iactrace(geo, telescope, camera, **scan_kwargs):
    """Scan *telescope* + *camera* and adapt the result to an instrument."""
    try:
        from iactrace.analysis import effective_aperture
    except ImportError as exc:  # pragma: no cover - exercised only without iactrace
        raise ImportError(_MISSING) from exc

    return build_from_table(geo, effective_aperture(telescope, camera, **scan_kwargs))


def build_from_table(geo, table):
    """Adapt an ``EffectiveApertureTable`` to an instrument."""
    from nyx.instrument.effective_aperture import EffectiveApertureInstrument

    lattice = PixelLattice(
        origin=table.origin,
        step=table.step,
        offset=table.offset,
        grid_shape=table.values.shape[-2:],
    )
    if not table.on_axis_area > 0:
        raise ValueError(
            "the table's on-axis effective area is zero, so the response cannot be "
            "normalised; the telescope collects no light on axis"
        )
    values = np.asarray(table.values, dtype=np.float32) / np.float32(table.on_axis_area)
    bandpass = tabulated_bandpass(table.wavelengths, table.spectral_area)

    _warn_on_mismatch(
        geo, table.wavelengths, table.spectral_area, response_centroid(lattice, values)
    )
    return EffectiveApertureInstrument(geo, bandpass, lattice, values)


#: Edge transmission, as a fraction of peak, above which a wavelength grid
#: that stops short of the tabulated band is judged to be cutting it off.
_EDGE_TOL = 0.01


def _warn_on_mismatch(geo, wavelengths, transmission, centers, stacklevel=4):
    """Flag a Geometry that does not cover what the table describes.

    Called both when a table is scanned and when an instrument is loaded
    from disk -- the latter is the path users take every day, so the check
    has to fire there too.

    A grid wider than the table is only worth mentioning when the bandpass
    is still responding at the edge.  Tables normally stop where the
    instrument stops, so the common case costs nothing and warning about
    it would just teach users to ignore the warning.
    """
    wvls = np.asarray(geo.wvls, dtype=float)
    trx = np.abs(np.asarray(transmission, dtype=float))
    lo, hi = float(wavelengths[0]), float(wavelengths[-1])
    peak = float(trx.max()) or 1.0

    cut = [
        f"{side} {edge:.1f} nm, where it is still {100 * float(trx[index]) / peak:.1f}% of peak"
        for side, edge, index, outside in (
            ("below", lo, 0, wvls.min() < lo),
            ("above", hi, -1, wvls.max() > hi),
        )
        if outside and float(trx[index]) > _EDGE_TOL * peak
    ]
    if cut:
        warnings.warn(
            f"geo.wvls spans {wvls.min():.1f}-{wvls.max():.1f} nm but the instrument "
            f"bandpass is only tabulated over {lo:.1f}-{hi:.1f} nm and is cut off "
            + " and ".join(cut)
            + "; outside the table the bandpass reads zero, so that response is lost",
            NyxWarning,
            stacklevel=stacklevel,
        )

    reach = float(np.abs(np.asarray(centers)).max())
    if reach > float(geo.fov):
        warnings.warn(
            f"pixels reach {np.degrees(reach):.2f} deg off axis but geo.fov is "
            f"{np.degrees(float(geo.fov)):.2f} deg; the in-scattering grid does not cover "
            f"the camera, so outer pixels will receive no scattered light",
            NyxWarning,
            stacklevel=stacklevel,
        )
