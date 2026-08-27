from __future__ import annotations

import warnings

import numpy as np

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

    _warn_on_mismatch(geo, table.wavelengths, response_centroid(lattice, values))
    return EffectiveApertureInstrument(geo, bandpass, lattice, values)


def _warn_on_mismatch(geo, wavelengths, centers):
    """Flag a Geometry that does not cover what the table describes.
    """
    wvls = np.asarray(geo.wvls, dtype=float)
    lo, hi = float(wavelengths[0]), float(wavelengths[-1])
    if wvls.min() < lo or wvls.max() > hi:
        warnings.warn(
            f"geo.wvls spans {wvls.min():.1f}-{wvls.max():.1f} nm but the instrument "
            f"bandpass is only tabulated over {lo:.1f}-{hi:.1f} nm; outside that range "
            f"the bandpass reads zero, so those bins contribute nothing",
            UserWarning,
            stacklevel=4,
        )

    reach = float(np.abs(np.asarray(centers)).max())
    if reach > float(geo.fov):
        warnings.warn(
            f"pixels reach {np.degrees(reach):.2f} deg off axis but geo.fov is "
            f"{np.degrees(float(geo.fov)):.2f} deg; the in-scattering grid does not cover "
            f"the camera, so outer pixels will receive no scattered light",
            UserWarning,
            stacklevel=4,
        )