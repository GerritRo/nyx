from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from nyx import NyxWarning
from nyx.instrument._interpolation import PixelLattice, response_centroid
from nyx.instrument.io import tabulated_bandpass

_MISSING = (
    "from_iactrace needs the iactrace ray tracer, which nyx does not require by "
    'default.  Install it with `pip install "nyx[iactrace]"`.'
)

#: Marker iactrace writes into an effective-aperture archive; see
#: ``iactrace.io.aperture_table`` for the format.
_TABLE_FORMAT = "iactrace-aperture-table"

#: Major version of that format this reader understands.
_TABLE_MAJOR = "1"

#: The arrays a table is made of, under their own names in the archive.
_TABLE_ARRAYS = ("origin", "step", "offset", "values", "wavelengths", "spectral_area")


class ApertureTable:
    """An iactrace effective-aperture table, read from a file.

    Field-for-field what ``iactrace.analysis.EffectiveApertureTable`` carries,
    so :func:`build_from_table` accepts either; reading one needs numpy alone.

    Attributes
    ----------
    origin : numpy.ndarray, shape (2,)
        Field offset of lattice node ``(0, 0)``, ``[lon, lat]`` in radians.
    step : numpy.ndarray, shape (2,)
        Node spacing along ``[lon, lat]``, in radians.
    offset : numpy.ndarray, shape (n_pixels, 2)
        Node index of each pixel's response window corner.
    values : numpy.ndarray, shape (n_pixels, W, W)
        Effective area at each node, in m^2.
    on_axis_area : float
        Band-averaged on-axis effective area over all pixels, in m^2; what
        :attr:`values` is normalised by.
    wavelengths : numpy.ndarray, shape (K,)
        Bandpass grid, in nm.
    spectral_area : numpy.ndarray, shape (K,)
        On-axis total effective area at each wavelength, in m^2.
    meta : dict
        Provenance recorded by the scan.
    """

    __slots__ = (
        "meta",
        "offset",
        "on_axis_area",
        "origin",
        "spectral_area",
        "step",
        "values",
        "wavelengths",
    )

    def __init__(
        self,
        origin: np.ndarray,
        step: np.ndarray,
        offset: np.ndarray,
        values: np.ndarray,
        on_axis_area: float,
        wavelengths: np.ndarray,
        spectral_area: np.ndarray,
        meta: dict,
    ):
        self.origin = origin
        self.step = step
        self.offset = offset
        self.values = values
        self.on_axis_area = on_axis_area
        self.wavelengths = wavelengths
        self.spectral_area = spectral_area
        self.meta = meta

    def __repr__(self) -> str:
        return (
            f"ApertureTable({self.values.shape[0]} pixels, "
            f"{self.values.shape[-1]}x{self.values.shape[-1]} window, "
            f"{self.wavelengths[0]:.0f}-{self.wavelengths[-1]:.0f} nm)"
        )


def load_aperture_table(path: str | Path) -> ApertureTable:
    """Read an effective-aperture table iactrace saved to ``.npz``.

    Parameters
    ----------
    path : str or path-like
        A file written by ``iactrace.io.save_aperture_table``.

    Returns
    -------
    ApertureTable

    Raises
    ------
    ValueError
        If the file is not an aperture table, or is a format version this
        reader does not know.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        if "format" not in archive or str(archive["format"]) != _TABLE_FORMAT:
            raise ValueError(
                f"{path} is not an {_TABLE_FORMAT} file; write one with "
                f"iactrace.io.save_aperture_table"
            )
        version = str(archive["format_version"])
        if version.split(".")[0] != _TABLE_MAJOR:
            raise ValueError(f"{path} is format {version}, and this nyx reads {_TABLE_MAJOR}.x")
        fields: dict[str, Any] = {name: archive[name] for name in _TABLE_ARRAYS}
        fields["on_axis_area"] = float(archive["on_axis_area"])
        fields["meta"] = json.loads(str(archive["meta"]))
    return ApertureTable(**fields)


def build_from_iactrace(geo, telescope, camera, **scan_kwargs):
    """Scan *telescope* and *camera*, and adapt the result to an instrument.

    Returns
    -------
    dict
        Keyword arguments for the instrument constructor.
    """
    try:
        from iactrace.analysis import effective_aperture
    except ImportError as exc:  # pragma: no cover - exercised only without iactrace
        raise ImportError(_MISSING) from exc

    return build_from_table(geo, effective_aperture(telescope, camera, **scan_kwargs))


def build_from_table(geo, table):
    """Adapt an effective-aperture table to an instrument.

    Parameters
    ----------
    table : EffectiveApertureTable, ApertureTable, str or path-like
        A path is read here, so no iactrace import is involved.
    geo : Geometry

    Returns
    -------
    dict
        Keyword arguments for the instrument constructor.
    """
    from nyx.instrument.effective_aperture import EffectiveApertureInstrument

    if isinstance(table, str | Path):
        table = load_aperture_table(table)

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
    """Warn about a Geometry that does not cover what the table describes.

    A grid wider than the table is flagged only where the bandpass is still
    responding at its edge.
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
