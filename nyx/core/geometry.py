from __future__ import annotations

import hashlib
from typing import Any

import astropy.units as u
import healpy as hp
import numpy as np
from numpy.typing import ArrayLike

from nyx.core.units import to_angle_rad, to_wavelength_nm


class Geometry:
    """Resolution and grid configuration for rendering.

    Parameters
    ----------
    wvls : array-like or astropy Quantity
        Wavelength grid (converted to nm).
    nside : int
        HEALPix nside (power of 2) for hemisphere discretisation.
    ngrid : int
        Number of grid points per axis over the FOV for in-scattering.
    fov : float or astropy Quantity
        Field of view half-angle (converted to radians).
    """

    def __init__(
        self,
        wvls: ArrayLike | u.Quantity,
        nside: int,
        ngrid: int,
        fov: float | u.Quantity,
    ) -> None:
        self.wvls = to_wavelength_nm(wvls)
        wvl_arr = np.asarray(self.wvls)
        if wvl_arr.ndim != 1 or wvl_arr.size < 2:
            raise ValueError(
                f"wvls must be a 1-D grid of at least 2 points, got shape {wvl_arr.shape}"
            )
        if not np.all(np.diff(wvl_arr) > 0):
            raise ValueError("wvls must be strictly monotonically increasing")
        if not hp.isnsideok(nside, nest=True):
            raise ValueError(f"nside must be a positive power of two, got {nside}")
        if ngrid < 2:
            # The FOV grid is sampled with a hat kernel between neighbouring
            # nodes, which needs a spacing to be defined at all.
            raise ValueError(f"ngrid must be at least 2, got {ngrid}")
        self.nside = nside
        self.ngrid = ngrid
        self.fov = to_angle_rad(fov)
        if not self.fov > 0:
            raise ValueError(f"fov must be positive, got {float(self.fov)} rad")

        # HEALPix hemisphere grid
        npix = hp.nside2npix(nside)
        theta, phi = hp.pix2ang(nside, np.arange(npix))
        self.mask = theta < np.pi / 2
        self.lon = phi[self.mask]  # azimuth, rad
        self.lat = np.pi / 2 - theta[self.mask]  # altitude, rad
        self.nsky = int(np.sum(self.mask))

        # FOV evaluation grid, laid out lon-major: axis 0 runs along
        # longitude and axis 1 along latitude, the one grid convention
        # nyx uses (see nyx.instrument._interpolation).
        grid_1d = np.linspace(-self.fov, self.fov, ngrid)
        self.X, self.Y = np.meshgrid(grid_1d, grid_1d, indexing="ij")

        self.pixel_area = hp.nside2pixarea(nside)

    @property
    def signature(self) -> tuple[Any, ...]:
        """Exact identity of this geometry, compared by
        :func:`check_shared_geometry`.

        The wavelength grid is hashed rather than stored, keeping the
        signature small while still separating two grids that share their
        endpoints and length.
        """
        wvls = np.asarray(self.wvls, dtype=np.float64)
        digest = hashlib.blake2b(wvls.tobytes(), digest_size=8).hexdigest()
        return (int(self.nside), int(self.ngrid), float(self.fov), int(wvls.size), digest)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Geometry):
            return NotImplemented
        return self.signature == other.signature

    def __hash__(self) -> int:
        return hash(self.signature)

    def __repr__(self) -> str:
        wvls = np.asarray(self.wvls)
        return (
            f"Geometry(nside={self.nside}, ngrid={self.ngrid}, "
            f"fov={np.degrees(float(self.fov)):.4g} deg, "
            f"wvls={wvls.size} points {wvls[0]:.4g}-{wvls[-1]:.4g} nm, "
            f"nsky={self.nsky})"
        )


def _describe(signature: tuple[Any, ...]) -> str:
    """Readable form of a :attr:`~nyx.core.geometry.Geometry.signature`."""
    nside, ngrid, fov, n_wvl, digest = signature
    return (
        f"nside={nside}, ngrid={ngrid}, fov={np.degrees(fov):.4g} deg, "
        f"wvls={n_wvl} points (#{digest[:8]})"
    )


def check_shared_geometry(
    obs_list: dict[str, Any],
    atmosphere: Any,
    instruments: dict[str, Any],
    sources: dict[str, Any],
) -> None:
    """Raise unless every component was built against the same Geometry.

    Components cache geometry-derived constants at construction, so
    mixing geometries passes every shape check and silently computes the
    wrong number.  A component recording no signature is skipped: an
    unchecked component is not an error.
    """
    labelled: list[tuple[str, tuple[Any, ...]]] = [
        (f"observation {name!r}", obs.geom.signature) for name, obs in obs_list.items()
    ]
    labelled.append(("atmosphere", getattr(atmosphere, "_geo_signature", None)))
    labelled += [
        (f"instrument {n!r}", getattr(i, "_geo_signature", None)) for n, i in instruments.items()
    ]
    labelled += [(f"source {n!r}", getattr(s, "_geo_signature", None)) for n, s in sources.items()]

    known = [(label, sig) for label, sig in labelled if sig is not None]
    _, reference = known[0]
    odd = [(label, sig) for label, sig in known if sig != reference]
    if not odd:
        return
    lines = [f"  {label}: {_describe(sig)}" for label, sig in [known[0], *odd]]
    raise ValueError(
        "components were built against different Geometry objects, which would "
        "silently render the wrong result (each caches geometry-derived "
        "constants at construction):\n"
        + "\n".join(lines)
        + "\nBuild every instrument, atmosphere, emitter and Observation from one Geometry."
    )
