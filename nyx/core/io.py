from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import astropy.units as u
import h5py
import numpy as np
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time

from nyx.core.geometry import Geometry
from nyx.core.observation import Observation
from nyx.core.paramtree import dump_params

if TYPE_CHECKING:
    from nyx.core.scene import Scene


__all__ = ["FitResult", "ObservationRecord", "save_fit", "load_fit"]


def _dump_times(group: h5py.Group, times: Time) -> None:
    group.create_dataset("times", data=np.asarray(times.utc.isot, dtype="S"))
    group["times"].attrs["scale"] = "utc"
    group["times"].attrs["format"] = "isot"


def _load_times(group: h5py.Group) -> Time:
    return Time(
        [b.decode() for b in group["times"][...]],
        format=group["times"].attrs.get("format", "isot"),
        scale=group["times"].attrs.get("scale", "utc"),
    )


def _dump_target(group: h5py.Group, target: SkyCoord) -> None:
    icrs = target.icrs
    group.attrs["target_ra_rad"] = float(icrs.ra.rad)
    group.attrs["target_dec_rad"] = float(icrs.dec.rad)


def _load_target(group: h5py.Group) -> SkyCoord:
    return SkyCoord(
        ra=float(group.attrs["target_ra_rad"]) * u.rad,
        dec=float(group.attrs["target_dec_rad"]) * u.rad,
        frame="icrs",
    )


def _dump_location(group: h5py.Group, location: EarthLocation) -> None:
    group.attrs["location_lon_deg"] = float(location.lon.deg)
    group.attrs["location_lat_deg"] = float(location.lat.deg)
    group.attrs["location_height_m"] = float(location.height.to(u.m).value)


def _load_location(group: h5py.Group) -> EarthLocation:
    return EarthLocation.from_geodetic(
        lon=float(group.attrs["location_lon_deg"]) * u.deg,
        lat=float(group.attrs["location_lat_deg"]) * u.deg,
        height=float(group.attrs["location_height_m"]) * u.m,
    )


def _dump_geom(group: h5py.Group, geom: Geometry) -> None:
    g = group.create_group("geom")
    g.create_dataset("wvls_nm", data=np.asarray(geom.wvls))
    g.attrs["nside"] = int(geom.nside)
    g.attrs["ngrid"] = int(geom.ngrid)
    g.attrs["fov_rad"] = float(geom.fov)


def _load_geom(group: h5py.Group) -> Geometry:
    g = group["geom"]
    return Geometry(
        wvls=np.asarray(g["wvls_nm"]) * u.nm,
        nside=int(g.attrs["nside"]),
        ngrid=int(g.attrs["ngrid"]),
        fov=float(g.attrs["fov_rad"]) * u.rad,
    )


def _dump_refract(group: h5py.Group, refract_pointing: bool) -> None:
    group.attrs["refract_pointing"] = bool(refract_pointing)


def _load_refract(group: h5py.Group) -> bool:
    return bool(group.attrs.get("refract_pointing", False))


def _dump_altaz_kwargs(group: h5py.Group, altaz_kwargs: dict[str, Any] | None) -> None:
    if not altaz_kwargs:
        return
    gak = group.create_group("altaz_kwargs")
    for k, v in altaz_kwargs.items():
        # A value may be a plain Python value or an astropy Quantity; the unit
        # goes in a sibling attribute so the load side can rebuild the Quantity.
        if hasattr(v, "unit") and hasattr(v, "value"):
            gak.attrs[k] = float(v.value)
            gak.attrs[f"{k}__unit"] = str(v.unit)
        else:
            gak.attrs[k] = v


def _load_altaz_kwargs(group: h5py.Group) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "altaz_kwargs" not in group:
        return out
    gak = group["altaz_kwargs"]
    for k in gak.attrs:
        if k.endswith("__unit"):
            continue
        v = gak.attrs[k]
        unit_key = f"{k}__unit"
        if unit_key in gak.attrs:
            v = float(v) * u.Unit(gak.attrs[unit_key])
        out[k] = v
    return out


# One row per stored field
_OBSERVATION_SCHEMA: tuple[
    tuple[str, str, Callable[[h5py.Group, Any], None], Callable[[h5py.Group], Any]], ...
] = (
    ("times", "times", _dump_times, _load_times),
    ("target", "target_icrs", _dump_target, _load_target),
    ("location", "location", _dump_location, _load_location),
    ("geom", "geom", _dump_geom, _load_geom),
    ("refract_pointing", "_refract_pointing", _dump_refract, _load_refract),
    ("altaz_kwargs", "_altaz_kwargs", _dump_altaz_kwargs, _load_altaz_kwargs),
)


def _dump_observation(group: h5py.Group, obs: Observation) -> None:
    """Write the primary inputs of *obs* into an open h5py *group*."""
    for _field, attr, dump, _load in _OBSERVATION_SCHEMA:
        dump(group, getattr(obs, attr))


def _load_observation_record(group: h5py.Group) -> ObservationRecord:
    """Read an :class:`ObservationRecord` from an open h5py *group*."""
    return ObservationRecord(**{f: load(group) for f, _a, _dump, load in _OBSERVATION_SCHEMA})


@dataclass
class ObservationRecord:
    """The inputs needed to reconstruct an :class:`Observation`.

    Pointing matrices, scattering angles and frame caches are not stored.
    """

    times: Time
    target: SkyCoord
    location: EarthLocation
    geom: Geometry
    refract_pointing: bool = False
    altaz_kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def nobs(self) -> int:
        return len(self.times)


@dataclass
class FitResult:
    """Loaded fit bundle: parameter values plus per-instrument metadata.

    Attributes
    ----------
    params : dict of str to numpy.ndarray
        One entry per Parameter of the saved Scene, keyed as
        :meth:`~nyx.core.scene.Scene.set_params` expects.
    observations : dict of str to ObservationRecord
        Keyed by instrument name.
    """

    params: dict[str, np.ndarray]
    observations: dict[str, ObservationRecord]

    def __repr__(self) -> str:
        return (
            f"FitResult(params={len(self.params)} entries, observations={list(self.observations)})"
        )


def save_fit(
    path: str | os.PathLike[str], scene: Scene, observations: dict[str, Observation]
) -> None:
    """Write a fit bundle to *path* in HDF5.

    Parameters
    ----------
    path : str or path-like
    scene : Scene
        Fitted scene; its Parameter values go to ``/params``.
    observations : dict of str to Observation
        The same dict passed to :meth:`~nyx.core.scene.Scene.build`, which
        the Scene itself does not retain.

    Raises
    ------
    KeyError
        If the keys do not match the scene's instrument names.
    """
    inst_names = set(scene.instruments)
    obs_names = set(observations)
    if inst_names != obs_names:
        raise KeyError(
            f"observations keys {sorted(obs_names)} do not match "
            f"scene instruments {sorted(inst_names)}"
        )

    with h5py.File(path, "w") as f:
        gp = f.create_group("params")
        for k, v in dump_params(scene).items():
            gp.create_dataset(k, data=np.asarray(v))
        go = f.create_group("observations")
        for name, obs in observations.items():
            _dump_observation(go.create_group(name), obs)


def load_fit(path: str | os.PathLike[str]) -> FitResult:
    """Read a fit bundle written by :func:`save_fit`; no Scene is reconstructed.

    Parameters
    ----------
    path : str or path-like

    Returns
    -------
    FitResult
    """
    with h5py.File(path, "r") as f:
        params = {k: f["params"][k][...] for k in f["params"]}
        observations = {
            name: _load_observation_record(f["observations"][name]) for name in f["observations"]
        }
    return FitResult(params=params, observations=observations)
