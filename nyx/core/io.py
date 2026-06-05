from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import astropy.units as u
import h5py
import numpy as np
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time

from nyx.core.geometry import Geometry
from nyx.core.observation import Observation
from nyx.core.parameter import dump_params

if TYPE_CHECKING:
    from nyx.core.scene import Scene


__all__ = ["FitResult", "ObservationRecord", "save_fit", "load_fit"]


# Observation <-> HDF5 group


def _dump_observation(group: h5py.Group, obs: Observation) -> None:
    """Write *obs*'s primary inputs into an open h5py *group*."""
    times_iso = np.asarray(obs.times.utc.isot, dtype="S")
    group.create_dataset("times", data=times_iso)
    group["times"].attrs["scale"] = "utc"
    group["times"].attrs["format"] = "isot"

    icrs = obs.target_icrs.icrs
    group.attrs["target_ra_rad"] = float(icrs.ra.rad)
    group.attrs["target_dec_rad"] = float(icrs.dec.rad)

    loc = obs.location
    group.attrs["location_lon_deg"] = float(loc.lon.deg)
    group.attrs["location_lat_deg"] = float(loc.lat.deg)
    group.attrs["location_height_m"] = float(loc.height.to(u.m).value)

    g = group.create_group("geom")
    g.create_dataset("wvls_nm", data=np.asarray(obs.geom.wvls))
    g.attrs["nside"] = int(obs.geom.nside)
    g.attrs["ngrid"] = int(obs.geom.ngrid)
    g.attrs["fov_rad"] = float(obs.geom.fov)

    group.attrs["refract_pointing"] = bool(getattr(obs, "_refract_pointing", False))

    # AltAz refraction kwargs (pressure, temperature, ...).  Each may be
    # a plain Python value or an astropy Quantity; we store the unit
    # string as a sibling attribute when present.
    altaz_kwargs = getattr(obs, "_altaz_kwargs", {}) or {}
    if altaz_kwargs:
        gak = group.create_group("altaz_kwargs")
        for k, v in altaz_kwargs.items():
            if hasattr(v, "unit") and hasattr(v, "value"):
                gak.attrs[k] = float(v.value)
                gak.attrs[f"{k}__unit"] = str(v.unit)
            else:
                gak.attrs[k] = v


def _load_observation_record(group: h5py.Group) -> ObservationRecord:
    """Read an :class:`ObservationRecord` from an open h5py *group*."""
    times = Time(
        [b.decode() for b in group["times"][...]],
        format=group["times"].attrs.get("format", "isot"),
        scale=group["times"].attrs.get("scale", "utc"),
    )
    target = SkyCoord(
        ra=float(group.attrs["target_ra_rad"]) * u.rad,
        dec=float(group.attrs["target_dec_rad"]) * u.rad,
        frame="icrs",
    )
    location = EarthLocation.from_geodetic(
        lon=float(group.attrs["location_lon_deg"]) * u.deg,
        lat=float(group.attrs["location_lat_deg"]) * u.deg,
        height=float(group.attrs["location_height_m"]) * u.m,
    )
    g = group["geom"]
    geom = Geometry(
        wvls=np.asarray(g["wvls_nm"]) * u.nm,
        nside=int(g.attrs["nside"]),
        ngrid=int(g.attrs["ngrid"]),
        fov=float(g.attrs["fov_rad"]) * u.rad,
    )
    refract_pointing = bool(group.attrs.get("refract_pointing", False))

    altaz_kwargs: dict[str, Any] = {}
    if "altaz_kwargs" in group:
        gak = group["altaz_kwargs"]
        for k in gak.attrs:
            if k.endswith("__unit"):
                continue
            v = gak.attrs[k]
            unit_key = f"{k}__unit"
            if unit_key in gak.attrs:
                v = float(v) * u.Unit(gak.attrs[unit_key])
            altaz_kwargs[k] = v

    return ObservationRecord(
        times=times,
        target=target,
        location=location,
        geom=geom,
        refract_pointing=refract_pointing,
        altaz_kwargs=altaz_kwargs,
    )


# Lightweight analysis containers


@dataclass
class ObservationRecord:
    """Lightweight record of an :class:`Observation`'s primary inputs.

    Holds only the inputs needed to reconstruct an Observation:
    Pointing matrices, scattering angles, and frame caches are not
    stored.
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

    def to_observation(self) -> Observation:
        """Rebuild a full :class:`Observation` from this record."""
        return Observation(
            location=self.location,
            times=self.times,
            target=self.target,
            geom=self.geom,
            refract_pointing=self.refract_pointing,
            **self.altaz_kwargs,
        )


@dataclass
class FitResult:
    """Loaded fit bundle: parameter values plus per-instrument metadata.

    Attributes
    ----------
    params : dict[str, ndarray]
        ``{dotted_path: value}`` for every Parameter in the saved Scene.
        Keys match :meth:`Scene.set_params`.
    observations : dict[str, ObservationRecord]
        ``{instrument_name: ObservationRecord}`` for every instrument
        that contributed to the fit.
    """

    params: dict[str, np.ndarray]
    observations: dict[str, ObservationRecord]

    def __repr__(self) -> str:
        return (
            f"FitResult(params={len(self.params)} entries, observations={list(self.observations)})"
        )


# Top-level save / load


def save_fit(
    path: str | os.PathLike[str], scene: Scene, observations: dict[str, Observation]
) -> None:
    """Write a fit bundle to *path* in HDF5.

    Parameters
    ----------
    path : str or path-like
        Destination filename.
    scene : Scene
        Fitted scene; its Parameter values are written to ``/params``.
    observations : dict of {name: Observation}
        The same dict passed to :meth:`Scene.build`.  Required because
        Scene retains only the precomputed JAX pytrees, not the
        original astropy objects needed to round-trip.

    Raises
    ------
    KeyError
        If ``observations`` keys do not match the scene's instrument
        names.
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
    """Read a fit bundle written by :func:`save_fit` for analysis.

    The result is a :class:`FitResult` with ``params`` (a plain
    ``{path: ndarray}`` dict) and ``observations`` (a dict of
    :class:`ObservationRecord`).  No Scene is reconstructed — load
    the originals into a fresh Scene via :meth:`Scene.load_params`
    if you want to render again.

    Parameters
    ----------
    path : str or path-like
        Path to a bundle written by :func:`save_fit`.
    """
    with h5py.File(path, "r") as f:
        params = {k: f["params"][k][...] for k in f["params"]}
        observations = {
            name: _load_observation_record(f["observations"][name]) for name in f["observations"]
        }
    return FitResult(params=params, observations=observations)
