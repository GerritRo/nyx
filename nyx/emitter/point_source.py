from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
from astropy.coordinates import SkyCoord

from nyx.core.records import PerObs, SourceObsData
from nyx.emitter.base import Emitter
from nyx.emitter.catalogs.astrometry import altaz_track
from nyx.spectra import SpectralModel, StoredSpectrum, blackbody_photon_flux

__all__ = ["PointSource"]


class PointSource(Emitter):
    """One point source at a fixed ICRS position.

    Parameters
    ----------
    geo : Geometry
        Resolution configuration.
    coord : astropy.coordinates.SkyCoord
        Scalar ICRS position, transformed to AltAz per observation. Proper
        motion is applied when the coordinate carries it.
    spectrum : SpectralModel or array-like, optional
        Spectrum on ``geo.wvls``, in ``photon / s / m^2 / nm``. An array is
        wrapped as a :class:`~nyx.spectra.StoredSpectrum`. Defaults to
        a black body at *mag* and *temperature_k*.
    mag : float
        Magnitude of the default black body; ignored if *spectrum* is given.
    temperature_k : float
        Temperature of the default black body.
    brightness : array-like
        Multiplier on the spectrum, and the source's one free parameter.  A
        scalar is one brightness for the run, ``(nobs,)`` a light curve, and
        ``(nobs, n_wvl)`` a chromatic one.
    transform : str or None
        Domain of *brightness*, ``'log'`` by default.
    inscatter : bool
        Whether to compute the source's own scattered halo.
    """

    def __init__(
        self,
        geo: Any,
        coord: SkyCoord,
        spectrum: SpectralModel | Any = None,
        *,
        mag: float = 9.0,
        temperature_k: float = 9000.0,
        brightness: Any = 1.0,
        transform: str | None = "log",
        inscatter: bool = False,
    ) -> None:
        if not coord.isscalar:
            raise TypeError(
                f"PointSource takes one position; got a SkyCoord of shape {coord.shape}. "
                f"Use one PointSource per source, or an emitter with a catalog."
            )
        n_wvl = int(np.size(np.asarray(geo.wvls)))

        if spectrum is None:
            spectrum = blackbody_photon_flux(geo.wvls, temperature_k, mag)
        if not isinstance(spectrum, SpectralModel):
            values = jnp.atleast_2d(jnp.asarray(spectrum))
            if values.shape != (1, n_wvl):
                raise ValueError(
                    f"spectrum must be one spectrum on geo.wvls ({n_wvl} points), "
                    f"got shape {tuple(jnp.shape(spectrum))}"
                )
            spectrum = StoredSpectrum(spectra=values)

        super().__init__(geo, spectrum, brightness, transform)
        self._coord = coord
        self._inscatter = bool(inscatter)

    def _prepare(self, obs: Any) -> SourceObsData:
        """Transform the position into each observation's AltAz frame.

        Parameters
        ----------
        obs : Observation

        Returns
        -------
        SourceObsData
        """
        coords = altaz_track(self._coord, obs.times, obs.altaz_frames)  # (nobs, 1, 2)
        return SourceObsData(
            source_coords=PerObs(jnp.asarray(coords)),
            inscatter=self._inscatter,
        )

    def _repr_parts(self) -> list[str]:
        icrs = self._coord.icrs
        shape = () if self._brightness is None else tuple(np.shape(self._brightness.factor))
        return [
            f"ra={icrs.ra.deg:.4f} dec={icrs.dec.deg:.4f} deg",
            "constant" if not shape else f"brightness {shape}",
            *(["inscatter"] if self._inscatter else []),
        ]
