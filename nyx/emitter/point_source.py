"""A single point source at a fixed sky position."""

from __future__ import annotations

from typing import Any

import astropy.units as u
import jax
import jax.numpy as jnp
import numpy as np
from astropy.coordinates import SkyCoord

from nyx.core.parameter import Parameter
from nyx.core.protocols import PointSourceData, SourceModel, SourceObsData
from nyx.core.spectral import SpectralModel, StoredSpectrum
from nyx.emitter._base import BaseEmitter

__all__ = ["PointSource", "VariableSource", "blackbody_photon_flux"]

#: Photon flux at 550 nm of a magnitude-zero source, ``photon / s / m^2 / nm``.
#: Sets what "magnitude" means here; any consistent choice cancels in a ratio.
V_ZERO_POINT = 1.0e8


def blackbody_photon_flux(wvls: Any, temperature_k: float, mag: float = 0.0) -> jax.Array:
    """Black-body photon flux density, ``photon / s / m^2 / nm``.

    Normalised to *mag* against :data:`V_ZERO_POINT` at 550 nm.
    """
    from astropy.constants import c, h, k_B

    lam_nm = np.asarray(wvls.to(u.nm).value if isinstance(wvls, u.Quantity) else wvls, dtype=float)
    lam_m = lam_nm * 1e-9
    # Photon (not energy) radiance, so one power of lambda less than Planck.
    exponent = h.value * c.value / (lam_m * k_B.value * float(temperature_k))
    shape = 1.0 / (lam_m**4 * np.expm1(exponent))
    return jnp.asarray(V_ZERO_POINT * 10 ** (-0.4 * mag) * shape / np.interp(550.0, lam_nm, shape))


class VariableSource(SourceModel):
    """A source whose brightness is a parameter: one value per frame."""

    brightness: Parameter

    def point_sources(self, obs_data: SourceObsData | None = None) -> PointSourceData | None:
        points = super().point_sources(obs_data)
        if points is None:
            return None
        return PointSourceData(
            spectra=self.brightness.value * points.spectra,
            coords=points.coords,
        )


class PointSource(BaseEmitter):
    """One point source at a fixed ICRS position.

    Parameters
    ----------
    geo : Geometry
        Resolution configuration.
    coord : astropy.coordinates.SkyCoord
        Where it is.  Scalar; fixed in ICRS and transformed to AltAz for
        each observation.
    spectrum : SpectralModel or array-like, optional
        The source's spectrum on ``geo.wvls``, in
        ``photon / s / m^2 / nm``.  An array is wrapped as a
        :class:`~nyx.core.spectral.StoredSpectrum`.  Defaults to a 9000 K
        black body at *mag*.
    mag : float
        Magnitude of the default black body; ignored if *spectrum* is given.
    temperature_k : float
        Temperature of the default black body.
    nobs : int, optional
        Required with *variable*: :meth:`~nyx.core.scene.Scene.build` asks
        for a source's model before preparing it, so the light curve
        cannot size itself.  Pass ``obs.nobs``; checked at prepare time.
    variable : bool
        Give the source a free per-observation ``brightness``, log-
        transformed so it cannot go negative::

            nova = PointSource(geo, coord, mag=9.0, variable=True, nobs=obs.nobs)
            scene = unfreeze(freeze_all(scene), 'nova.brightness')
    inscatter : bool
        Compute the source's own scattered halo.  Worth it for the Moon,
        wasted on a faint star.
    """

    def __init__(
        self,
        geo: Any,
        coord: SkyCoord,
        spectrum: SpectralModel | Any = None,
        *,
        mag: float = 9.0,
        temperature_k: float = 9000.0,
        nobs: int | None = None,
        variable: bool = False,
        inscatter: bool = False,
    ) -> None:
        if not coord.isscalar:
            raise TypeError(
                f"PointSource takes one position; got a SkyCoord of shape {coord.shape}. "
                f"Use one PointSource per source, or an emitter with a catalog."
            )
        if spectrum is None:
            spectrum = blackbody_photon_flux(geo.wvls, temperature_k, mag)
        if not isinstance(spectrum, SpectralModel):
            values = jnp.atleast_2d(jnp.asarray(spectrum))
            n_wvl = int(np.size(np.asarray(geo.wvls)))
            if values.shape != (1, n_wvl):
                raise ValueError(
                    f"spectrum must be one spectrum on geo.wvls ({n_wvl} points), "
                    f"got shape {tuple(jnp.shape(spectrum))}"
                )
            spectrum = StoredSpectrum(spectra=values)

        if variable and nobs is None:
            raise ValueError(
                "a variable PointSource needs nobs: its brightness is one free "
                "value per frame, and Scene.build asks for a source's model "
                "before preparing it against an observation, so the length "
                "cannot be inferred. Pass nobs=obs.nobs."
            )

        self._coord = coord
        self._spectral_model = spectrum
        self._variable = bool(variable)
        self._inscatter = bool(inscatter)
        self._nobs = None if nobs is None else int(nobs)
        self._geo_signature = geo.signature

    def model(self) -> SourceModel:
        """The shared source model, with a light curve if one was asked for."""
        if not self._variable:
            return SourceModel(spectral_model=self._spectral_model)
        return VariableSource(
            spectral_model=self._spectral_model,
            brightness=Parameter.from_value(
                jnp.ones(self._nobs), scale=1.0, per_obs=True, transform="log"
            ),
        )

    def prepare(self, obs: Any) -> SourceObsData:
        """Transform the position into each observation's AltAz frame."""
        if self._variable and obs.nobs != self._nobs:
            raise ValueError(
                f"this PointSource was built for a {self._nobs}-frame light curve "
                f"but the observation has {obs.nobs} frames; rebuild it with "
                f"nobs={obs.nobs}"
            )
        altaz = [self._coord.transform_to(frame) for frame in obs.altaz_frames]
        coords = np.stack([[[c.az.rad, c.alt.rad]] for c in altaz])  # (nobs, 1, 2)
        return SourceObsData(
            source_coords=jnp.asarray(coords),
            inscatter=self._inscatter,
            per_obs=("source_coords",),
        )

    def __repr__(self) -> str:
        icrs = self._coord.icrs
        kind = "variable" if self._variable else "fixed"
        return (
            f"PointSource(ra={icrs.ra.deg:.4f} dec={icrs.dec.deg:.4f} deg, {kind}"
            f"{', inscatter' if self._inscatter else ''}, "
            f"spectrum={type(self._spectral_model).__name__})"
        )
