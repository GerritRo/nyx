from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp

from nyx.core.filters import tile_per_obs
from nyx.core.parameter import Parameter
from nyx.core.records import AtmosphereResult, PointSourceData, SkyGeometry, SourceObsData

if TYPE_CHECKING:
    from nyx.core.observation import Observation

__all__ = [
    "AtmosphereModel",
    "EmitterLike",
    "InstrumentModel",
    "SkySource",
]


class SkySource(eqx.Module):
    """Source model: the shared physics mapping conditions to photons.

    Stored at the :class:`~nyx.core.scene.Scene` level and holds the
    trainable spectral parameters.
    """

    def diffuse_radiance(self, obs_data: SourceObsData | None = None) -> jax.Array | None:
        """Diffuse radiance at the hemisphere positions.

        Parameters
        ----------
        obs_data : SourceObsData or None

        Returns
        -------
        jax.Array, shape (..., n_wvl), or None
            ``None`` if this source has no diffuse component.
        """
        return None

    def point_sources(self, obs_data: SourceObsData | None = None) -> PointSourceData | None:
        """Point sources.

        Parameters
        ----------
        obs_data : SourceObsData or None

        Returns
        -------
        PointSourceData or None
            ``None`` if this source has no point component.
        """
        return None


@runtime_checkable
class EmitterLike(Protocol):
    """Protocol for emitter builders, consumed by :meth:`Scene.build`.

    ``model()`` returns the shared :class:`SkySource` stored in the scene
    pytree; ``prepare(obs)`` the per-observation
    :class:`~nyx.core.records.SourceObsData`.
    """

    def model(self) -> SkySource: ...

    def prepare(self, obs: Observation) -> SourceObsData: ...


class AtmosphereModel(eqx.Module):
    """Extinction and scattering, shared across every instrument in a scene.

    Trainable quantities are the :class:`~nyx.core.parameter.Parameter`
    fields a subclass declares.
    """

    def evaluate(self, sky: SkyGeometry) -> AtmosphereResult:
        """Extinction and scattering for the current trainable parameters.

        Parameters
        ----------
        sky : SkyGeometry

        Returns
        -------
        AtmosphereResult
        """
        raise NotImplementedError

    def extinct(self, altitudes: jax.Array, spectra: jax.Array, height_km: jax.Array) -> jax.Array:
        """Extinct point-source spectra; the default returns them unchanged.

        Parameters
        ----------
        altitudes : jax.Array, shape (n_sources, 1)
            Radians.
        spectra : jax.Array, shape (n_sources, n_wvl)
        height_km : jax.Array
            Observer height above sea level.

        Returns
        -------
        jax.Array, shape (n_sources, n_wvl)
        """
        return spectra

    def scatter_sources(
        self,
        sky: SkyGeometry,
        source_coords: jax.Array,
        source_spectra: jax.Array,
        bp: jax.Array,
    ) -> jax.Array:
        """Scatter discrete point sources into the FOV; the default is none.

        Parameters
        ----------
        sky : SkyGeometry
        source_coords : jax.Array, shape (n_sources, 2)
            AltAz in radians.
        source_spectra : jax.Array, shape (n_sources, n_wvl)
        bp : jax.Array, shape (n_wvl,)
            Instrument bandpass.

        Returns
        -------
        jax.Array, shape (n_lon, n_lat), or a scalar zero
        """
        return jnp.array(0.0)


class InstrumentModel(eqx.Module):
    """Sensor that maps sky radiance to detector counts.

    Trainable quantities are the :class:`~nyx.core.parameter.Parameter`
    fields a subclass declares, with ``per_obs=True`` for those carrying an
    observation axis.
    """

    efficiency: eqx.AbstractVar[Parameter]

    def prepare(self, obs: Observation) -> InstrumentModel:
        """Batch for multi-observation rendering; the default tiles ``per_obs`` fields.

        Parameters
        ----------
        obs : Observation

        Returns
        -------
        InstrumentModel
        """
        return tile_per_obs(self, obs.nobs)

    @property
    def bandpass(self) -> jax.Array:
        """Spectral transmission curve, shape ``(n_wvl,)``, excluding efficiency."""
        raise NotImplementedError

    def corrected_pm(self, pm: jax.Array) -> jax.Array:
        """Pointing matrix corrected for detector shift and rotation.

        Parameters
        ----------
        pm : jax.Array, shape (3, 3)

        Returns
        -------
        jax.Array, shape (3, 3)
        """
        return pm

    def project_scattered(self, eval_grid_values: jax.Array) -> jax.Array:
        """Scattering eval grid to pixel rates; the default is no contribution.

        Parameters
        ----------
        eval_grid_values : jax.Array, shape (n_lon, n_lat)

        Returns
        -------
        jax.Array, shape (n_pixels,), or scalar
        """
        return jnp.array(0.0)

    def project_diffuse(self, hp_values: jax.Array, pm: jax.Array) -> jax.Array:
        """HEALPix sky values to pixel rates.

        Parameters
        ----------
        hp_values : jax.Array, shape (npix,)
        pm : jax.Array, shape (3, 3)

        Returns
        -------
        jax.Array, shape (n_pixels,)
        """
        raise NotImplementedError

    def project_catalog(self, source_coords: jax.Array, source_fluxes: jax.Array) -> jax.Array:
        """Point sources to pixel rates; the default is no contribution.

        Parameters
        ----------
        source_coords : jax.Array, shape (n_sources, 2)
        source_fluxes : jax.Array, shape (n_sources,)

        Returns
        -------
        jax.Array, shape (n_pixels,), or scalar
        """
        return jnp.array(0.0)
