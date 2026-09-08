from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import jax.numpy as jnp
import numpy as np

from nyx.core.parameter import Parameter
from nyx.core.protocols import SkySource
from nyx.core.records import SourceObsData
from nyx.emitter.sources import SpectralSource
from nyx.utils.spectra import SpectralModel

__all__ = ["Emitter"]


def _as_brightness(
    brightness: Any, n_wvl: int, transform: str | None
) -> tuple[Parameter | None, int | None]:
    """Validate a brightness and wrap it as a :class:`Parameter`.

    Parameters
    ----------
    brightness : array-like or None
        Scalar, ``(nobs,)`` light curve, or ``(nobs, n_wvl)`` chromatic curve.
    n_wvl : int
        Length of ``geo.wvls``, which a chromatic curve must match.
    transform : str or None
        Domain of the parameter; see :class:`~nyx.core.parameter.Parameter`.

    Returns
    -------
    parameter : Parameter or None
        ``None`` if *brightness* was ``None``.
    nobs : int or None
        Frame count the curve commits to, or ``None`` for a scalar.

    Raises
    ------
    ValueError
        If *brightness* has too many axes, or a chromatic curve is not on
        ``geo.wvls``.
    """
    if brightness is None:
        return None, None
    b = jnp.asarray(brightness, dtype=float)
    if b.ndim > 2:
        raise ValueError(
            f"brightness must be a scalar, (nobs,) or (nobs, n_wvl); got shape {b.shape}"
        )
    if b.ndim == 2 and b.shape[1] != n_wvl:
        raise ValueError(
            f"a chromatic brightness is (nobs, n_wvl) on geo.wvls ({n_wvl} points), "
            f"got shape {b.shape}"
        )
    parameter = Parameter.from_value(b, per_obs=b.ndim > 0, transform=transform)
    return parameter, (int(b.shape[0]) if b.ndim > 0 else None)


class Emitter(ABC):
    """Shared base for :class:`~nyx.core.protocols.EmitterLike` implementations.

    Parameters
    ----------
    geo : Geometry
        Resolution configuration.  Its signature is recorded here, which is
        what lets :meth:`~nyx.core.scene.Scene.build` catch a component built
        against a different one.
    spectral_model : SpectralModel
        Maps conditions to spectra.
    brightness : array-like or None
        A fittable multiplier on everything this source emits.  ``None`` gives
        the source no amplitude of its own.
    transform : str or None
        Domain of *brightness*, ``'log'`` by default.  Zero is a fixed point
        of that transform, so a curve reaching zero wants ``None``.
    """

    def __init__(
        self,
        geo: Any,
        spectral_model: SpectralModel,
        brightness: Any = None,
        transform: str | None = "log",
    ) -> None:
        self._geo = geo
        self._geo_signature = geo.signature
        self._spectral_model = spectral_model
        self._brightness, self._nobs = _as_brightness(
            brightness, int(np.size(np.asarray(geo.wvls))), transform
        )

    def prepare(self, obs: Any) -> SourceObsData:
        """Precompute per-observation data for the render loop.

        Subclasses implement :meth:`_prepare`; this wrapper is what makes the
        frame-count check impossible to forget.

        Parameters
        ----------
        obs : Observation

        Returns
        -------
        SourceObsData
        """
        self._check_nobs(obs)
        return self._prepare(obs)

    @abstractmethod
    def _prepare(self, obs: Any) -> SourceObsData:
        """Per-observation data for the render loop; see :meth:`prepare`."""

    def _check_nobs(self, obs: Any) -> None:
        """Raise if a per-observation brightness does not match *obs*.

        Raises
        ------
        ValueError
            If the frame counts disagree.
        """
        if self._nobs is not None and obs.nobs != self._nobs:
            raise ValueError(
                f"this {type(self).__name__} was built with a {self._nobs}-frame "
                f"brightness but the observation has {obs.nobs} frames; rebuild "
                f"it with a brightness of shape ({obs.nobs},) -- or a scalar, "
                f"for one brightness over the whole run"
            )

    def model(self) -> SkySource:
        """The runtime pytree for this emitter.

        Override to return a different :class:`~nyx.core.protocols.SkySource`
        when a source needs physics :class:`~nyx.emitter.sources.SpectralSource`
        does not express.

        Returns
        -------
        SkySource
        """
        return SpectralSource(self._spectral_model, self._brightness)

    def _repr_parts(self) -> list[str]:
        """Fragments describing this emitter's own state, for :meth:`__repr__`.

        Returns
        -------
        list of str
        """
        return []

    def __repr__(self) -> str:
        """Name the emitter, whatever it reports, and its spectrum."""
        bits = [*self._repr_parts(), f"spectrum={type(self._spectral_model).__name__}"]
        return f"{type(self).__name__}({', '.join(bits)})"
