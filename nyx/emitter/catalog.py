from __future__ import annotations

from typing import Any

import astropy.units as u
import jax.numpy as jnp
import numpy as np
from astropy.coordinates import SkyCoord

from nyx.core.geometry import Geometry
from nyx.emitter.base import Emitter
from nyx.emitter.catalogs.index import HEALPixIndex
from nyx.emitter.point_source import PointSource
from nyx.spectra import SpectralModel

__all__ = ["CatalogEmitter"]

#: Default match radius for :meth:`Stars.pop`, in arcseconds.
_POP_RADIUS = 10 * u.arcsec


class CatalogEmitter(Emitter):
    """A star catalog.

    Subclasses supply :meth:`_pop_conditions` and, if their stars carry their
    own scattered halo, ``_pop_inscatter``.

    Parameters
    ----------
    geo : Geometry
    spectral_model : SpectralModel
    coords : astropy.coordinates.SkyCoord
        Catalog positions.
    brightness, transform
        See :class:`~nyx.emitter.base.Emitter`.
    """

    # Whether a star popped out of this catalog computes its own halo.
    _pop_inscatter = False

    def __init__(
        self,
        geo: Geometry,
        spectral_model: SpectralModel,
        coords: SkyCoord,
        brightness: Any = None,
        transform: str | None = "log",
    ) -> None:
        super().__init__(geo, spectral_model, brightness, transform)
        self._coords = coords
        self._taken = np.zeros(len(coords), dtype=bool)
        self._hpx_index: HEALPixIndex | None = None

    @property
    def _index(self) -> HEALPixIndex:
        """Cone-search index over the catalog, built on first use."""
        if self._hpx_index is None:
            self._hpx_index = HEALPixIndex(self._coords.ra.deg, self._coords.dec.deg)
        return self._hpx_index

    def _pop_conditions(self, idx: int) -> np.ndarray:
        """Conditions row necessary for popped star.

        Parameters
        ----------
        idx : int
            Row of the catalog.

        Returns
        -------
        numpy.ndarray, shape (n_cond,)
        """
        raise NotImplementedError

    def _match_row(self, coord: SkyCoord, radius: u.Quantity | float) -> int:
        """Row matching *coord*, refusing an ambiguous or repeat match.

        Parameters
        ----------
        coord : astropy.coordinates.SkyCoord
            Scalar query position.
        radius : astropy.units.Quantity or float
            Match radius; a bare number is degrees.

        Returns
        -------
        int

        Raises
        ------
        TypeError
            If *coord* is not scalar.
        LookupError
            If no star lies within *radius*, or the match was already popped.
        """
        if not coord.isscalar:
            raise TypeError(f"pop takes one position; got a SkyCoord of shape {coord.shape}")
        icrs = coord.icrs
        ra, dec = float(icrs.ra.deg), float(icrs.dec.deg)
        radius_deg = (
            float(radius.to_value(u.deg)) if isinstance(radius, u.Quantity) else float(radius)
        )

        idx, sep = self._index.nearest(ra, dec, radius_deg)
        if idx < 0:
            how_close = (
                "no star within a degree of it"
                if not np.isfinite(sep)
                else f"the nearest is {sep * 3600:.1f} arcsec away"
            )
            raise LookupError(
                f"no catalog star within {radius_deg * 3600:.1f} arcsec of ra={ra:.5f} "
                f"dec={dec:.5f} deg -- {how_close}. Widen radius=, or check the star is "
                f"in this catalog and bright enough to be resolved by it. Positions are "
                f"matched at the catalog epoch, so a high proper motion star needs room."
            )
        if self._taken[idx]:
            raise LookupError(
                f"the catalog star {sep * 3600:.1f} arcsec from ra={ra:.5f} dec={dec:.5f} deg "
                f"has already been popped; a star can only be taken out of a catalog once."
            )
        return int(idx)

    def pop(
        self, coord: SkyCoord, *, radius: u.Quantity | float = _POP_RADIUS, **kwargs: Any
    ) -> PointSource:
        """Take one star out of the catalog, as a standalone point source.

        The star leaves this catalog's point-source list and comes back as a
        :class:`~nyx.emitter.point_source.PointSource` with its catalog
        position, its catalog spectrum, and a free ``brightness``.

        Parameters
        ----------
        coord : astropy.coordinates.SkyCoord
            Scalar position to match, at the catalog's own epoch.
        radius : astropy.units.Quantity or float
            Match radius, 10 arcsec by default; a bare number is degrees.
        **kwargs
            Passed to :class:`~nyx.emitter.point_source.PointSource`.

        Returns
        -------
        PointSource

        Raises
        ------
        LookupError
            If no star lies within *radius*, or the match was already popped.

        Notes
        -----
        This mutates the catalog: the star is gone from it afterwards, and a
        scene built from it earlier still holds the old, complete one.
        """
        idx = self._match_row(coord, radius)
        self._taken[idx] = True
        spectrum = self._spectral_model(jnp.asarray(self._pop_conditions(idx))[None, :])
        kwargs.setdefault("inscatter", self._pop_inscatter)
        return PointSource(self._geo, self._coords[idx], spectrum, **kwargs)
