from __future__ import annotations

import healpy as hp
import numpy as np

from nyx.emitter.catalogs.astrometry import angular_separation_deg

__all__ = ["HEALPixIndex"]


class HEALPixIndex:
    """Cone search over ra/dec, indexed by nested-ordering HEALPix pixel.

    Parameters
    ----------
    ra, dec : array-like
        Catalog positions in degrees.
    nside : int
        HEALPix nside of the index.
    """

    def __init__(self, ra: np.ndarray, dec: np.ndarray, nside: int = 256):
        """Build the index; see the class docstring for the arguments."""
        self.ra = np.asarray(ra, dtype=np.float64)
        self.dec = np.asarray(dec, dtype=np.float64)
        self.nside = nside

        theta = np.radians(90.0 - self.dec)
        phi = np.radians(self.ra)

        pix = hp.ang2pix(nside, theta, phi, nest=True)

        # Sorted by pixel so a disc's pixels are contiguous slices.
        order = np.argsort(pix)
        self._sorted_pix = pix[order]
        self._sorted_idx = order

    def query(self, ra: float, dec: float, radius: float) -> np.ndarray:
        """Rows within *radius* of ``(ra, dec)``.

        Parameters
        ----------
        ra, dec : float
            Cone centre in degrees.
        radius : float
            Search radius in degrees.

        Returns
        -------
        numpy.ndarray
            Integer indices into the original catalog.
        """
        theta_c = np.radians(90.0 - dec)
        phi_c = np.radians(ra)
        vec = hp.ang2vec(theta_c, phi_c)
        rad = np.radians(radius)

        candidate_pix = hp.query_disc(self.nside, vec, rad, nest=True, inclusive=True)

        lefts = np.searchsorted(self._sorted_pix, candidate_pix, side="left")
        rights = np.searchsorted(self._sorted_pix, candidate_pix, side="right")
        idx_lists = [
            self._sorted_idx[left:right]
            for left, right in zip(lefts, rights, strict=False)
            if left < right
        ]
        if not idx_lists:
            return np.array([], dtype=np.int64)
        return np.concatenate(idx_lists)

    def nearest(self, ra: float, dec: float, radius: float) -> tuple[int, float]:
        """Closest catalog row to ``(ra, dec)``, if one is within *radius*.

        Parameters
        ----------
        ra, dec : float
            Query position in degrees.
        radius : float
            Match radius in degrees.

        Returns
        -------
        index : int
            Row index into the original catalog, or ``-1`` for no match.
        separation : float
            Separation in degrees.
        """
        idx = self.query(ra, dec, radius)
        if len(idx) == 0:
            idx = self.query(ra, dec, max(10.0 * radius, 1.0))
        if len(idx) == 0:
            return -1, float("inf")

        seps = angular_separation_deg(ra, dec, self.ra[idx], self.dec[idx])
        j = int(np.argmin(seps))
        if seps[j] > radius:
            return -1, float(seps[j])
        return int(idx[j]), float(seps[j])
