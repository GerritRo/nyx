"""Profile-likelihood scans."""

from __future__ import annotations

import dataclasses

import numpy as np

#: Delta chi-squared at 1, 2 and 3 sigma, by number of scanned parameters.
_SIGMA_LEVELS: dict[int, tuple[float, float, float]] = {
    1: (1.00, 4.00, 9.00),
    2: (2.30, 6.18, 11.83),
}


@dataclasses.dataclass(frozen=True)
class ProfileGrid:
    """A profile-likelihood scan: chi-squared over a grid, nuisances refitted.

    Indexed in grid-axis order: ``chi2[i, j]`` is at
    ``axes[0][i], axes[1][j]``.
    """

    names: list[str]
    axes: list[np.ndarray]
    chi2: np.ndarray

    @property
    def delta_chi2(self) -> np.ndarray:
        """Chi-squared relative to the best point on the grid."""
        return self.chi2 - float(np.nanmin(self.chi2))

    @property
    def best(self) -> dict[str, float]:
        """Where on the grid the chi-squared is lowest."""
        index = np.unravel_index(int(np.nanargmin(self.chi2)), self.chi2.shape)
        return {n: float(a[i]) for n, a, i in zip(self.names, self.axes, index, strict=True)}

    @property
    def levels(self) -> tuple[float, float, float]:
        """Delta chi-squared at 1, 2 and 3 sigma, for a contour call."""
        try:
            return _SIGMA_LEVELS[len(self.names)]
        except KeyError:
            raise ValueError(
                f"sigma contours are tabulated for 1 or 2 scanned parameters, not {len(self.names)}"
            ) from None

    def __repr__(self) -> str:
        span = ", ".join(
            f"{n} in [{a.min():.4g}, {a.max():.4g}] ({a.size})"
            for n, a in zip(self.names, self.axes, strict=True)
        )
        best = ", ".join(f"{k}={v:.6g}" for k, v in self.best.items())
        return (
            f"ProfileGrid({span})\n"
            f"  best chi2 {float(np.nanmin(self.chi2)):.6g} at {best}\n"
            f"  1/2/3 sigma at delta chi2 {self.levels}"
        )
