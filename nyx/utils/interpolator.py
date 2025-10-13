import numpy as np
import astropy.units as u

from scipy.interpolate import RegularGridInterpolator

class UnitRegularGridInterpolator:
    def __init__(self, points, values, threshold=3, unit=None, **kwargs):
        """
        points: list of ndarrays (grid axes for RegularGridInterpolator)
        values: ndarray with astropy Quantity (values on the grid)
        threshold: log10 range above which log interpolation is used
        unit: optionally convert output to this unit
        kwargs: passed to RegularGridInterpolator
        """
        if not hasattr(values, "unit"):
            raise ValueError("`values` must be an astropy Quantity.")

        self.unit = unit or values.unit
        self.log_values = False
        self.original_unit = values.unit

        val_data = values.to_value(self.unit)

        # Decide if we want to interpolate in log space
        min_val = np.nanmin(val_data)
        max_val = np.nanmax(val_data)

        if min_val > 0 and np.log10(max_val / min_val) > threshold:
            self.log_values = True
            val_data = np.log10(val_data)

        self.interpolator = RegularGridInterpolator(points, val_data, **kwargs)

    def __call__(self, xi):
        """
        Evaluate the interpolator at locations xi (no units).

        xi: array-like shape (..., ndim), must match the number of grid axes.
        Returns: Quantity with same unit as self.unit
        """
        xi = np.atleast_2d(xi)
        result = self.interpolator(xi)

        if self.log_values:
            result = 10 ** result

        return result * self.unit
