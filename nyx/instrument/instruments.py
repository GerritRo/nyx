import jax.numpy as jnp
import numpy as np
from scipy.integrate import simpson as simps
import astropy.units as u
from astropy.coordinates import SkyCoord

from nyx.core import get_wavelengths
from nyx.core import InstrumentQuery, ParameterSpec
from nyx.core.model import InstrumentProtocol

class EffectiveApertureInstrument(InstrumentProtocol):
    """
    Effective aperture representation of instrument
    """
    def __init__(self, bandpass, grid, values):
        """
        Parameters
        ----------
        """
        self.bandpass = bandpass
        self.grid = grid
        self.values = values

        self.centers = np.mean(grid, axis=2)
        self.weight = np.asarray([simps(simps(values[i], grid[i][0]), grid[i][1]) for i in range(len(grid))])

    def get_generator(self, observation):
        """
        Create JAX generator function
        """
        # Load relevant global parameters:
        wavelengths = get_wavelengths()

        # Compute base AltAz coordinates (before any shift)
        c_coords = SkyCoord(self.centers, unit='rad', frame=observation.frame).transform_to(observation.AltAz)
        base_theta = jnp.array(np.pi/2 - c_coords.alt.rad)  # colatitude
        base_phi = jnp.array(c_coords.az.rad)               # azimuth

        # Evaluate bandpass function:
        bp_values = self.bandpass(wavelengths*u.nm) * np.diff(wavelengths).mean()

        # Convert to jax arrays:
        centers = jnp.array(self.centers)
        grid = jnp.array(self.grid)
        weight = jnp.array(self.weight)
        pixel_values = jnp.array(self.values)
        bp_values = jnp.array(bp_values)

        def generator(params):
            shifted_theta = base_theta - params['shift'][1]  # altitude shift (negative because theta = pi/2 - alt)
            shifted_phi = base_phi + params['shift'][0]      # azimuth shift

            return InstrumentQuery(
                centers=centers + params['shift'],
                hp_theta=shifted_theta,
                hp_phi=shifted_phi,
                weight=weight,
                grid=grid + params['shift'][None,:,None],
                values=pixel_values,
                bandpass=bp_values * params['eff']
            )

        param_specs={
            'eff': ParameterSpec((1,), 1, description="Telescope efficiency"),
            'shift': ParameterSpec((2,), jnp.zeros(2), description="Pixel shift in rad"),
        }

        return generator, param_specs