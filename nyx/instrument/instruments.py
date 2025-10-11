import jax.numpy as jnp
import numpy as np
import healpy as hp
from scipy.integrate import simpson as simps
import astropy.units as u
from astropy.coordinates import SkyCoord

from nyx.core import get_wavelengths, get_healpix_nside
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
        nside = get_healpix_nside()
        
        # Determine the healpix interpolation:
        c_coords = SkyCoord(self.centers.copy(), unit='rad', frame=observation.frame).transform_to(observation.AltAz)
        hp_pixel, hp_weight = hp.get_interp_weights(nside, np.pi/2 - c_coords.alt.rad, c_coords.az.rad)

        # Evaluate bandpass function:
        bp_values = self.bandpass(wavelengths*u.nm) * np.diff(wavelengths).mean()

        # Convert to jax arrays:
        centers = jnp.array(self.centers)
        grid = jnp.array(self.grid)
        weight = jnp.array(self.weight)
        pixel_values = jnp.array(self.values)
        hp_pixel = jnp.array(hp_pixel)
        hp_weight = jnp.array(hp_weight * self.weight)
        bp_values = jnp.array(bp_values)
        
        def generator(params):
            return InstrumentQuery(
                centers=centers + params['shift'],
                hp_pixels=hp_pixel,
                hp_weight=hp_weight,
                weight=weight,
                grid=grid + params['shift'][None,:,None],
                values=pixel_values * params['flatfield'][:,None,None],
                bandpass=bp_values
            )

        param_specs={
            'shift': ParameterSpec((2,), jnp.zeros(2), description="Pixel shift in rad"),
            'flatfield': ParameterSpec((960,), jnp.ones(960), description="Flatfielding values"),
        }

        return generator, param_specs