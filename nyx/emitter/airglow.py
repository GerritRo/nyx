from nyx import ASSETS_PATH

import jax.numpy as jnp
import numpy as np
import healpy as hp
import astropy.units as u

from nyx.core.scene import ComponentType
from nyx.core import Spectrum
from nyx.core import get_wavelengths, get_healpix_nside
from nyx.core import DiffuseQuery, ParameterSpec
from nyx.core.model import EmitterProtocol

from nyx.units import Radiance

class ESOSkyCalc(EmitterProtocol):
    def __init__(self):
        """
        Parameters
        ----------
        """
        ag_array = np.genfromtxt(ASSETS_PATH+'eso_skycalc_airglow_130sfu.dat')
        self.wvl = ag_array[:,0]*u.nm
        self.flx = ag_array[:,1]*u.ph/u.s/u.m**2/u.micron/u.arcsec**2
    
    def get_generator(self, observation):
        # Load relevant global parameters:
        wavelengths = get_wavelengths()
        nside = get_healpix_nside()
    
        # Pre-compute hemisphere grid:
        npix = hp.nside2npix(nside)
        theta, phi = hp.pix2ang(nside, np.arange(npix))
        mask = theta < np.pi/2  # Upper hemisphere

        # Write values to jax array:
        Z_vals = jnp.array(theta[mask])
        height = jnp.array(observation.AltAz.location.height.to(u.km).value)

        # Resample flux to wavelengths:
        flx = Spectrum.from_arrays(self.wvl, Radiance(self.flx).value).resample(wavelengths).flux
        
        def generator(params):
            # Scaling with solar flux
            sfu_val = (0.2 + 0.00614 * params['sfu'])
            # Scaling for zenith with van rhjin function
            weight = 1 / (1 - (6738 / (6738 + height))**2 * jnp.sin(Z_vals) ** 2) ** 0.5
            return DiffuseQuery(flux_map=sfu_val*weight[:,None]*flx[None,:])

        param_specs = {
            'sfu': ParameterSpec((1,), 100, description="Solar flux value [SFU]"),
        }

        return generator, param_specs, ComponentType.DIFFUSE