from nsb2 import ASSETS_PATH

import jax.numpy as jnp
import numpy as np
import healpy as hp
import astropy.units as u

from nsb3.core import get_wavelengths, get_healpix_nside
from nsb3.core import DiffuseQuery, ParameterSpec
from nsb3.core.model import EmitterProtocol

class ESOSkyCalc(EmitterProtocol):
    def __init__(self):
        """
        Parameters
        ----------
        """
        ag_array = np.genfromtxt(ASSETS_PATH+'eso_skycalc_airglow_130sfu.dat')
        self.wvl = ag_array[:,0]*u.nm
        self.flx = ag_array[:,1]/u.s/u.m**2/u.micron/u.arcsec**2
    
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
        flx = jnp.array(self.flx.value)
        
        def generator(params):
            # Scaling with solar flux
            sfu_val = (0.2 + 0.00614 * params['sfu'])
            # Scaling for zenith with van rhjin function
            weight = 1 / (1 - (6738 / (6738 + params['obs_height_km']))**2 * jnp.sin(Z_vals) ** 2) ** 0.5
            return DiffuseQuery(flux_map=sfu_val*weight[:,None]*flx[None,:])

        param_specs = {
            'obs_height_km': ParameterSpec((1,), 1.0, description="Observatory height [km]"),
            'sfu': ParameterSpec((1,), 100, description="Solar flux value [SFU]"),
        }

        return generator, param_specs