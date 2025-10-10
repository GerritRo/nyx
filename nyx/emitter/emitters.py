import jax.numpy as jnp
import numpy as np
import healpy as hp
import astropy.units as u
from astropy.coordinates import SkyCoord

from nyx.core import get_wavelengths, get_healpix_nside
from nyx.core import CatalogQuery, DiffuseQuery, ParameterSpec
from nyx.core.model import EmitterProtocol

class DiffuseEmitter(EmitterProtocol):
    def __init__(self, coords, weight, data, spectral_grid):
        """
        Parameters
        ----------
        """
        self.frame = coords.frame
        self.coords = coords

    def get_generator(self, observation):
        def generator(params):
            return DiffuseQuery(flux_map=base_weight[:,None]*(params['continuum']*base_spec + params['440_nm']*line_440)[None,:])



class CatalogEmitter(EmitterProtocol):
    def __init__(self, coords, weight, data, spectral_grid):
        """
        Parameters
        ----------
        """
        self.frame = coords.frame
        self.coords = coords

    def get_generator(self, observation):
        sec_Z = jnp.ones(N_sources)
        image_coords = np.random.uniform(-1, 1, size=(N_sources, 2))
        base_flux_map = np.ones((3072, N_wvl))[latmask]

        def generator(params):
            return CatalogQuery(
                sec_Z=sec_Z,
                image_coords=image_coords,
                flux_values=params['source_fluxes'] * np.ones((N_sources, N_wvl)),
                flux_map=base_flux_map * params['map_scaling']
            )

        
