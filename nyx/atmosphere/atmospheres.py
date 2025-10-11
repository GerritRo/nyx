import jax
import jax.numpy as jnp
import numpy as np
import healpy as hp

from astropy.coordinates import SkyCoord
from nyx.core.model import AtmosphereProtocol
from nyx.core import get_wavelengths, get_healpix_nside
from nyx.core import AtmosphereQuery, ParameterSpec
from nyx.atmosphere import get_airmass_formula
from nyx.atmosphere.scattering import rayleigh_phase, henyey_greenstein_phase, gradation_function

class SingleScatteringAtmosphere(AtmosphereProtocol):
    """
    Single scattering atmosphere model
    """
    def __init__(self, 
                 tau_rayleigh_func: callable,
                 tau_mie_func: callable,
                 tau_absorption_func: callable = None,
                 hg_asymmetry: float = 0.75):
        """
        Parameters
        ----------
        tau_rayleigh_func : Function (wavelength[nm], height[km]) -> optical depth
        tau_mie_func : Function (wavelength[nm], height[km], aod, angstrom) -> optical depth
        tau_absorption_func : Function (wavelength[nm]) -> optical depth
        hg_asymmetry : Henyey-Greenstein asymmetry parameter
        """
        self.tau_rayleigh_func = tau_rayleigh_func
        self.tau_mie_func = tau_mie_func
        self.tau_absorption_func = tau_absorption_func or (lambda wl: np.zeros_like(wl))
        self.hg_asymmetry = hg_asymmetry
        
    def get_generator(self, observation):
        """
        Create JAX generator function
        Fixed units: wavelengths in nm
        """
        # Load relevant global parameters:
        wavelengths = get_wavelengths()
        nside = get_healpix_nside()
        
        # Pre-compute hemisphere grid:
        npix = hp.nside2npix(nside)
        theta, phi = hp.pix2ang(nside, np.arange(npix))
        mask = theta < np.pi/2  # Upper hemisphere
        hp_coords = SkyCoord(phi[mask], np.pi/2-theta[mask], unit='rad', frame=observation.AltAz)

        # Get airmass formula:
        airmass = get_airmass_formula()

        # Get observation coordinates:
        obs_coords = observation.get_eval_coordinates()
        scat_angle = jnp.array(obs_coords[:,:,None].separation(hp_coords[None,None,:]).rad)
        sec_Z_grid = jnp.array(airmass(np.pi/2 - obs_coords.alt.rad))
        sec_z = jnp.array(airmass(jnp.array(theta[mask])))

        YiXi = jnp.array(observation.get_eval_coordinates(altaz=False))

        # Get scattering correction for hp area:
        scat_corr = jnp.array(hp.nside2pixarea(nside))
        
        def generator(params):
            # Calculate optical depths (all dimensionless)
            tau_r = self.tau_rayleigh_func(wavelengths, params['obs_height_km'])
            tau_m = self.tau_mie_func(wavelengths, params['obs_height_km'], 
                                     params['aod_500'], params['angstrom_exp'])
            tau_a = self.tau_absorption_func(wavelengths)
            
            tau_total = tau_r + tau_m + tau_a
            
            # Scattering phase functions
            p_rayleigh = rayleigh_phase(scat_angle)
            p_mie = henyey_greenstein_phase(scat_angle, params['hg_asymmetry'])
            
            # Weight by scattering optical depths
            indicatrix = ((tau_r[None,:] * p_rayleigh[...,None] + 
                           tau_m[None,:] * p_mie[...,None]) / 
                           jnp.maximum(tau_total, 1e-10))
            
            # Gradation function
            gradation = gradation_function(tau_total[None,None,None,:],
                                           sec_Z_grid[:,:,None,None], 
                                           sec_z[None,None,:,None])
            
            # Extinction along line of sight
            extinction = jnp.exp(-tau_total[None,:] * sec_z[:,None])
            
            return AtmosphereQuery(
                YiXi=[YiXi, YiXi],
                tau=tau_total,
                extinction=extinction,
                scattering=indicatrix * gradation * scat_corr
            )
        
        param_specs = {
            'obs_height_km': ParameterSpec((1,), 0.0, description="Observatory height [km]"),
            'aod_500': ParameterSpec((1,), 0.1, description="AOD at 500nm"),
            'angstrom_exp': ParameterSpec((1,), 1.0, description="Angstrom exponent"),
            'hg_asymmetry': ParameterSpec((1,), self.hg_asymmetry, 
                                         description="HG asymmetry", bounds=(0, 1))
        }
        
        return generator, param_specs